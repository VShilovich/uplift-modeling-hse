# tests/full_uplift_api_test.py
import requests
import json
import time
import sqlite3
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

DB_PATH = Path("data/uplift-modeling.db")
ADMIN_USERNAME = "myadmin"
ADMIN_PASSWORD = "mypass123"
FORWARD_CALLS = 3


def ensure_first_admin():
    """Создать первого админа myadmin/mypass123 в SQLite, если его ещё нет."""
    print("===> ensure_first_admin: проверяем/создаём базового админа в SQLite")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS admins ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "username TEXT UNIQUE, "
        "password_hash TEXT)"
    )
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM admins WHERE username = ?", (ADMIN_USERNAME,))
    exists = cur.fetchone()[0] > 0

    if not exists:
        cur.execute(
            "INSERT INTO admins (username, password_hash) VALUES (?, ?)",
            (ADMIN_USERNAME, ADMIN_PASSWORD),
        )
        conn.commit()
        print(f"   создан базовый админ: {ADMIN_USERNAME}/{ADMIN_PASSWORD}")
    else:
        print(f"   базовый админ {ADMIN_USERNAME} уже существует")

    cur.execute("SELECT id, username FROM admins")
    print("   текущие админы:", cur.fetchall())
    conn.close()


def print_result(name: str, ok: bool, extra: str = ""):
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {name}")
    if extra:
        print("      ", extra)


def test_open_routes():
    results = {}

    r = requests.get(f"{BASE_URL}/docs")
    results["GET /docs"] = (r.status_code == 200, f"status={r.status_code}")

    payload = {"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD}
    r = requests.post(f"{BASE_URL}/login", json=payload)
    results["POST /login (base admin)"] = (
        r.status_code == 200,
        f"status={r.status_code}, body={r.text[:200]}",
    )

    return results


def login_and_get_token(username: str, password: str):
    payload = {"username": username, "password": password}
    r = requests.post(f"{BASE_URL}/login", json=payload)
    ok = r.status_code == 200
    if not ok:
        return ok, None, f"status={r.status_code}, body={r.text[:200]}"
    data = r.json()
    return True, data["access_token"], ""


def create_test_admins(token: str):
    headers = {"Authorization": f"Bearer {token}"}
    test_admins = [("admin2", "pass456"), ("admin3", "pass789")]
    results = {}

    for username, password in test_admins:
        payload = {"username": username, "password": password}
        r = requests.post(f"{BASE_URL}/admins", json=payload, headers=headers)
        ok = r.status_code == 200
        results[f"POST /admins {username}"] = (
            ok,
            f"status={r.status_code}, body={r.text[:200]}",
        )
    return results


def login_multiple_admins():
    admins = [
        ("myadmin", "mypass123"),
        ("admin2", "pass456"),
        ("admin3", "pass789"),
    ]
    tokens = {}
    results = {}

    for username, password in admins:
        ok, token, extra = login_and_get_token(username, password)
        results[f"POST /login {username}"] = (ok, extra)
        if ok:
            tokens[username] = token

    return tokens, results


def test_forward_from_all_admins(tokens: dict):
    forward_data = {
        "client": [
            {
                "client_id": 123,
                "age": 35,
                "gender": "F",
                "first_issue_date": "2022-01-10",
                "first_redeem_date": None,
            },
            {
                "client_id": 555,
                "age": 42,
                "gender": "M",
                "first_issue_date": "2021-05-15",
                "first_redeem_date": "2021-06-20",
            },
        ],
        "purchases": [
            {
                "client_id": 123,
                "transaction_id": 1,
                "transaction_datetime": "2024-02-01 12:30:00",
                "purchase_sum": 540,
                "store_id": "54a4a11a29",
                "regular_points_received": 20,
                "express_points_received": 0,
                "regular_points_spent": 0,
                "express_points_spent": 0,
                "product_id": "9a80204f78",
                "product_quantity": 2,
                "trn_sum_from_iss": 540,
                "trn_sum_from_red": 0,
            },
            {
                "client_id": 123,
                "transaction_id": 2,
                "transaction_datetime": "2024-02-02 15:45:00",
                "purchase_sum": 1200,
                "store_id": "b2c3d4e5f6",
                "regular_points_received": 48,
                "express_points_received": 10,
                "regular_points_spent": 0,
                "express_points_spent": 0,
                "product_id": "1b2c3d4e5f",
                "product_quantity": 3,
                "trn_sum_from_iss": 1200,
                "trn_sum_from_red": 0,
            },
            {
                "client_id": 555,
                "transaction_id": 1,
                "transaction_datetime": "2024-02-01 12:30:00",
                "purchase_sum": 340,
                "store_id": "54a4a11a29",
                "regular_points_received": 12,
                "express_points_received": 0,
                "regular_points_spent": 5,
                "express_points_spent": 0,
                "product_id": "9a80204f78",
                "product_quantity": 1,
                "trn_sum_from_iss": 340,
                "trn_sum_from_red": 0,
            },
        ],
    }

    results = {}
    total_requests = 0

    for admin_name in tokens.keys():
        for i in range(1, FORWARD_CALLS + 1):
            name = f"POST /forward #{i} from {admin_name}"
            r = requests.post(f"{BASE_URL}/forward", json=forward_data)
            ok = r.status_code == 200
            results[name] = (ok, f"status={r.status_code}, body={r.text[:200]}")
            total_requests += 1
            if i < FORWARD_CALLS:
                time.sleep(0.1)

    results["/forward total_requests"] = (True, f"{total_requests}")
    return results


def test_final_admin_routes(token: str):
    headers = {"Authorization": f"Bearer {token}"}
    results = {}

    r = requests.get(f"{BASE_URL}/history", headers=headers)
    ok = r.status_code == 200
    extra = f"status={r.status_code}"
    if ok:
        hist = r.json()
        extra += f", len={len(hist)}"
    results["GET /history before delete"] = (ok, extra)

    r = requests.get(f"{BASE_URL}/stats", headers=headers)
    ok_stats = r.status_code == 200
    results["GET /stats"] = (ok_stats, f"status={r.status_code}")

    r = requests.delete(f"{BASE_URL}/history", headers=headers)
    ok_del = r.status_code == 200
    results["DELETE /history"] = (ok_del, f"status={r.status_code}, body={r.text[:200]}")

    r = requests.get(f"{BASE_URL}/history", headers=headers)
    ok2 = r.status_code == 200
    extra2 = f"status={r.status_code}"
    if ok2:
        hist2 = r.json()
        extra2 += f", len={len(hist2)}"
    results["GET /history after delete"] = (ok2, extra2)

    return results


def main():
    summary = {}

    print("=== ПОЛНЫЙ ТЕСТ UPLIFT API ===")
    print("Проверяются: /docs, /login, /admins, /forward, /history, /stats, DELETE /history\n")

    # 0. гарантируем базового админа в SQLite
    ensure_first_admin()

    # 1. открытые маршруты
    res = test_open_routes()
    summary.update(res)

    ok, token_main, extra = login_and_get_token(ADMIN_USERNAME, ADMIN_PASSWORD)
    summary["POST /login main admin (for setup)"] = (ok, extra)
    if not ok:
        print("\n!!! Не удалось залогиниться как базовый админ, дальше смысла нет.")
        for k, (okv, extrav) in summary.items():
            print_result(k, okv, extrav)
        return

    res = create_test_admins(token_main)
    summary.update(res)

    tokens, res = login_multiple_admins()
    summary.update(res)

    res = test_forward_from_all_admins(tokens)
    summary.update(res)

    any_token = next(iter(tokens.values()))
    res = test_final_admin_routes(any_token)
    summary.update(res)

    print("\n=== ИТОГОВАЯ СВОДКА ===")
    ok_all = True
    for name, (ok, extra) in summary.items():
        print_result(name, ok, extra)
        if not ok:
            ok_all = False

    print("\n=== РЕЗУЛЬТАТ:", "ВСЕ ОК" if ok_all else "ЕСТЬ ОШИБКИ", "===")


if __name__ == "__main__":
    main()
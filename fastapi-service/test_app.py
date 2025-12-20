"""
Полный тестовый скрипт для проверки всех фич сервиса:

1) создаёт первого админа через sqlite3 (если его ещё нет);
2) шлёт тестовый /forward;
3) логинится как админ и получает JWT;
4) под этим токеном дергает:
   - GET /history
   - GET /stats
   - DELETE /history
   - GET /history (проверка очистки)
"""

import sqlite3
import requests
import json
from pathlib import Path

DB_PATH = Path("data/uplift-modeling.db")
BASE_URL = "http://127.0.0.1:8000"

ADMIN_USERNAME = "myadmin"
ADMIN_PASSWORD = "mypass123"


def ensure_first_admin():
    """Создать первого админа в SQLite, если его ещё нет."""
    print("===> Проверяем / создаём первого админа в SQLite")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS admins (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password_hash TEXT)")
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM admins")
    count = cur.fetchone()[0]

    if count == 0:
        cur.execute(
            "INSERT INTO admins (username, password_hash) VALUES (?, ?)",
            (ADMIN_USERNAME, ADMIN_PASSWORD),
        )
        conn.commit()
        print(f"Создан первый админ: {ADMIN_USERNAME} / {ADMIN_PASSWORD}")
    else:
        print(f"Админы уже есть в БД, всего: {count}")

    cur.execute("SELECT id, username FROM admins")
    print("Текущие админы:", cur.fetchall())

    conn.close()


def call_forward():
    print("\n===> Тестовый запрос /forward")
    data = {
        "client": [
            {
                "client_id": 123,
                "age": 35,
                "gender": "F",
                "first_issue_date": "2022-01-10",
                "first_redeem_date": None,
            }
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
            }
        ],
    }

    resp = requests.post(f"{BASE_URL}/forward", json=data)
    print("Status /forward:", resp.status_code)
    try:
        print("Response /forward:", resp.json())
    except Exception:
        print("Raw response /forward:", resp.text)


def login_and_get_token():
    print("\n===> Логин через /login")
    payload = {"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD}
    resp = requests.post(f"{BASE_URL}/login", json=payload)
    print("Status /login:", resp.status_code)
    try:
        data = resp.json()
        print("Response /login:", data)
    except Exception:
        print("Raw response /login:", resp.text)
        raise

    if resp.status_code != 200:
        raise RuntimeError("Не удалось залогиниться, проверь логин/пароль админа")

    return data["access_token"]


def call_admin_routes(token: str):
    headers = {"Authorization": f"Bearer {token}"}

    print("\n===> GET /history (до очистки)")
    r = requests.get(f"{BASE_URL}/history", headers=headers)
    print("Status /history:", r.status_code)
    try:
        print("Response /history:", json.dumps(r.json(), ensure_ascii=False, indent=2))
    except Exception:
        print("Raw response /history:", r.text)

    print("\n===> GET /stats")
    r = requests.get(f"{BASE_URL}/stats", headers=headers)
    print("Status /stats:", r.status_code)
    try:
        print("Response /stats:", json.dumps(r.json(), ensure_ascii=False, indent=2))
    except Exception:
        print("Raw response /stats:", r.text)

    print("\n===> DELETE /history")
    r = requests.delete(f"{BASE_URL}/history", headers=headers)
    print("Status DELETE /history:", r.status_code)
    try:
        print("Response DELETE /history:", r.json())
    except Exception:
        print("Raw response DELETE /history:", r.text)

    print("\n===> GET /history (после очистки)")
    r = requests.get(f"{BASE_URL}/history", headers=headers)
    print("Status /history:", r.status_code)
    try:
        print("Response /history:", json.dumps(r.json(), ensure_ascii=False, indent=2))
    except Exception:
        print("Raw response /history:", r.text)


if __name__ == "__main__":
    print("=== ТЕСТОВЫЙ СКРИПТ ДЛЯ UPLIFT API ===")
    print("Важно: перед запуском этого скрипта должен быть запущен сервер FastAPI,")
    print("и в окружении должен быть задан JWT_SECRET, например:")
    print('  export JWT_SECRET="supersecret123"\n')

    ensure_first_admin()
    call_forward()
    token = login_and_get_token()
    call_admin_routes(token)
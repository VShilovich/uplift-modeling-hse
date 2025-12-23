import sqlite3
import requests
import json
from pathlib import Path
import time

DB_PATH = Path("data/uplift-modeling.db")
BASE_URL = "http://127.0.0.1:8000"

ADMIN_USERNAME = "myadmin"
ADMIN_PASSWORD = "mypass123"
FORWARD_CALLS = 3


def ensure_first_admin():
    """Создать первого админа в SQLite, если его ещё нет."""
    print("===> Проверяем/создаём первого админа в SQLite")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS admins ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "username TEXT UNIQUE, "
        "password_hash TEXT)"
    )
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


def test_open_routes():
    """Тест открытых роутов (без токена)."""
    print("\n" + "-" * 60)
    print("ТЕСТ ОТКРЫТЫХ РОУТОВ")
    print("-" * 60)

    print("\nGET /docs")
    r = requests.get(f"{BASE_URL}/docs")
    print("Status /docs:", r.status_code)

    print("\nPOST /login (myadmin)")
    payload = {"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD}
    r = requests.post(f"{BASE_URL}/login", json=payload)
    print("Status /login:", r.status_code)
    if r.status_code == 200:
        print("Response /login:", r.json())


def login_and_get_token():
    """Логин myadmin и возврат токена."""
    print("\n===> Логин myadmin (для создания других админов)")
    payload = {"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD}
    resp = requests.post(f"{BASE_URL}/login", json=payload)
    print("Status /login myadmin:", resp.status_code)

    if resp.status_code != 200:
        raise RuntimeError("Не удалось залогиниться как myadmin")

    data = resp.json()
    print("Токен myadmin получен")
    return data["access_token"]


def create_test_admins(token: str):
    """Создать 2 тестовых админа через /admins."""
    print("\n" + "-" * 60)
    print("СОЗДАНИЕ ТЕСТОВЫХ АДМИНОВ")
    print("-" * 60)

    headers = {"Authorization": f"Bearer {token}"}
    test_admins = [
        ("admin2", "pass456"),
        ("admin3", "pass789"),
    ]

    for username, password in test_admins:
        print(f"\nPOST /admins {username}")
        payload = {"username": username, "password": password}
        r = requests.post(f"{BASE_URL}/admins", json=payload, headers=headers)
        print(f"Status /admins {username}:", r.status_code)
        try:
            print("Response:", r.json())
        except Exception:
            print("Raw:", r.text)


def login_multiple_admins():
    """Логин от всех админов."""
    print("\n" + "-" * 60)
    print("ЛОГИН ОТ ВСЕХ АДМИНОВ")
    print("-" * 60)

    admins = [
        ("myadmin", "mypass123"),
        ("admin2", "pass456"),
        ("admin3", "pass789"),
    ]

    tokens = {}
    for username, password in admins:
        print(f"\nЛогин {username}")
        payload = {"username": username, "password": password}
        r = requests.post(f"{BASE_URL}/login", json=payload)
        print(f"Status /login {username}:", r.status_code)
        if r.status_code == 200:
            data = r.json()
            tokens[username] = data["access_token"]
            print(f"Токен получен для {username}")
        else:
            print(f"Ошибка логина для {username}")

    return tokens


def test_forward_from_all_admins(tokens: dict):
    """Нагрузка /forward от ВСЕХ админов."""
    print("\n" + "-" * 60)
    print("НАГРУЗКА /FORWARD ОТ ВСЕХ АДМИНОВ")
    print("-" * 60)

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

    total_requests = 0
    for admin_name, _token in tokens.items():
        print(f"\n{admin_name} гонит {FORWARD_CALLS} /forward запросов")
        for i in range(1, FORWARD_CALLS + 1):
            print(f"  --- /forward #{i}/{FORWARD_CALLS} от {admin_name} ---")
            r = requests.post(f"{BASE_URL}/forward", json=forward_data)
            print("  Status:", r.status_code)
            try:
                print("  Response:", r.json())
            except Exception:
                print("  Raw:", r.text)
            total_requests += 1
            if i < FORWARD_CALLS:
                time.sleep(0.1)

    print(f"\nВсего сделано {total_requests} /forward запросов от {len(tokens)} админов")
    print("Ожидаемое количество записей в /history и /stats:", total_requests)


def test_final_admin_routes(tokens: dict):
    """
    Финальный тест /history + /stats:
    - один раз печатаем полный history и stats (до очистки),
    - затем ОДИН раз чистим history и проверяем, что он пустой.
    """
    print("\n" + "-" * 60)
    print("ФИНАЛЬНАЯ ПРОВЕРКА /STATS + /HISTORY")
    print("-" * 60)

    # Возьмём первого админа для финальной проверки
    any_admin, any_token = next(iter(tokens.items()))
    headers = {"Authorization": f"Bearer {any_token}"}

    # 1. Показать /history
    print(f"\n===> GET /history (до очистки), админ {any_admin}")
    r = requests.get(f"{BASE_URL}/history", headers=headers)
    print("Status /history:", r.status_code)
    if r.status_code == 200:
        history = r.json()
        print(f"Всего записей в history: {len(history)}")
        print("Первые записи (id, status, client_ids, uplift):")
        for row in history[:5]:
            client_ids = []
            try:
                for c in row["input"]["client"]:
                    client_ids.append(c["client_id"])
            except Exception:
                client_ids = ["?"]
            try:
                uplifts = row["output"]["uplift"]
            except Exception:
                uplifts = "?"
            print(
                f"  id={row['id']}, status={row['status']}, "
                f"clients={client_ids}, uplift={uplifts}"
            )

    # 2. Показать /stats
    print(f"\n===> GET /stats (до очистки), админ {any_admin}")
    r = requests.get(f"{BASE_URL}/stats", headers=headers)
    print("Status /stats:", r.status_code)
    if r.status_code == 200:
        stats = r.json()
        print("Stats JSON:")
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    # 3. Очистка history
    print(f"\n===> DELETE /history, админ {any_admin}")
    r = requests.delete(f"{BASE_URL}/history", headers=headers)
    print("Status DELETE /history:", r.status_code)
    try:
        print("Response:", r.json())
    except Exception:
        print("Raw:", r.text)

    # 4. Проверка, что history пустой
    print(f"\n===> GET /history (после очистки), админ {any_admin}")
    r = requests.get(f"{BASE_URL}/history", headers=headers)
    print("Status /history:", r.status_code)
    if r.status_code == 200:
        history = r.json()
        print("Записей в history после очистки:", len(history))


if __name__ == "__main__":
    print("ПОЛНЫЙ ТЕСТ UPLIFT API")
    print("Проверит: /docs, /login, /admins, /forward, /history, /stats, DELETE /history")
    print("Важно: запусти сервер: uvicorn main:app --host 127.0.0.1 --port 8000 --reload")
    print('JWT_SECRET должен быть задан, например: export JWT_SECRET="supersecret123"\n')

    # 1. Создать первого админа в SQLite
    ensure_first_admin()

    # 2. Тест открытых роутов
    test_open_routes()

    # 3. Логин от myadmin и создание других админов
    token1 = login_and_get_token()
    create_test_admins(token1)

    # 4. Логин от всех админов
    tokens = login_multiple_admins()

    # 5. Нагрузка /forward от всех админов
    test_forward_from_all_admins(tokens)

    # 6. Показать history, stats и проверить DELETE
    test_final_admin_routes(tokens)

    print("\nТЕСТ ЗАВЕРШЕН! Все роуты протестированы.")
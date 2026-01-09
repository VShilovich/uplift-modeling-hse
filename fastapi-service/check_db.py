import sqlite3

conn = sqlite3.connect('data/uplift-modeling.db')
cursor = conn.cursor()

# получить все таблицы
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Таблицы в БД:")
for table in tables:
    print(f"  - {table[0]}")

# проверить таблицу alembic_version (создается Alembic)
cursor.execute("SELECT * FROM alembic_version")
version = cursor.fetchone()
print(f"\nВерсия Alembic: {version[0] if version else 'Не найдена'}")

conn.close()

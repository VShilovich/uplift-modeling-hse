#!/usr/bin/env python3
"""
Скрипт для управления миграциями Alembic
"""
import os
import sys
import subprocess

# установка переменной окружения для Alembic
os.environ["JWT_SECRET"] = "temp_for_alembic"

def run_command(cmd):
    """Выполнить команду и показать результат"""
    print(f"\n>>> Выполняю: {cmd}")
    result = subprocess.run(cmd, shell=True, text=True)
    if result.returncode != 0:
        print(f"Ошибка: {result.stderr}")
    return result.returncode

def main():
    if len(sys.argv) < 2:
        print("Использование: python migrate.py [команда]")
        print("Команды:")
        print("  create <message>  - создать новую миграцию")
        print("  upgrade           - применить все миграции")
        print("  downgrade         - откатить последнюю миграцию")
        print("  history           - показать историю миграций")
        print("  current           - показать текущую версию")
        print("  status            - проверить состояние БД")
        return
    
    command = sys.argv[1]
    
    if command == "create" and len(sys.argv) > 2:
        message = sys.argv[2]
        run_command(f"alembic revision --autogenerate -m \"{message}\"")
    elif command == "upgrade":
        run_command("alembic upgrade head")
    elif command == "downgrade":
        run_command("alembic downgrade -1")
    elif command == "history":
        run_command("alembic history")
    elif command == "current":
        run_command("alembic current")
    elif command == "status":
        # првоерка состояния БД
        import sqlite3
        conn = sqlite3.connect('data/uplift-modeling.db')
        cursor = conn.cursor()
        
        # таблицы
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall()]
        print(f"Таблицы в БД ({len(tables)}): {', '.join(tables)}")
        
        # версия Alembic
        if 'alembic_version' in tables:
            cursor.execute("SELECT version_num FROM alembic_version")
            version = cursor.fetchone()
            print(f"Текущая версия миграции: {version[0] if version else 'Не определена'}")
        
        conn.close()
    else:
        print(f"Неизвестная команда: {command}")

if __name__ == "__main__":
    main()

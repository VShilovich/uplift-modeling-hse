import sys
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# установка переменной окружения перед импортом app
os.environ["JWT_SECRET"] = "temp_for_alembic"

# блокировка импорта app.py
# создание временного модуля app, чтобы не загружать app.py
import types
sys.modules['app'] = types.ModuleType('app')

# безопасно добавляю путь к проекту
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# создание моделей напрямую в env.py без импорта из app
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, Text, MetaData

# создание базового класса
Base = declarative_base()

# определение моделей
class History(Base):
    __tablename__ = "history"
    
    id = Column(Integer, primary_key=True, index=True)
    ts = Column(String)
    processing_time = Column(Float)
    input_size = Column(Integer)
    input_tokens = Column(Integer)
    status_code = Column(Integer)
    input_data = Column(Text)
    output_data = Column(Text)

class Admin(Base):
    __tablename__ = "admins"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)

print("Alembic: модели созданы напрямую в env.py")


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

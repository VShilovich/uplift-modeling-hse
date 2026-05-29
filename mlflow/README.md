# Быстрый старт

## 0. Установка mlflow

```bash
pip install "mlflow==2.7.1"
pip install boto3
```

## 1. Запуск контейнеров

```bash
cd mlflow
docker compose up -d
```

Проверить статус:
```bash
docker compose ps
```

## 2. Доступ к сервисам

| Сервис | URL | Логин |
|--------|-----|-------|
| MLflow UI | http://localhost:5000 | - |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |

## 3. Создание бакета в MinIO

Откройте http://localhost:9001 -> Buckets -> Create Bucket -> назовите `mlflow`

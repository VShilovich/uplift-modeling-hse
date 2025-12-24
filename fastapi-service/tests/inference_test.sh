#!/usr/bin/env bash
set -e

export JWT_SECRET="supersecret123"

echo ">>> Запускаю uvicorn app:app на 8000"
uvicorn app:app --host 0.0.0.0 --port 8000 --reload &
UVICORN_PID=$!

sleep 5

echo ">>> Гоняю test_app.py"
python -m tests.test_app
TEST_RC=$?

echo ">>> Останавливаю uvicorn (PID=${UVICORN_PID})"
kill "${UVICORN_PID}" 2>/dev/null || true

echo ">>> Код возврата теста: ${TEST_RC}"
if [ "${TEST_RC}" -eq 0 ]; then
  echo ">>> ВСЁ ОК"
else
  echo ">>> ЕСТЬ ОШИБКИ"
fi

exit "${TEST_RC}"
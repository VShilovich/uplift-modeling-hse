КАК ЗАПУСТИТЬ JWT И АДМИН-РАУТЫ

1. Задать JWT секрет и запустить сервис
---------------------------------------

В отдельном терминале:

    export JWT_SECRET="supersecret123"   # свой секрет
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload


2. Создать первого админа через sqlite3
---------------------------------------

Один раз, чтобы было с кем логиниться:

    sqlite3 data/uplift-modeling.db
    INSERT INTO admins (username, password_hash) VALUES ('myadmin', 'mypass123');
    SELECT * FROM admins;
    .exit

Теперь в БД есть админ: логин `myadmin`, пароль `mypass123`.
В проде полагаем, что админа создает тот, у кого есть доступ к дб, поэтому такой путь.


3. Получить JWT токен при логине
--------------------------------

Через Swagger UI:

    1) Открыть в браузере http://127.0.0.1:8000/docs
    2) Найти блок POST /login → "Try it out"
    3) В тело запроса поставить:
       {
         "username": "myadmin",
         "password": "mypass123"
       }
    4) Нажать "Execute"
    5) В ответе внизу увидеть поле access_token — это и есть JWT токен.

Через curl (альтернатива):

    curl -X POST http://127.0.0.1:8000/login \
      -H "Content-Type: application/json" \
      -d '{"username":"myadmin","password":"mypass123"}'

Ответ:

    {
      "access_token": "<JWT_СТРОКА>",
      "token_type": "bearer"
    }


4. Ходить под админом в защищённые рауты
----------------------------------------

Через Swagger:

    1) В правом верхнем углу нажать кнопку "Authorize".
    2) В поле "value" ввести:
         Bearer <access_token>
       (именно с префиксом Bearer и пробелом).
    3) Нажать "Authorize" → "Close".

После этого Swagger будет сам подставлять заголовок:
    Authorization: Bearer <access_token>
во все запросы, и станут доступны:

    - GET    /history
    - GET    /stats
    - DELETE /history
    - POST   /admins   (создание новых админов)

Через curl:

    TOKEN="<access_token>"

    # история
    curl http://127.0.0.1:8000/history \
      -H "Authorization: Bearer $TOKEN"

    # статистика
    curl http://127.0.0.1:8000/stats \
      -H "Authorization: Bearer $TOKEN"

    # очистка истории
    curl -X DELETE http://127.0.0.1:8000/history \
      -H "Authorization: Bearer $TOKEN"

    # создание нового админа
    curl -X POST http://127.0.0.1:8000/admins \
      -H "Authorization: Bearer $TOKEN" \
      -H "Content-Type: application/json" \
      -d '{"username":"second_admin","password":"secondpass"}'

import pickle
import sqlite3
import json
import uvicorn
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, Request, Response, Header
from feature_extraction import UpliftFeatureExtractor

app = FastAPI()

try:
    with open("model.pkl", "rb") as f:
        loaded = pickle.load(f)
    model = loaded["model"]
    feature_names = loaded["feature_names"]
except FileNotFoundError:
    model = None
    feature_names = []

fe = UpliftFeatureExtractor(drop_redundant=True)

DB_FILE = "uplift-modeling.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    # сохраняем все как текст, так проще парсить потом
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            input_data TEXT,
            output_data TEXT,
            status_code INTEGER
        )
    """)
    conn.commit()
    conn.close()

init_db()

def log_request_to_db(input_data, output_data, status):
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO history (ts, input_data, output_data, status_code) VALUES (?, ?, ?, ?)",
            (
                datetime.now().isoformat(),
                json.dumps(input_data, ensure_ascii=False), # ensure_ascii чтобы кириллица норм читалась
                json.dumps(output_data, ensure_ascii=False),
                status
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        # если и упало, сервис будет работать
        print(f"Одна ошибка и ты ошибся: {e}")

@app.post("/forward")
async def forward(request: Request):
    try:
        data = await request.json()
        input_data = data
    except Exception:
        log_request_to_db({}, {"error": "JSON parse error"}, 400)
        return Response("bad request", status_code=400)

    try:
        client_df = pd.DataFrame(data["client"])
        purchases_df = pd.DataFrame(data["purchases"])

        client_id = client_df["client_id"]
    except Exception:
        log_request_to_db(input_data, {"error": "Invalid data structure"}, 400)
        return Response("bad request", status_code=400)

    train_df = pd.DataFrame({"client_id": client_id})

    # заглушки
    treatment_df = pd.DataFrame({"treatment_flg": 0}, index=[0])
    target_df = pd.DataFrame({"target": 0}, index=[0])

    try:
        df = fe.calculate_features(
            client_df,
            train_df,
            treatment_df,
            target_df,
            purchases_df,
        )

        X = df[feature_names]
        uplift = model.predict(X)

        uplift_result = uplift.tolist()[0]
        response_body = {"uplift": uplift_result}
        log_request_to_db(input_data, response_body, 200)

        return response_body


    except Exception:
        log_request_to_db(input_data, {"error": "Model processing failed"}, 403)
        return Response(
            "модель не смогла обработать данные",
            status_code=403,
        )

# GET-запрос /history
@app.get("/history")
async def get_history():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM history ORDER BY id DESC")
    rows = cur.fetchall()
    
    history = []
    for row in rows:
        try:
            inp = json.loads(row["input_data"])
        except:
            inp = row["input_data"]
            
        try:
            out = json.loads(row["output_data"])
        except:
            out = row["output_data"]

        history.append({
            "id": row["id"],
            "timestamp": row["ts"],
            "input": inp,
            "output": out,
            "status": row["status_code"]
        })
    
    conn.close()
    return history

# DELETE-запрос /history
@app.delete("/history")
async def clear_history(token: str = Header(None)):
    if token != "token52":
        return Response("Неверный токен, доступ запрещен", status_code=401)
    
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("DELETE FROM history")
    conn.commit()
    conn.close()
    
    return {"status": "ok", "message": "History cleared"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
import os
import pickle
import sqlite3
import json
import uvicorn
import pandas as pd
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Request, Response, Header, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from utils.feature_extraction import UpliftFeatureExtractor
import jwt # PyJWT !!!!!!!!!!!!!!! pip show PyJWT чек
import numpy as np
from typing import Optional
from pydantic import BaseModel

app = FastAPI()

security = HTTPBearer()

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise ValueError("JWT_SECRET environment variable is required!")

ALGORITHM = "HS256"

class AdminUser(BaseModel):
    username: str
    password: str

try:
    with open("data/model.pkl", "rb") as f:
        loaded = pickle.load(f)
    model = loaded["model"]
    feature_names = loaded["feature_names"]
except FileNotFoundError:
    model = None
    feature_names = []

fe = UpliftFeatureExtractor(drop_redundant=True)

DB_FILE = "data/uplift-modeling.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            processing_time REAL,
            input_size INTEGER,
            input_tokens INTEGER,
            status_code INTEGER,
            input_data TEXT,
            output_data TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def log_request_to_db(input_data, output_data, status, processing_time: float, input_size: int, input_tokens: int):
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO history (ts, processing_time, input_size, input_tokens, status_code, input_data, output_data) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            processing_time,
            input_size,
            input_tokens,
            status,
            json.dumps(input_data, ensure_ascii=False),
            json.dumps(output_data, ensure_ascii=False)
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Ошибка логирования: {e}")

def verify_token(token: str) -> bool:
    """Проверяет JWT токен"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        return True
    except jwt.ExpiredSignatureError:
        return False
    except jwt.InvalidTokenError:
        return False

async def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials or not credentials.credentials or not verify_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid or missing JWT token")
    return credentials.credentials

@app.post("/admins", dependencies=[Depends(get_current_admin)])
async def create_admin(user: AdminUser):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO admins (username, password_hash) VALUES (?, ?)",
        (user.username, user.password)
    )
    conn.commit()
    conn.close()
    return {"status": "ok", "message": f"admin {user.username} created"}

@app.post("/login")
async def admin_login(user: AdminUser):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT * FROM admins WHERE username = ?", (user.username,))
    admin = cur.fetchone()
    conn.close()

    if not admin or admin[2] != user.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    exp = datetime.now(timezone.utc) + timedelta(hours=1)
    payload = {
        "username": user.username,
        "exp": int(exp.timestamp()),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)

    return {"access_token": token, "token_type": "bearer"}

@app.post("/forward")
async def forward(request: Request):
    start_time = datetime.now()
    
    try:
        data = await request.json()
        input_data = data
        # сериализуем запрос в строку
        raw = json.dumps(data, ensure_ascii=False)
        # длина сообщения в байтах
        input_size = len(raw.encode("utf-8"))
        # токены как количество слов (по пробелам)
        input_tokens = len(raw.split())
    except Exception:
        processing_time = (datetime.now() - start_time).total_seconds()
        log_request_to_db({}, {"error": "JSON parse error"}, 400, processing_time, 0, 0)
        return Response("bad request", status_code=400)

    try:
        # clients и purchases из запроса
        client_df = pd.DataFrame(data["client"])
        purchases_df = pd.DataFrame(data["purchases"])

        if "client_id" not in client_df.columns or "client_id" not in purchases_df.columns:
            raise ValueError("client_id is required")

        client_ids = client_df["client_id"].astype(int)

        train_df = pd.DataFrame({"client_id": client_ids})
        treatment_df = pd.DataFrame({"treatment_flg": np.zeros(len(client_ids), dtype=int)})
        target_df = pd.DataFrame({"target": np.zeros(len(client_ids), dtype=int)})

    except Exception:
        processing_time = (datetime.now() - start_time).total_seconds()
        log_request_to_db(
            input_data,
            {"error": "Invalid data structure"},
            400,
            processing_time,
            input_size,
            input_tokens,
        )
        return Response("bad request", status_code=400)

    try:
        df = fe.calculate_features(
            clients_df=client_df,
            train_df=train_df,
            treatment_df=treatment_df,
            target_df=target_df,
            purchases_df=purchases_df,
        )

        X = df[feature_names]

        uplift = model.predict(X)
        uplift_list = uplift.tolist()

        client_ids_out = df.index.tolist()

        response_body = {
            "uplift": [
                {"client_id": int(cid), "uplift": float(u)}
                for cid, u in zip(client_ids_out, uplift_list)
            ]
        }

        processing_time = (datetime.now() - start_time).total_seconds()
        log_request_to_db(input_data, response_body, 200, processing_time, input_size, input_tokens)
        return response_body

    except Exception:
        processing_time = (datetime.now() - start_time).total_seconds()
        log_request_to_db(
            input_data,
            {"error": "Model processing failed"},
            403,
            processing_time,
            input_size,
            input_tokens,
        )
        return Response("Модель не смогла обработать данные", status_code=403)

# GET-запрос /history
@app.get("/history", dependencies=[Depends(get_current_admin)])
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
            "processing_time": row["processing_time"],
            "input_size": row["input_size"],
            "input_tokens": row["input_tokens"],
            "input": inp,
            "output": out,
            "status": row["status_code"]
        })
    
    conn.close()
    return history

# DELETE-запрос /history
@app.delete("/history", dependencies=[Depends(get_current_admin)])
async def clear_history():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("DELETE FROM history")
    conn.commit()
    conn.close()
    
    return {"status": "ok", "message": "History cleared"}

@app.get("/stats", dependencies=[Depends(get_current_admin)])
async def get_stats():
    """Статистика запросов: время обработки, квантили, характеристики входных данных"""
    conn = sqlite3.connect(DB_FILE)
    
    # Статистика времени обработки
    cur = conn.cursor()
    cur.execute("""
        SELECT processing_time, input_size, input_tokens 
        FROM history 
        WHERE processing_time IS NOT NULL AND processing_time > 0
    """)
    rows = cur.fetchall()
    conn.close()
    
    if not rows:
        return {"error": "No processing data available"}
    
    times = [row[0] for row in rows]
    input_sizes = [row[1] for row in rows if row[1]]
    input_token_counts = [row[2] for row in rows if row[2]]
    
    stats = {
        "processing_time": {
            "mean": float(np.mean(times)),
            "p50": float(np.percentile(times, 50)),
            "p95": float(np.percentile(times, 95)),
            "p99": float(np.percentile(times, 99)),
            "count": len(times),
            "total": float(np.sum(times)),
        },
        "input_characteristics": {
            "input_size_bytes": {
                "mean": float(np.mean(input_sizes)) if input_sizes else 0.0,
                "total": float(np.sum(input_sizes)) if input_sizes else 0.0,
                "count": len(input_sizes),
            },
            "input_tokens": {
                "mean": float(np.mean(input_token_counts)) if input_token_counts else 0.0,
                "total": float(np.sum(input_token_counts)) if input_token_counts else 0.0,
                "count": len(input_token_counts),
            },
        },
    }
    
    return stats

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
import pickle
import pandas as pd
from fastapi import FastAPI, Request, Response
from feature_extraction import UpliftFeatureExtractor
import uvicorn

app = FastAPI()

with open("model.pkl", "rb") as f:
    loaded = pickle.load(f)

model = loaded["model"]
feature_names = loaded["feature_names"]

fe = UpliftFeatureExtractor(drop_redundant=True)

@app.post("/forward")
async def forward(request: Request):
    try:
        data = await request.json()
    except Exception:
        return Response("bad request", status_code=400)

    try:
        client_df = pd.DataFrame(data["client"])
        purchases_df = pd.DataFrame(data["purchases"])

        client_id = client_df["client_id"]
    except Exception:
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

        return {
            "uplift": uplift.tolist()[0]
        }

    except Exception:
        return Response(
            "модель не смогла обработать данные",
            status_code=403,
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
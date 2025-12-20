import pickle
from sklift.datasets import fetch_x5
from utils.feature_extraction import UpliftFeatureExtractor
from utils.model_extraction import build_t_learner_logreg

TARGET_COL = "target"
TREATMENT_COL = "treatment_flg"

print("Загружается датасет")

dataset = fetch_x5()
data = dataset.data

extractor = UpliftFeatureExtractor(drop_redundant=True)

print("Начинается создание признаков")

df = extractor.calculate_features(
    clients_df=data.clients,
    train_df=data.train,
    treatment_df=dataset.treatment,
    target_df=dataset.target,
    purchases_df=data.purchases
)

features = extractor.feature_names

print(f"Создано признаков: {len(features)}")
print(f"Размер датафрейма: {df.shape}")
print(f"Признаки: {features}")

y = df[TARGET_COL].values
t = df[TREATMENT_COL].values
X_all = df[features].copy()

num_cols = X_all.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X_all.select_dtypes(include=["object"]).columns.tolist()

print("Начинается обучение модели")

t_model = build_t_learner_logreg(num_cols=num_cols, cat_cols=cat_cols)
t_model.fit(X_all, y, treatment=t)

with open("data/model.pkl", "wb") as f:
    pickle.dump(
        {
            "model": t_model,
            "feature_names": features
        },
        f
    )

print("Модель сохранена")
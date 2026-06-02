# X-Learner CatBoost uplift pipeline with MLflow + MinIO

## Обзор проекта

В проекте реализован локальный pipeline для uplift-моделирования на датасете X5 RetailHero с использованием:

- `MLflow` для трекинга экспериментов и реестра моделей;
- `MinIO` как локального S3-совместимого хранилища артефактов;
- `X-Learner (CatBoost)` как финальной uplift-модели;
- `UpliftFeatureExtractorExpanded` для построения клиентских агрегированных признаков;
- `UmapClusterTransformer` для добавления кластерных представлений в пространство признаков.

## Финальная модель

В текущей конфигурации inference-ноутбука production/inference-версия хранит 3 компоненты модели:

- `model_tau_0`
- `model_tau_1`
- `model_propensity`

---

## Структура проекта

- `docker-compose.yml` - подъем локального `mlflow` и `minio`
- `Experiments.ipynb` - обучение и логирование финальной uplift-модели
- `Inference_S3.ipynb` - загрузка модели из MLflow Registry и тестовый inference
- `helpers/feature_extraction_expanded.py` - построение expanded feature space
- `helpers/classic_ml_models.py` - реализация X-Learner и других ML uplift-моделей
- `helpers/clusterization_features_extraction.py` - UMAP + KMeans кластеризация

---

## Инфраструктура

Через `docker-compose` поднимаются два сервиса:

### 1. MLflow
- URL: `http://localhost:5000`

### 2. MinIO
- S3 endpoint: `http://localhost:9000`
- web console: `http://localhost:9001`

### Дефолтные credentials
- `AWS_ACCESS_KEY_ID = minioadmin`
- `AWS_SECRET_ACCESS_KEY = minioadmin`

Артефакты MLflow сохраняются в:

```text
s3://mlflow/artifacts
```

---

## Как запустить инфраструктуру

### Шаг 1. Проверить Docker

Убедитесь, что установлен Docker Desktop и доступна команда:

```bash
docker compose version
```

### Шаг 2. Запустить сервисы

В корне проекта выполните:

```bash
docker compose up -d
```

### Шаг 3. Проверить доступность сервисов

После запуска откройте:

- MLflow UI: `http://localhost:5000`
- MinIO Console: `http://localhost:9001`

Логин и пароль для MinIO:

```text
minioadmin
minioadmin
```

### Шаг 4. Создать бакет `mlflow`

В MinIO Console вручную создайте бакет:

```text
mlflow
```

Это необходимо, потому что MLflow server пишет артефакты в:

```text
s3://mlflow/artifacts
```

---

## Как подключиться к MLflow из ноутбуков

И `Experiments.ipynb`, и `Inference_S3.ipynb` используют одинаковую локальную конфигурацию окружения:

```python
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
```

Поэтому дополнительных настроек не требуется: достаточно, чтобы `docker compose` был поднят и бакет `mlflow` существовал.

---

## Как воспроизвести обучение модели

### Ноутбук: `Experiments.ipynb`

Ноутбук выполняет следующие шаги:

1. Загружает датасет через `fetch_x5()`;
2. Строит итоговый клиентский датасет через `UpliftFeatureExtractorExpanded`;
3. Делит данные на:
   - `x_train_full`
   - `x_test_holdout`
4. На train-части строит кластерное представление через `UmapClusterTransformer`;
5. Обучает `X-Learner (CatBoost)`:
   - outcome models
   - effect models
   - propensity model
6. Считает uplift-метрики на `test_holdout`;
7. Логирует параметры и метрики в `MLflow`;
8. Регистрирует в реестре три компонента модели:
   - `model_tau_0`
   - `model_tau_1`
   - `model_propensity`

### Что логируется в текущей версии ноутбука

В текущей версии `Experiments.ipynb` логируются:

- гиперпараметры outcome / effect / propensity моделей;
- итоговые uplift-метрики на 'val' и `test_holdout`
- результаты анализа ошибок, устойчивости и baseline-сопоставления
- зарегистрированные модели в MLflow Model Registry

---

## Как подготовить данные под версионирование модели

В текущем pipeline данные для модели готовятся строго так же, как в `Experiments.ipynb`.

### Источник данных
Используется:

```python
dataset = fetch_x5()
```

### Feature engineering
Финальное признаковое пространство строится через:

```python
extractor = UpliftFeatureExtractorExpanded(drop_redundant=True)
df = extractor.calculate_features(...)
```

### Итоговые признаки
Из итогового `df` берутся:

- `features = extractor.feature_names`
- `TARGET_COL = "target"`
- `TREATMENT_COL = "treatment_flg"`

### Разделение выборки
Используется стратифицированный split по комбинации:

- `treatment`
- `target`

то есть страта задается как:

```python
pd.Series(t_all).astype(str) + '_' + pd.Series(y_all).astype(str)
```

После этого выполняется:

```python
train_test_split(..., test_size=0.2, random_state=42, stratify=strata_all)
```

### Кластерные признаки
На train-части строится кластеризатор:

```python
clusterizer_full = UmapClusterTransformer(num_cols=current_num_cols, n_clusters=5)
```

Далее:

- `x_train_full_clust = clusterizer_full.fit_transform(x_train_full)`
- `x_test_holdout_clust = clusterizer_full.transform(x_test_holdout)`

Эти данные подаются в финальный `X-Learner`.

---

## Как выполнить inference

### Ноутбук: `Inference_S3.ipynb`

Этот ноутбук:

1. Подключается к `MLflow` и `MinIO`;
2. Загружает из Model Registry:
   - `model_tau_0`
   - `model_tau_1`
   - `model_propensity`
3. Создает объект `S3XLearnerPredictor`;
4. Собирает признаки для входного payload через `UpliftFeatureExtractorExpanded`;
5. Применяет `UmapClusterTransformer`;
6. Считает итоговый uplift:

```python
uplift = g * tau0 + (1 - g) * tau1
```

### Что нужно для запуска inference

Перед запуском `Inference_S3.ipynb` должны быть выполнены условия:

- поднят `docker compose`
- работает `MLflow`
- работает `MinIO`
- в реестре MLflow существуют:
  - `model_tau_0`
  - `model_tau_1`
  - `model_propensity`

### Порядок запуска

1. Сначала один раз обучить и зарегистрировать модель через `Experiments.ipynb`;
2. Затем открыть `Inference_S3.ipynb`;
3. Выполнить ноутбук сверху вниз;
4. Получить uplift-предсказание для mock payload.

---

## Формула итогового uplift-предсказания

В inference используется стандартная формула X-Learner:

```text
uplift(x) = g(x) * tau_0(x) + (1 - g(x)) * tau_1(x)
```

где:

- `tau_0(x)` - оценка эффекта, обученная на control-группе;
- `tau_1(x)` - оценка эффекта, обученная на treatment-группе;
- `g(x)` - propensity score, то есть вероятность назначения treatment.

---

## Анализ финальной production-модели и её параметров

В качестве основной production-модели был выбран **X-Learner на базе CatBoost**, поскольку именно этот подход показал наилучшее сочетание качества ранжирования uplift, бизнес-эффекта в top-сегментах и устойчивости на cross-validation и holdout. В отличие от S-Learner, который учит одну общую модель отклика, и T-Learner, который независимо моделирует отклик в treatment и control, **X-Learner дополнительно строит псевдо-эффекты** и тем самым лучше адаптирован именно к задаче оценки **индивидуального treatment effect**.

### Логика работы X-Learner в нашей реализации

Наша реализация состоит из трёх типов моделей:

1. **Outcome models**  
   Две модели классификации:
   - `mu_0(x)` - вероятность отклика в control,
   - `mu_1(x)` - вероятность отклика в treatment.

   Они обучаются отдельно на контрольной и тестовой группах и дают оценку двух потенциальных исходов для клиента.

2. **Effect models**  
   После обучения outcome-моделей для объектов обеих групп строятся псевдо-эффекты:
   - для control: `D0 = mu_1(x) - y`
   - для treatment: `D1 = y - mu_0(x)`

   Далее обучаются две модели эффекта:
   - `tau_0(x)` на control-объектах,
   - `tau_1(x)` на treatment-объектах.

   Именно эти модели уже напрямую приближают uplift.

3. **Propensity model**  
   Отдельно обучается модель вероятности treatment:
   - `g(x) = P(T=1 | X=x)`

   На этапе предсказания итоговый uplift считается как взвешенная комбинация:
   - `uplift(x) = g(x) * tau_0(x) + (1 - g(x)) * tau_1(x)`

   Такая схема позволяет гибко комбинировать информацию из обеих групп и обычно работает лучше, чем простой T-Learner, если эффект по группам неоднороден.

### Почему для X-Learner выбран именно CatBoost

Для всех трёх блоков X-Learner в качестве базового алгоритма был выбран **CatBoost**, потому что он хорошо подходит для табличных данных X5:

- умеет стабильно работать со смешанными числовыми и категориальными признаками;
- хорошо обрабатывает нелинейности и взаимодействия признаков;
- устойчив к шуму и пропускам;
- даёт сильное качество без тяжёлого ручного препроцессинга категорий;
- особенно полезен в uplift-задаче, где эффект часто задаётся сложной комбинацией клиентских паттернов, а не одним линейным правилом.
- был выбран в рамках использования фреймворка optuna (против логарифмического регрессора и других подходов).

Дополнительно в коде используется `_sanitize_catboost_input`, чтобы все категориальные признаки гарантированно приводились к строковому формату и корректно обрабатывались CatBoost даже после калибровки и обёрток.

### Выбранные параметры outcome-моделей

```python
XL_OUTCOME_CATBOOST_PARAMS = {
    'iterations': 147,
    'learning_rate': 0.03633155899517177,
    'depth': 5,
    'l2_leaf_reg': 5.860350130719548,
    'random_seed': 42,
    'verbose': 100,
    'allow_writing_files': False,
}
```

Outcome-модели отвечают за оценку вероятностей `mu_0(x)` и `mu_1(x)`, поэтому здесь используется **CatBoostClassifier** с умеренно консервативной конфигурацией:

- **`iterations = 147`** - число деревьев выбрано относительно небольшим, чтобы не переобучать outcome-модель на клиентских агрегатах. Для X-Learner слишком агрессивные outcome-модели опасны: шум в `mu_0` и `mu_1` потом напрямую переносится в псевдо-эффекты.
- **`learning_rate = 0.0363`** - малый learning rate делает обучение более плавным и устойчивым. Это особенно важно здесь, потому что outcome-модели - фундамент всей последующей X-Learner-схемы.
- **`depth = 5`** - глубина выбрана умеренной: модель уже умеет ловить нелинейные зависимости и взаимодействия признаков, но ещё не становится слишком сложной для сравнительно компактной клиентской выборки на уровне агрегатов.
- **`l2_leaf_reg = 5.86`** - усиленная L2-регуляризация дополнительно сдерживает переобучение. Для uplift-задачи это полезно, потому что далее мы работаем не с обычным target, а с разностями предсказаний, и шум легко усиливается.
- **`random_seed = 42`** - нужен для воспроизводимости.
- **`allow_writing_files = False`** - отключает служебную запись файлов CatBoost.

Итого outcome-блок настроен **не на максимальную агрессивность**, а на **стабильное вероятностное моделирование**.

### Выбранные параметры effect-моделей

```python
XL_EFFECT_CATBOOST_PARAMS = {
    'iterations': 147,
    'learning_rate': 0.03633155899517177,
    'depth': 5,
    'l2_leaf_reg': 5.860350130719548,
    'loss_function': 'RMSE',
    'random_seed': 42,
    'verbose': 100,
    'allow_writing_files': False,
}
```

Effect-модели учат уже не бинарный target, а **псевдо-эффекты** `D0` и `D1`, поэтому здесь используется **CatBoostRegressor**.

Параметры почти совпадают с outcome-блоком:
- архитектурно мы хотим, чтобы модели эффекта имели сопоставимую сложность;
- те же `iterations`, `learning_rate`, `depth`, `l2_leaf_reg` дают такой же баланс между гибкостью и устойчивостью.

Ключевое отличие:
- **`loss_function = 'RMSE'`**, потому что `D0` и `D1` - это уже непрерывные величины, а не классы.

### Выбранные параметры propensity-модели

```python
XL_PROPENSITY_CATBOOST_PARAMS = {
    'iterations': 100,
    'learning_rate': 0.1,
    'depth': 4,
    'random_seed': 42,
    'verbose': 0,
    'allow_writing_files': False,
}
```

Propensity-модель решает более простую задачу: оценить вероятность treatment `g(x)`. В нашем датасете treatment распределён достаточно ровно, поэтому здесь не нужна такая же сложная модель, как для outcome/effect блоков.

- **`iterations = 100`** - меньше деревьев, потому что propensity здесь скорее вспомогательная часть, а не главный источник uplift-сигнала.
- **`learning_rate = 0.1`** - можно позволить чуть более быстрый шаг обучения, так как задача проще и менее чувствительна к тонким вероятностным различиям.
- **`depth = 4`** - ещё более простая структура дерева. Этого достаточно, чтобы схватить основные паттерны назначения treatment, не усложняя лишний раз модель.
- **`verbose = 0`** - у propensity нет необходимости в подробном логе обучения.

Итого propensity-блок специально сделан **легче и проще**, чем outcome/effect, потому что его роль - корректно задавать веса при смешивании `tau_0` и `tau_1`, а не быть самой сложной частью пайплайна.

### Почему используется калибровка

В пайплайне outcome- и propensity-модели дополнительно оборачиваются в `CalibratedClassifierCV` с `method='isotonic'`. Это важно, потому что:

- X-Learner использует **вероятности**, а не просто классы;
- псевдо-эффекты строятся как разности вероятностей и факта;
- плохая калибровка вероятностей ведёт к шумным `D0` и `D1`.

Поэтому перед оборачиванием используются параметры без `use_best_model=True` и без `eval_metric`, чтобы модель корректно встраивалась в calibration-wrapper.

### Итог

Итоговая конфигурация **X-Learner (CatBoost)** выбрана как компромисс между:

- достаточной нелинейностью,
- контролем переобучения,
- устойчивостью вероятностных оценок,
- хорошей работой с категориальными признаками,
- и итоговым качеством uplift-ранжирования.

Ключевая идея выбора параметров состояла не в том, чтобы сделать максимально сложный CatBoost, а в том, чтобы построить **стабильный многошаговый uplift-пайплайн**, где каждая из outcome, effect и propensity элементов решает свою задачу с адекватным уровнем сложности.

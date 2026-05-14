import re
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from models_specs_and_vizuals import *

"""
Интерпретация DL uplift-моделей через activation-based importance.

- выбирает лучшую DL-модель по метрикам CV или holdout;
- при необходимости загружает сохраненную модель или дообучает ее;
- считает вклад признаков в uplift-предсказание через gradient × activation;
- отдельно для:
    1) T-Learner NN,
    2) S-Learner NN,
    3) Classic TARNet,
    4) Attention TARNet;
- агрегирует importance в удобную таблицу и строит barplot.

Идея:
мы интерпретируем не propensity to buy сама по себе, а именно uplift:
    uplift(x) = p_treatment(x) - p_control(x)

Поэтому вклад признаков считается относительно этой разности,
а не относительно одной вероятности отклика.
"""

def choose_best_dl_model_name(
    cv_results_df,
    test_results_df,
    dl_model_names,
    score_col='Qini',
    source='cv'
):
    """
    Выбирает лучшую DL-модель по заданной метрике.

    Логика:
    - если source='cv', берется среднее значение score_col по CV-фолдам;
    - если source='holdout', берется score_col на holdout;
    - выбор делается только среди моделей из dl_model_names.
    """
    if source == 'holdout':
        df = test_results_df[test_results_df['Model'].isin(dl_model_names)].copy()
        best_name = df.sort_values(score_col, ascending=False).iloc[0]['Model']
    else:
        df = cv_results_df[cv_results_df['Model'].isin(dl_model_names)].copy()
        best_name = (
            df.groupby('Model')[score_col]
            .mean()
            .sort_values(ascending=False)
            .index[0]
        )
    return best_name


def fit_or_load_best_dl_model_for_interpretation(
    model_name,
    holdout_specs,
    x_train_full,
    y_train_full,
    t_train_full,
    dl_model_repo,
    use_saved_dl_models=False,
    save_fitted_dl_models=False,
    device='cpu'
):
    """
    Загружает лучшую DL-модель из репозитория или обучает ее заново
    для последующей интерпретации.

    Что делает:
    - берет spec модели из holdout_specs;
    - строит путь до сохраненной модели;
    - если включен use_saved_dl_models и файл существует, грузит модель с диска;
    - иначе обучает модель на полном train и, при необходимости, сохраняет ее.
    """
    spec = holdout_specs[model_name]
    model = spec['model']
    x_train = spec['x_train']

    model_key = make_model_key(model_name)
    model_path = dl_model_repo / f'{model_key}.pt'

    print('Interpretation model:', model_name)

    if use_saved_dl_models and model_path.exists():
        model = torch.load(model_path, map_location=device, weights_only=False)
        model = move_loaded_dl_model_to_device(model, device)
    else:
        model.fit(x_train, y_train_full, t_train_full)
        if save_fitted_dl_models:
            torch.save(model, model_path)

    return model, spec


def _sample_dataframe(X, max_rows=4096, random_state=42):
    """
    Берет подвыборку строк для интерпретации.

    - activation-based importance на полной выборке может быть дорогой по памяти и времени;

    Если число строк меньше max_rows, возвращается копия всей таблицы.
    """
    if isinstance(X, pd.DataFrame):
        if len(X) > max_rows:
            return X.sample(max_rows, random_state=random_state).copy()
        return X.copy()
    X = pd.DataFrame(X)
    if len(X) > max_rows:
        return X.sample(max_rows, random_state=random_state).copy()
    return X.copy()


def _safe_zero_grad(module):
    """
    Безопасно обнуляет градиенты torch-модуля.
    - где возможно, использует zero_grad(set_to_none=True);
    - если версия PyTorch или объект это не поддерживает, вызывает обычный zero_grad().
    """
    if hasattr(module, 'zero_grad'):
        try:
            module.zero_grad(set_to_none=True)
        except TypeError:
            module.zero_grad()


def _processed_to_raw_feature_map(feature_names, num_cols, cat_cols):
    """
    Преобразует имена признаков после sklearn-preprocessing обратно к исходным именам.

    Особенно важно для T-Learner NN:
    - после ColumnTransformer и OneHotEncoder признаки имеют служебные префиксы;
    - функция агрегирует one-hot признаки обратно в исходный raw feature name,
    чтобы итоговая importance-таблица была читаемой.

    Пример:
    - num__age -> age
    - cat__gender_male -> gender
    """
    cat_cols_sorted = sorted(list(cat_cols), key=len, reverse=True)
    raw_names = []

    for name in feature_names:
        if name.startswith('num__'):
            raw_names.append(name.replace('num__', '', 1))
            continue

        if name.startswith('cat__'):
            stripped = name.replace('cat__', '', 1)
            matched = None
            for col in cat_cols_sorted:
                if stripped == col or stripped.startswith(col + '_'):
                    matched = col
                    break
            raw_names.append(matched if matched is not None else stripped)
            continue

        raw_names.append(name)

    return raw_names


def _fit_tlearner_nn_importance(model, X, batch_size=1024):
    """
    Считает activation-based importance для T-Learner NN.

    Логика:
    - вход прогоняется через уже обученный sklearn-preprocessor;
    - затем считаются предсказания двух нейросетей:
        treatment model и control model;
    - uplift определяется как p_treatment - p_control;
    - importance считается как среднее |grad(uplift) * input| по батчам.

    для T-Learner NN importance сначала считается в processed feature space
    (one-hot / scaled features), а затем агрегируется обратно в raw feature space.
    """
    X_df = X.copy()
    x_processed = model.preprocessor_.transform(X_df)
    if hasattr(x_processed, 'toarray'):
        x_processed = x_processed.toarray()
    x_processed = np.asarray(x_processed, dtype=np.float32)

    feature_names = model.preprocessor_.get_feature_names_out()
    raw_feature_names = _processed_to_raw_feature_map(
        feature_names,
        getattr(model, 'num_cols', []),
        getattr(model, 'cat_cols', [])
    )

    contrib_sum = np.zeros(x_processed.shape[1], dtype=np.float64)
    n_total = 0

    treated_nn = model.treated_model_.model_
    control_nn = model.control_model_.model_
    treated_nn.eval()
    control_nn.eval()

    device = model.device

    for start in range(0, len(x_processed), batch_size):
        batch = x_processed[start:start + batch_size]
        x_t = torch.tensor(batch, dtype=torch.float32, device=device, requires_grad=True)

        _safe_zero_grad(treated_nn)
        _safe_zero_grad(control_nn)

        p_t = treated_nn(x_t)
        p_c = control_nn(x_t)
        uplift = (p_t - p_c).sum()
        uplift.backward()

        contrib = torch.abs(x_t.grad * x_t).mean(dim=0).detach().cpu().numpy()
        contrib_sum += contrib * len(batch)
        n_total += len(batch)

    contrib_mean = contrib_sum / max(n_total, 1)

    df_proc = pd.DataFrame({
        'feature': feature_names,
        'raw_feature': raw_feature_names,
        'activation_importance': contrib_mean
    }).sort_values('activation_importance', ascending=False)

    df_raw = (
        df_proc.groupby('raw_feature', as_index=False)['activation_importance']
        .sum()
        .rename(columns={'raw_feature': 'feature'})
        .sort_values('activation_importance', ascending=False)
        .reset_index(drop=True)
    )

    return df_raw, df_proc


def _slearner_forward_with_tracking(net, x_num, x_cat, treatment):
    """
    Прямой проход S-Learner NN с сохранением градиентов по числовым признакам
    и embedding-активациям категориальных признаков.

    Используется для вычисления feature importance по uplift:
    отдельно прогоняет treatment=1 и treatment=0, а затем сравнивает вклад
    признаков в разность предсказаний.
    """
    x_num_req = x_num.clone().detach().requires_grad_(True)

    cat_embs = []
    if x_cat.size(1) > 0:
        for i, emb_layer in enumerate(net.embeddings):
            emb = emb_layer(x_cat[:, i])
            emb.retain_grad()
            cat_embs.append(emb)
        x_cat_emb = torch.cat(cat_embs, dim=1)
    else:
        x_cat_emb = torch.zeros((x_num.size(0), 0), dtype=x_num.dtype, device=x_num.device)

    t_col = treatment.unsqueeze(1) if treatment.ndim == 1 else treatment
    x_full = torch.cat([x_num_req, x_cat_emb, t_col], dim=1)
    out = torch.sigmoid(net.network(x_full)).squeeze(-1)
    return out, x_num_req, cat_embs


def _classic_tarnet_forward_with_tracking(net, x_num, x_cat):
    """
    Прямой проход Classic TARNet с сохранением градиентов по:
    - числовым признакам,
    - embedding-активациям категориальных признаков.

    Возвращает обе вероятности:
    - p_treatment,
    - p_control,
    а также промежуточные объекты, нужные для attribution.
    """
    x_num_req = x_num.clone().detach().requires_grad_(True)

    cat_embs = []
    if x_cat.size(1) > 0:
        for i, emb_layer in enumerate(net.embeddings):
            emb = emb_layer(x_cat[:, i])
            emb.retain_grad()
            cat_embs.append(emb)
        x_cat_emb = torch.cat(cat_embs, dim=1)
        x_full = torch.cat([x_num_req, x_cat_emb], dim=1)
    else:
        x_full = x_num_req

    rep = net.shared_representation(x_full)
    p_t = torch.sigmoid(net.treatment_head(rep)).squeeze(-1)
    p_c = torch.sigmoid(net.control_head(rep)).squeeze(-1)

    return p_t, p_c, x_num_req, cat_embs


def _attention_tarnet_forward_with_tracking(net, x_num, x_cat):
    """
    Прямой проход Attention TARNet с сохранением градиентов по токенам.

    Что отслеживается:
    - числовые токены после NumericTokenizer;
    - категориальные токены после CategoricalTokenizer, если они есть.

    Позволяет считать importance на уровне token-активаций.
    """
    x_num_req = x_num.clone().detach().requires_grad_(True)

    num_tokens = net.num_tokenizer(x_num_req)
    num_tokens.retain_grad()

    cat_tokens = None
    if x_cat.size(1) > 0:
        cat_tokens = net.cat_tokenizer(x_cat)
        cat_tokens.retain_grad()
        tokens = torch.cat([num_tokens, cat_tokens], dim=1)
    else:
        tokens = num_tokens

    cls = net.cls_token.expand(tokens.size(0), -1, -1)
    tokens = torch.cat([cls, tokens], dim=1)
    shared = net.transformer(tokens)
    cls_repr = shared[:, 0, :]

    p_t = torch.sigmoid(net.treatment_head(cls_repr)).squeeze(-1)
    p_c = torch.sigmoid(net.control_head(cls_repr)).squeeze(-1)

    return p_t, p_c, num_tokens, cat_tokens


def _fit_mixed_dl_importance(model, X, batch_size=512):
    """
    Считает activation-based importance для mixed-input DL uplift-моделей:
    - S-Learner NN,
    - Classic TARNet,
    - Attention TARNet.

    - для S-Learner NN importance считается через разность treatment/control forward pass;
    - для Classic TARNet — через |grad * activation| по числовым признакам и embedding-ам;
    - для Attention TARNet — через |grad * activation| по token-представлениям.

    На выходе importance агрегируется в две группы:
    - числовые признаки;
    - категориальные признаки.
    """
    X_df = X.copy()
    x_num, x_cat = model._transform_mixed(X_df)

    x_num = np.asarray(x_num, dtype=np.float32)
    x_cat = np.asarray(x_cat, dtype=np.int64)

    num_cols = list(getattr(model, 'num_cols', []))
    cat_cols = list(getattr(model, 'cat_cols', []))

    num_contrib_sum = np.zeros(len(num_cols), dtype=np.float64)
    cat_contrib_sum = np.zeros(len(cat_cols), dtype=np.float64)
    n_total = 0

    net = model.model_
    device = model.device
    net.eval()

    class_name = model.__class__.__name__

    for start in range(0, len(x_num), batch_size):
        batch_num = x_num[start:start + batch_size]
        batch_cat = x_cat[start:start + batch_size]

        x_num_t = torch.tensor(batch_num, dtype=torch.float32, device=device)
        x_cat_t = torch.tensor(batch_cat, dtype=torch.long, device=device)

        _safe_zero_grad(net)

        if class_name == 'SLearnerNNUplift':
            ones = torch.ones((len(batch_num),), dtype=torch.float32, device=device)
            zeros = torch.zeros((len(batch_num),), dtype=torch.float32, device=device)

            p_t, x_num_t1, cat_embs_t = _slearner_forward_with_tracking(net, x_num_t, x_cat_t, ones)
            p_c, x_num_t0, cat_embs_c = _slearner_forward_with_tracking(net, x_num_t, x_cat_t, zeros)

            uplift = (p_t - p_c).sum()
            uplift.backward()

            num_contrib = torch.abs((x_num_t1.grad - x_num_t0.grad) * x_num_t1).mean(dim=0).detach().cpu().numpy()
            num_contrib_sum += num_contrib * len(batch_num)

            for j in range(len(cat_cols)):
                diff = (cat_embs_t[j].grad - cat_embs_c[j].grad) * cat_embs_t[j]
                contrib_j = torch.abs(diff).sum(dim=1).mean().item()
                cat_contrib_sum[j] += contrib_j * len(batch_num)

        elif class_name == 'ClassicTARNetUplift':
            p_t, p_c, x_num_req, cat_embs = _classic_tarnet_forward_with_tracking(net, x_num_t, x_cat_t)

            uplift = (p_t - p_c).sum()
            uplift.backward()

            num_contrib = torch.abs(x_num_req.grad * x_num_req).mean(dim=0).detach().cpu().numpy()
            num_contrib_sum += num_contrib * len(batch_num)

            for j in range(len(cat_cols)):
                contrib_j = torch.abs(cat_embs[j].grad * cat_embs[j]).sum(dim=1).mean().item()
                cat_contrib_sum[j] += contrib_j * len(batch_num)

        elif class_name == 'AttentionTARNetUplift':
            p_t, p_c, num_tokens, cat_tokens = _attention_tarnet_forward_with_tracking(net, x_num_t, x_cat_t)

            uplift = (p_t - p_c).sum()
            uplift.backward()

            num_contrib = torch.abs(num_tokens.grad * num_tokens).sum(dim=2).mean(dim=0).detach().cpu().numpy()
            num_contrib_sum += num_contrib * len(batch_num)

            if cat_tokens is not None and len(cat_cols) > 0:
                cat_contrib = torch.abs(cat_tokens.grad * cat_tokens).sum(dim=2).mean(dim=0).detach().cpu().numpy()
                cat_contrib_sum += cat_contrib * len(batch_num)

        else:
            raise ValueError(f'Unsupported mixed DL model for importance: {class_name}')

        n_total += len(batch_num)

    num_contrib_mean = num_contrib_sum / max(n_total, 1)
    cat_contrib_mean = cat_contrib_sum / max(n_total, 1)

    df_num = pd.DataFrame({
        'feature': num_cols,
        'activation_importance': num_contrib_mean
    })

    df_cat = pd.DataFrame({
        'feature': cat_cols,
        'activation_importance': cat_contrib_mean
    })

    df = (
        pd.concat([df_num, df_cat], ignore_index=True)
        .sort_values('activation_importance', ascending=False)
        .reset_index(drop=True)
    )

    return df, None


def compute_best_dl_activation_importance(model, X, batch_size=512, max_rows=4096, random_state=42):
    """
    Главная функция расчета activation-based importance для лучшей DL-модели.

    Что делает:
    - при необходимости берет подвыборку объектов;
    - определяет тип модели по имени класса;
    - вызывает соответствующую специализированную функцию importance:
        - T-Learner NN -> _fit_tlearner_nn_importance
        - S-Learner NN / Classic TARNet / Attention TARNet -> _fit_mixed_dl_importance
    """
    X_use = _sample_dataframe(X, max_rows=max_rows, random_state=random_state)
    class_name = model.__class__.__name__

    if class_name == 'TLearnerNNUplift':
        return _fit_tlearner_nn_importance(model, X_use, batch_size=batch_size)

    if class_name in ['SLearnerNNUplift', 'ClassicTARNetUplift', 'AttentionTARNetUplift']:
        return _fit_mixed_dl_importance(model, X_use, batch_size=batch_size)

    raise ValueError(f'Unsupported DL model class: {class_name}')


def prepare_importance_table(df, top_n=25):
    """
    Подготавливает importance-таблицу к отображению.

    Что делает:
    - приводит activation_importance к числовому виду;
    - заменяет NaN на 0;
    - пересчитывает importance в проценты от общей суммы;
    - оставляет top_n самых важных признаков.
    """
    out = df.copy()
    out['activation_importance'] = pd.to_numeric(out['activation_importance'], errors='coerce').fillna(0.0)
    total = out['activation_importance'].sum()
    out['activation_importance_pct'] = np.where(
        total > 0,
        100.0 * out['activation_importance'] / total,
        0.0
    )
    return out.head(top_n).copy()


def plot_activation_importance(df, title, top_n=25, color='#4c72b0'):
    """
    Строит barplot по activation-based importance признаков.

    Пайплайн:
    - сначала вызывает prepare_importance_table(...);
    - затем строит горизонтальный barplot top_n признаков;
    - возвращает ту же уже подготовленную таблицу, чтобы ее можно было display/save.
    """
    plot_df = prepare_importance_table(df, top_n=top_n)

    plt.figure(figsize=(12, max(6, 0.38 * len(plot_df))))
    sns.barplot(
        data=plot_df,
        x='activation_importance_pct',
        y='feature',
        color=color
    )
    plt.xlabel('доля в activation-based uplift importance, %')
    plt.ylabel('feature')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    return plot_df
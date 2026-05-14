import numpy as np
import pandas as pd
import torch
from clusterization_features_extraction import UmapClusterTransformer
from metrics import calculate_metrics, get_uplift_percentile_table
from classic_ml_models import build_t_learner_logreg, build_x_learner_catboost
from uplift_dl_models import (
    TLearnerNNUplift,
    SLearnerNNUplift,
    ClassicTARNetUplift,
    AttentionTARNetUplift,
    get_default_dl_model_configs,
)
from collections import OrderedDict
from sklearn.calibration import CalibratedClassifierCV
from sklift.viz import plot_qini_curve, plot_uplift_curve
from sklift.metrics import weighted_average_uplift
import matplotlib.pyplot as plt
import math

"""
Вспомогательные функции для uplift-экспериментов с ML- и DL-моделями.

- глобальные константы эксперимента;
- единый порядок моделей и стили их визуализации;
- функции подготовки stratification и списков числовых / категориальных признаков;
- функции построения cluster-based представлений признаков;
- model specs для OOF и holdout;
- единая функция fit + predict для модели на конкретном фолде;
- функции построения uplift / qini curves и percentile uplift-графиков;
- функции подготовки красивой percentile-таблицы;
- служебные функции для загрузки сохраненных DL-моделей и генерации model key.

Идея:
helper отделяет orchestration-логику эксперимента от ноутбука,
чтобы в ноутбуке остался только компактный пайплайн запуска.
"""

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

TARGET_COL = 'target'
TREATMENT_COL = 'treatment_flg'

HOLDOUT_SIZE = 0.2
N_SPLITS = 5
TOP_K_PERCENT = 0.30
N_CLUSTERS = 5

MODEL_ORDER = [
    'T-Learner (LogReg)',
    'X-Learner (CatBoost)',
    'T-Learner NN',
    'S-Learner NN',
    'TARNet (Classic MLP)',
    'TARNet (Attention)',
]

MODEL_COLORS = {
    'T-Learner (LogReg)': '#1f77b4',
    'X-Learner (CatBoost)': '#ff7f0e',
    'T-Learner NN': '#2ca02c',
    'S-Learner NN': '#d62728',
    'TARNet (Classic MLP)': '#9467bd',
    'TARNet (Attention)': '#8c564b',
}

MODEL_STYLES = {
    'T-Learner (LogReg)': {'ls': '-', 'alpha': 0.90},
    'X-Learner (CatBoost)': {'ls': '-', 'alpha': 1.00},
    'T-Learner NN': {'ls': '--', 'alpha': 0.95},
    'S-Learner NN': {'ls': ':', 'alpha': 0.95},
    'TARNet (Classic MLP)': {'ls': '-.', 'alpha': 0.95},
    'TARNet (Attention)': {'ls': '-', 'alpha': 0.95},
}

DL_CONFIGS = get_default_dl_model_configs()

def make_strata(y, t):
    """
    Формирует страты для stratified split по комбинации treatment и target.

    - в uplift-задаче важно сохранять баланс не только по target,
    но и по treatment/control;
    - комбинация вида treatment_target позволяет делать более устойчивые
    train/test и CV-разбиения.
    """
    return pd.Series(t).astype(str) + '_' + pd.Series(y).astype(str)


def get_num_cat_cols(df):
    """
    Разделяет признаки DataFrame на числовые и категориальные.

    Логика:
    - категориальными считаются колонки типов object и category;
    - все остальные колонки относятся к числовым.
    """
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]
    return num_cols, cat_cols


def fit_cluster_views(x_train, *others, n_clusters=N_CLUSTERS, random_state=SEED):
    """
    Строит кластерное представление признаков на train-части и применяет его
    к train и всем переданным дополнительным выборкам.

    Что делает:
    - берет только числовые признаки;
    - обучает UMAP + clustering transformer на train;
    - возвращает преобразованный train и преобразованные остальные фреймы
    в том же feature space.

    Зачем:
    - часть моделей в проекте использует expanded features + cluster features;
    - кластеризация должна обучаться только на train-части,
    чтобы не было leakage.
    """
    num_cols = x_train.select_dtypes(include=['number']).columns.tolist()
    clusterizer = UmapClusterTransformer(
        num_cols=num_cols,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    x_train_cl = clusterizer.fit_transform(x_train)
    transformed = [x_train_cl]
    for frame in others:
        transformed.append(clusterizer.transform(frame))
    return clusterizer, transformed


def build_fold_model_specs(x_tr_plain, x_val_plain, x_tr_cl, x_val_cl, device=DEVICE):
    """
    Собирает спецификации моделей для одного OOF-фолда.

    Что содержит каждый spec:
    - готовый объект модели;
    - train-матрицу признаков для этой модели;
    - validation-матрицу признаков для этой модели;
    - флаг is_dl, показывающий, нужна ли модели логика eval_set.

    Почему нужен отдельный слой specs:
    - разные модели работают на разных представлениях признаков;
    - ML- и DL-модели имеют разный интерфейс обучения;
    - вся логика выбора feature space прячется в одном месте,
    а не размазывается по OOF-циклу.

    Логика feature space:
    - T-Learner (LogReg) и X-Learner (CatBoost) используют cluster features;
    - T-Learner NN и S-Learner NN используют plain expanded features;
    - обе TARNet-модели используют cluster features.
    """
    num_plain, cat_plain = get_num_cat_cols(x_tr_plain)
    num_cl, cat_cl = get_num_cat_cols(x_tr_cl)

    specs = OrderedDict()

    t_learner_ml = build_t_learner_logreg(num_cl, cat_cl)
    t_learner_ml.estimator_trmnt = CalibratedClassifierCV(t_learner_ml.estimator_trmnt, method='isotonic', cv=3)
    t_learner_ml.estimator_ctrl = CalibratedClassifierCV(t_learner_ml.estimator_ctrl, method='isotonic', cv=3)
    specs['T-Learner (LogReg)'] = {
        'model': t_learner_ml,
        'x_train': x_tr_cl,
        'x_eval': x_val_cl,
        'is_dl': False,
    }

    x_learner_ml = build_x_learner_catboost(cat_features=cat_cl, use_calibration=True)
    x_learner_ml.outcome_learner = CalibratedClassifierCV(x_learner_ml.outcome_learner, method='isotonic', cv=3)
    x_learner_ml.propensity_learner = CalibratedClassifierCV(x_learner_ml.propensity_learner, method='isotonic', cv=3)
    specs['X-Learner (CatBoost)'] = {
        'model': x_learner_ml,
        'x_train': x_tr_cl,
        'x_eval': x_val_cl,
        'is_dl': False,
    }

    specs['T-Learner NN'] = {
        'model': TLearnerNNUplift(
            num_cols=num_plain,
            cat_cols=cat_plain,
            device=device,
            random_state=42,
            verbose=False,
            **DL_CONFIGS['t_learner_nn'],
        ),
        'x_train': x_tr_plain,
        'x_eval': x_val_plain,
        'is_dl': True,
    }

    specs['S-Learner NN'] = {
        'model': SLearnerNNUplift(
            num_cols=num_plain,
            cat_cols=cat_plain,
            device=device,
            random_state=SEED,
            verbose=False,
            **DL_CONFIGS['s_learner_nn'],
        ),
        'x_train': x_tr_plain,
        'x_eval': x_val_plain,
        'is_dl': True,
    }

    specs['TARNet (Classic MLP)'] = {
        'model': ClassicTARNetUplift(
            num_cols=num_cl,
            cat_cols=cat_cl,
            device=device,
            random_state=SEED,
            verbose=False,
            **DL_CONFIGS['tarnet_classic'],
        ),
        'x_train': x_tr_cl,
        'x_eval': x_val_cl,
        'is_dl': True,
    }

    specs['TARNet (Attention)'] = {
        'model': AttentionTARNetUplift(
            num_cols=num_cl,
            cat_cols=cat_cl,
            device=device,
            random_state=SEED,
            verbose=False,
            **DL_CONFIGS['tarnet_attention'],
        ),
        'x_train': x_tr_cl,
        'x_eval': x_val_cl,
        'is_dl': True,
    }

    return specs


def fit_model_and_predict(spec, y_train, t_train, y_eval, t_eval):
    """
    Обучает одну модель на train-части и возвращает предсказанный uplift
    на evaluation-части.

    Логика:
    - для DL-моделей вызывается fit(..., eval_set=(x_eval, y_eval, t_eval)),
    чтобы можно было использовать early stopping;
    - для ML-моделей используется обычный fit(x_train, y_train, t_train);
    - после обучения всегда вызывается model.predict(x_eval),
    который должен вернуть uplift-оценку.
    """
    model = spec['model']
    x_train = spec['x_train']
    x_eval = spec['x_eval']

    if spec['is_dl']:
        model.fit(x_train, y_train, t_train, eval_set=(x_eval, y_eval, t_eval))
    else:
        model.fit(x_train, y_train, t_train)

    preds_eval = np.asarray(model.predict(x_eval)).reshape(-1)
    return model, preds_eval


def build_holdout_model_specs(x_train_plain, x_test_plain, x_train_cl, x_test_cl, device=DEVICE):
    """
    Собирает спецификации моделей для финального обучения на train_full
    и последующего предсказания на holdout.

    По смыслу повторяет build_fold_model_specs, но вместо x_eval использует x_test.
    """
    num_plain, cat_plain = get_num_cat_cols(x_train_plain)
    num_cl, cat_cl = get_num_cat_cols(x_train_cl)

    specs = OrderedDict()

    t_learner_ml = build_t_learner_logreg(num_cl, cat_cl)
    t_learner_ml.estimator_trmnt = CalibratedClassifierCV(t_learner_ml.estimator_trmnt, method='isotonic', cv=3)
    t_learner_ml.estimator_ctrl = CalibratedClassifierCV(t_learner_ml.estimator_ctrl, method='isotonic', cv=3)
    specs['T-Learner (LogReg)'] = {
        'model': t_learner_ml,
        'x_train': x_train_cl,
        'x_test': x_test_cl,
        'is_dl': False,
    }

    x_learner_ml = build_x_learner_catboost(cat_features=cat_cl, use_calibration=True)
    x_learner_ml.outcome_learner = CalibratedClassifierCV(x_learner_ml.outcome_learner, method='isotonic', cv=3)
    x_learner_ml.propensity_learner = CalibratedClassifierCV(x_learner_ml.propensity_learner, method='isotonic', cv=3)
    specs['X-Learner (CatBoost)'] = {
        'model': x_learner_ml,
        'x_train': x_train_cl,
        'x_test': x_test_cl,
        'is_dl': False,
    }

    specs['T-Learner NN'] = {
        'model': TLearnerNNUplift(
            num_cols=num_plain,
            cat_cols=cat_plain,
            device=device,
            random_state=SEED,
            verbose=False,
            **DL_CONFIGS['t_learner_nn'],
        ),
        'x_train': x_train_plain,
        'x_test': x_test_plain,
        'is_dl': True,
    }

    specs['S-Learner NN'] = {
        'model': SLearnerNNUplift(
            num_cols=num_plain,
            cat_cols=cat_plain,
            device=device,
            random_state=SEED,
            verbose=False,
            **DL_CONFIGS['s_learner_nn'],
        ),
        'x_train': x_train_plain,
        'x_test': x_test_plain,
        'is_dl': True,
    }

    specs['TARNet (Classic MLP)'] = {
        'model': ClassicTARNetUplift(
            num_cols=num_cl,
            cat_cols=cat_cl,
            device=device,
            random_state=SEED,
            verbose=False,
            **DL_CONFIGS['tarnet_classic'],
        ),
        'x_train': x_train_cl,
        'x_test': x_test_cl,
        'is_dl': True,
    }

    specs['TARNet (Attention)'] = {
        'model': AttentionTARNetUplift(
            num_cols=num_cl,
            cat_cols=cat_cl,
            device=device,
            random_state=SEED,
            verbose=False,
            **DL_CONFIGS['tarnet_attention'],
        ),
        'x_train': x_train_cl,
        'x_test': x_test_cl,
        'is_dl': True,
    }

    return specs


def plot_curves_compare(preds_dict, y_true, treatment, split_name, top_k_percent=TOP_K_PERCENT):
    """
    Строит сравнение uplift curve и qini curve для набора моделей.

    Что делает:
    - на одной фигуре строит две панели:
    1) uplift curve
    2) qini curve
    - использует стили и цвета из MODEL_STYLES / MODEL_COLORS;
    - по умолчанию показывает только top_k_percent верхней части базы.
    """
    max_x = int(len(y_true) * top_k_percent)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for model_name, preds in preds_dict.items():
        style = MODEL_STYLES.get(model_name, {'ls': '-', 'alpha': 0.9})
        color = MODEL_COLORS.get(model_name, 'gray')

        plot_uplift_curve(
            y_true,
            preds,
            treatment,
            name=model_name,
            ax=axes[0],
            color=color,
            perfect=False,
            **style,
        )
        plot_qini_curve(
            y_true,
            preds,
            treatment,
            name=model_name,
            ax=axes[1],
            color=color,
            perfect=False,
            **style,
        )

    axes[0].set_title(f'Uplift curve ({split_name}, top {int(top_k_percent * 100)}%)', fontsize=16, fontweight='bold')
    axes[1].set_title(f'Qini curve ({split_name}, top {int(top_k_percent * 100)}%)', fontsize=16, fontweight='bold')

    for ax in axes:
        ax.set_xlim(0, max_x)
        ax.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout()
    plt.show()


def plot_percentile_grid_any(preds_dict, y_true, treatment, split_name, bins=10, ncols=2):
    """
    Строит сетку percentile uplift-графиков для произвольного числа моделей.

    Для каждой модели строятся два графика:
    1) uplift by percentile;
    2) response rate treatment vs control by percentile.

    Зачем:
    - позволяет визуально проверить, насколько хорошо модель поднимает
    убеждаемых клиентов в верхние бакеты;
    - показывает не только сам uplift, но и поведение treatment/control response rate.
    """
    model_names = list(preds_dict.keys())
    n_models = len(model_names)
    nrows = math.ceil(n_models / ncols)

    # Чуть выше фигура + больше расстояние между рядами
    fig = plt.figure(figsize=(9 * ncols, 7.4 * nrows))
    outer = fig.add_gridspec(
        nrows,
        ncols,
        wspace=0.24,
        hspace=0.42,
    )

    for i, model_name in enumerate(model_names):
        row, col = divmod(i, ncols)

        inner = outer[row, col].subgridspec(
            2,
            1,
            height_ratios=[1, 1.12],
            hspace=0.26,
        )

        ax_top = fig.add_subplot(inner[0])
        ax_bottom = fig.add_subplot(inner[1], sharex=ax_top)

        preds = preds_dict[model_name]
        df_pct = get_uplift_percentile_table(y_true, preds, treatment, bins=bins)

        x = np.arange(len(df_pct))
        width = 0.36

        ax_top.bar(
            x,
            df_pct['uplift'],
            yerr=df_pct['se_uplift'],
            capsize=2,
            color='red',
            alpha=0.95,
            edgecolor='black',
            linewidth=0.3,
            label='uplift',
        )
        ax_top.axhline(0, color='black', linewidth=1)
        ax_top.set_title(
            f"{model_name}\nUplift by percentile\nweighted average uplift = {weighted_average_uplift(y_true, preds, treatment):.4f}",
            fontsize=11,
            fontweight='bold',
            pad=8,
        )
        ax_top.set_ylabel('treatment RR - control RR', fontsize=10, labelpad=6)
        ax_top.legend(loc='upper right', fontsize=9, frameon=True)
        ax_top.grid(axis='y', linestyle='--', alpha=0.35)
        ax_top.tick_params(axis='x', labelbottom=False)

        ax_bottom.bar(
            x - width / 2,
            df_pct['response_rate_treatment'],
            width=width,
            yerr=df_pct['se_treatment'],
            capsize=2,
            color='forestgreen',
            alpha=0.95,
            edgecolor='black',
            linewidth=0.3,
            label='treatment RR',
        )
        ax_bottom.bar(
            x + width / 2,
            df_pct['response_rate_control'],
            width=width,
            yerr=df_pct['se_control'],
            capsize=2,
            color='orange',
            alpha=0.95,
            edgecolor='black',
            linewidth=0.3,
            label='control RR',
        )
        ax_bottom.set_title(
            'Response rate by percentile',
            fontsize=11,
            fontweight='bold',
            pad=6,
        )
        ax_bottom.set_ylabel('response rate', fontsize=10, labelpad=6)

        if row == nrows - 1:
            ax_bottom.set_xlabel('Percentile', fontsize=10, labelpad=10)
        else:
            ax_bottom.set_xlabel('')

        ax_bottom.set_xticks(x)
        ax_bottom.set_xticklabels(df_pct['percentile'], rotation=35)
        ax_bottom.tick_params(axis='x', pad=4)
        ax_bottom.legend(loc='upper right', fontsize=9, frameon=True)
        ax_bottom.grid(axis='y', linestyle='--', alpha=0.35)

    total_slots = nrows * ncols
    for j in range(n_models, total_slots):
        row, col = divmod(j, ncols)
        ax_empty = fig.add_subplot(outer[row, col])
        ax_empty.axis('off')

    fig.suptitle(
        f'Uplift by percentile | {split_name}',
        fontsize=16,
        fontweight='bold',
        y=0.995,
    )

    fig.subplots_adjust(top=0.94, bottom=0.06, left=0.07, right=0.98)
    plt.show()


def build_percentile_summary_table(preds_dict, y_true, treatment, split_name, bins=10):
    """
    Строит сводную таблицу фактического uplift по percentile-бакетам
    для набора моделей.

    Что делает:
    - для каждой модели вызывает get_uplift_percentile_table(...);
    - объединяет результаты в длинную таблицу;
    - строит pivot-таблицы по uplift, n_treatment, n_control и n_total;
    - формирует визуальную таблицу, где в каждой ячейке показаны:
        uplift,
        n_total,
        n_treatment,
        n_control;
    - дополнительно подсвечивает ячейки по величине uplift.

    Зачем:
    - это компактная табличная сводка вместо набора отдельных графиков;
    - удобно использовать для аналитики top-децилей и middle-tail поведения моделей.
    """
    frames = []
    for model_name, preds in preds_dict.items():
        tmp = get_uplift_percentile_table(y_true, preds, treatment, bins=bins).copy()
        tmp['Model'] = model_name
        tmp['Split'] = split_name
        frames.append(tmp)

    all_pct = pd.concat(frames, ignore_index=True)

    pivot_uplift = all_pct.pivot(index='percentile', columns='Model', values='uplift').reset_index()
    pivot_n_t = all_pct.pivot(index='percentile', columns='Model', values='n_treatment').reset_index()
    pivot_n_c = all_pct.pivot(index='percentile', columns='Model', values='n_control').reset_index()
    pivot_n = all_pct.assign(n_total=lambda d: d['n_treatment'] + d['n_control']).pivot(index='percentile', columns='Model', values='n_total').reset_index()

    display_df = pivot_uplift.copy().rename(columns={'percentile': 'Перцентиль (бакет)'})
    model_columns = [c for c in display_df.columns if c != 'Перцентиль (бакет)']

    for model_name in model_columns:
        display_df[model_name] = (
            pivot_uplift[model_name].apply(lambda x: f'{x:+.2%}' if pd.notna(x) else 'nan')
            + '\n'
            + 'n=' + pivot_n[model_name].round().astype('Int64').astype(str)
            + ', t=' + pivot_n_t[model_name].round().astype('Int64').astype(str)
            + ', c=' + pivot_n_c[model_name].round().astype('Int64').astype(str)
        )

    gmap_df = pivot_uplift[model_columns].copy()
    gmap_df.index = display_df.index

    print(f'Percentile uplift table | {split_name}')
    styled_df = (
        display_df.style
        .background_gradient(
            cmap='RdYlGn',
            subset=model_columns,
            gmap=gmap_df,
            vmin=-0.05,
            vmax=0.05,
            axis=None,
        )
        .set_properties(**{
            'text-align': 'center',
            'border': '1px solid black',
            'white-space': 'pre-line',
        })
        .set_table_styles([
            {
                'selector': 'th',
                'props': [
                    ('text-align', 'center'),
                    ('background-color', '#ff7f0e'),
                    ('border', '1px solid black'),
                ],
            }
        ])
        .hide(axis='index')
    )
    display(styled_df)
    return all_pct

def move_loaded_dl_model_to_device(model, device='cpu'):
    if hasattr(model, 'device'):
        model.device = device

    if hasattr(model, 'model_') and getattr(model, 'model_', None) is not None:
        model.model_ = model.model_.to(device)

    for sub_attr in ['treated_model_', 'control_model_']:
        if hasattr(model, sub_attr):
            sub_model = getattr(model, sub_attr, None)
            if sub_model is not None:
                if hasattr(sub_model, 'device'):
                    sub_model.device = device
                if hasattr(sub_model, 'model_') and getattr(sub_model, 'model_', None) is not None:
                    sub_model.model_ = sub_model.model_.to(device)

    return model

def make_model_key(model_name: str) -> str:
    """
    Преобразует имя модели в безопасный ключ для имени файла.

    Используется при сохранении и загрузке моделей:
    - приводит имя к нижнему регистру;
    - заменяет пробелы и дефисы на underscore;
    - убирает круглые скобки.

    Пример:
    'TARNet (Attention)' -> 'tarnet_attention'
    """
    return (
        model_name.lower()
        .replace(' ', '_')
        .replace('(', '')
        .replace(')', '')
        .replace('-', '_')
    )
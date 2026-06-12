import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklift.metrics import (
    uplift_auc_score,
    qini_auc_score,
    weighted_average_uplift,
    uplift_at_k,
    uplift_by_percentile
)

"""
Используемые метрики в uplfit проекте.
Включает как стандартные, так и бизнес метрики.

Также добавлена визуализация перцентильного представления uplift.
"""

# У реализации sklift не получается связка с текущей версией numpy, которую мы поставили для работы causalml на UpliftTree, поэтому напишем собственные графики
def get_uplift_percentile_table(y_true, uplift_preds, treatment, bins=10):
    y_true = np.asarray(y_true).reshape(-1)
    uplift_preds = np.asarray(uplift_preds).reshape(-1)
    treatment = np.asarray(treatment).reshape(-1)

    df = uplift_by_percentile(
        y_true=y_true,
        uplift=uplift_preds,
        treatment=treatment,
        strategy='overall',
        bins=bins,
        std=False,
        total=False,
        string_percentiles=True
    ).reset_index()

    if 'index' in df.columns:
        df = df.rename(columns={'index': 'percentile'})

    # На всякий случай приводим к числам
    numeric_cols = [
        'n_treatment',
        'n_control',
        'response_rate_treatment',
        'response_rate_control',
        'uplift'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Стандартные ошибки для error bars
    p_t = df['response_rate_treatment'].to_numpy()
    p_c = df['response_rate_control'].to_numpy()
    n_t = df['n_treatment'].to_numpy()
    n_c = df['n_control'].to_numpy()

    se_t = np.sqrt(np.where(n_t > 0, p_t * (1 - p_t) / n_t, np.nan))
    se_c = np.sqrt(np.where(n_c > 0, p_c * (1 - p_c) / n_c, np.nan))
    se_u = np.sqrt(np.nan_to_num(se_t, nan=0.0) ** 2 + np.nan_to_num(se_c, nan=0.0) ** 2)

    df['se_treatment'] = se_t
    df['se_control'] = se_c
    df['se_uplift'] = se_u

    return df

def plot_percentile_grid(preds_dict, y_true, treatment, split_name, bins=10):
    model_names = list(preds_dict.keys())

    fig = plt.figure(figsize=(18, 14))
    outer = fig.add_gridspec(2, 2, wspace=0.22, hspace=0.28)

    for i, model_name in enumerate(model_names):
        row, col = divmod(i, 2)

        inner = outer[row, col].subgridspec(
            2, 1,
            height_ratios=[1, 1.05],
            hspace=0.18
        )

        ax_top = fig.add_subplot(inner[0])
        ax_bottom = fig.add_subplot(inner[1], sharex=ax_top)

        preds = preds_dict[model_name]
        df = get_uplift_percentile_table(
            y_true=y_true,
            uplift_preds=preds,
            treatment=treatment,
            bins=bins
        )

        wau = weighted_average_uplift(y_true, preds, treatment)

        x = np.arange(len(df))
        width = 0.36

        # Верхний график: uplift by percentile
        ax_top.bar(
            x,
            df['uplift'],
            yerr=df['se_uplift'],
            capsize=2,
            color='red',
            alpha=0.95,
            edgecolor='black',
            linewidth=0.3,
            label='uplift'
        )
        ax_top.axhline(0, color='black', linewidth=1)

        ax_top.set_title(
            f'{model_name}\nUplift by percentile\nweighted average uplift = {wau:.4f}',
            fontsize=11,
            fontweight='bold'
        )
        ax_top.set_ylabel('Uplift = treatment response rate - control response rate', fontsize=10)
        ax_top.legend(loc='upper right', fontsize=9, frameon=True)
        ax_top.grid(axis='y', linestyle='--', alpha=0.35)
        ax_top.tick_params(axis='x', labelbottom=False)

        # Нижний график: response rate by percentile
        ax_bottom.bar(
            x - width / 2,
            df['response_rate_treatment'],
            width=width,
            yerr=df['se_treatment'],
            capsize=2,
            color='forestgreen',
            alpha=0.95,
            edgecolor='black',
            linewidth=0.3,
            label='treatment\nresponse rate'
        )

        ax_bottom.bar(
            x + width / 2,
            df['response_rate_control'],
            width=width,
            yerr=df['se_control'],
            capsize=2,
            color='orange',
            alpha=0.95,
            edgecolor='black',
            linewidth=0.3,
            label='control\nresponse rate'
        )

        ax_bottom.set_title('Response rate by percentile', fontsize=11, fontweight='bold')
        ax_bottom.set_ylabel('response rate', fontsize=10)
        ax_bottom.set_xlabel('Percentile', fontsize=10)
        ax_bottom.set_xticks(x)
        ax_bottom.set_xticklabels(df['percentile'], rotation=35)
        ax_bottom.legend(loc='upper right', fontsize=9, frameon=True)
        ax_bottom.grid(axis='y', linestyle='--', alpha=0.35)

    fig.suptitle(f'Uplift by percentile | {split_name}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.show()
    return fig

def _top_k_business_stats(y_true, uplift_preds, treatment, k=0.1):
    y_true = np.asarray(y_true).reshape(-1)
    uplift_preds = np.asarray(uplift_preds).reshape(-1)
    treatment = np.asarray(treatment).reshape(-1)

    n = len(y_true)
    n_target = int(np.ceil(n * k))

    order = np.argsort(-uplift_preds)
    top_idx = order[:n_target]

    y_top = y_true[top_idx]
    t_top = treatment[top_idx]

    treat_mask = (t_top == 1)
    ctrl_mask = (t_top == 0)

    n_treat = int(treat_mask.sum())
    n_ctrl = int(ctrl_mask.sum())

    treat_rate = y_top[treat_mask].mean() if n_treat > 0 else np.nan
    ctrl_rate = y_top[ctrl_mask].mean() if n_ctrl > 0 else np.nan

    abs_uplift = treat_rate - ctrl_rate if pd.notna(treat_rate) and pd.notna(ctrl_rate) else np.nan
    rel_lift = (treat_rate / ctrl_rate - 1.0) if pd.notna(ctrl_rate) and ctrl_rate > 0 else np.nan

    incremental_buyers = abs_uplift * n_target if pd.notna(abs_uplift) else np.nan
    incremental_buyers_per_1000 = abs_uplift * 1000 if pd.notna(abs_uplift) else np.nan

    return {
        'targeted_customers': n_target,
        'n_treatment': n_treat,
        'n_control': n_ctrl,
        'treatment_response_rate': treat_rate,
        'control_response_rate': ctrl_rate,
        'absolute_uplift': abs_uplift,
        'relative_lift': rel_lift,
        'incremental_buyers': incremental_buyers,
        'incremental_buyers_per_1000': incremental_buyers_per_1000
    }


def _flatten_topk_stats(stats_dict, prefix):
    return {f'{key}@{prefix}': value for key, value in stats_dict.items()}

def calculate_metrics(y_true, uplift_preds, treatment, business_k_list=(0.1, 0.2, 0.3, 0.5)):
    metrics = {
        "AUUC": uplift_auc_score(y_true, uplift_preds, treatment),
        "Qini": qini_auc_score(y_true, uplift_preds, treatment),
        "WAU": weighted_average_uplift(y_true, uplift_preds, treatment),
        "Uplift@10%": uplift_at_k(y_true, uplift_preds, treatment, strategy='overall', k=0.1),
        "Uplift@20%": uplift_at_k(y_true, uplift_preds, treatment, strategy='overall', k=0.2),
        "Uplift@30%": uplift_at_k(y_true, uplift_preds, treatment, strategy='overall', k=0.3),
        "Uplift@50%": uplift_at_k(y_true, uplift_preds, treatment, strategy='overall', k=0.5)
    }

    for k in business_k_list:
        label = f'{int(k * 100)}%'
        topk_stats = _top_k_business_stats(
            y_true=y_true,
            uplift_preds=uplift_preds,
            treatment=treatment,
            k=k
        )
        metrics.update(_flatten_topk_stats(topk_stats, label))

    return metrics
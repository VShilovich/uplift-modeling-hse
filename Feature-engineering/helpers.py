import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from pandas.api import types as ptypes
from sklearn.metrics import roc_auc_score, matthews_corrcoef

def cramers_v(x, y):
    ctab = pd.crosstab(x, y)
    if ctab.size == 0:
        return np.nan
    chi2 = chi2_contingency(ctab, correction=False)[0]
    n = ctab.to_numpy().sum()
    phi2 = chi2 / n
    r, k = ctab.shape
    # коррекция Беггена - из-за того, что категориальные признаки имеют разное кол-во категорий, решает переоценку силы связи
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)

    denom = min((k_corr - 1), (r_corr - 1))
    if denom <= 0: # если коэффициент с одним наблюдением, то тогда знаменатель на формуле коррекции будет ошибкой
        return np.nan

    return np.sqrt(phi2_corr / denom)

# 1) Подсчёт метрик
def feature_summary(df: pd.DataFrame, target: str, min_cat_freq: float = 0.005) -> pd.DataFrame:
    """
    Коротко:
      - numeric: Pearson, SMD, AUC
      - binary:  phi, SMD, AUC
      - categ:   eta^2 (через межгрупповую дисперсию), AUC
    df предочищен от пропусков. 
    """
    y = pd.to_numeric(df[target]).astype(int)
    out = []
    # Делим типы
    for col in (c for c in df.columns if c != target):
        s = df[col]
        if ptypes.is_bool_dtype(s) or (ptypes.is_numeric_dtype(s) and s.dropna().astype(float).nunique() <= 2 and set(s.dropna().astype(float)) <= {0.0, 1.0}):
            kind = "binary"
        elif ptypes.is_numeric_dtype(s):
            kind = "numeric"
        else:
            kind = "categorical"


        if kind == "numeric":
            x = pd.to_numeric(s).astype(float)
            r   = float(pd.Series(x).corr(y))
            # SMD
            x0, x1 = x[y == 0], x[y == 1]
            pooled = np.sqrt((x0.var(ddof=1) + x1.var(ddof=1)) / 2)
            smd = float((x1.mean() - x0.mean()) / pooled) if pooled > 0 else np.nan
            auc = float(roc_auc_score(y, x))
            eta2 = np.nan
            sign = r

        elif kind == "binary":
            xb = pd.to_numeric(s).astype(int)
            r   = float(matthews_corrcoef(y, xb))
            x0, x1 = xb[y == 0], xb[y == 1]
            pooled = np.sqrt((x0.var(ddof=1) + x1.var(ddof=1)) / 2)
            smd = float((x1.mean() - x0.mean()) / pooled) if pooled > 0 else np.nan
            auc = float(roc_auc_score(y, xb))
            eta2 = np.nan
            sign = r

        else:
            # редкие категории схлопываем (если есть)
            vc = s.value_counts(normalize=True)
            x = s.where(~s.isin(vc[vc < min_cat_freq].index), "RARE").astype(str)
            # eta^2: SS_between / SS_total
            mu  = y.mean()
            means = y.groupby(x).mean()
            counts = x.value_counts().reindex(means.index)
            ss_between = ((means - mu) ** 2 * counts).sum()
            ss_total   = ((y - mu) ** 2).sum()
            eta2 = float(ss_between / ss_total) if ss_total > 0 else np.nan
            # AUC
            tr = y.groupby(x).mean()
            score = x.map(tr).astype(float)
            auc = float(roc_auc_score(y, score))
            r = np.nan
            smd = np.nan
            sign = np.nan

        # нормировки 0-1
        corr = abs(r) if np.isfinite(r) else (eta2 if np.isfinite(eta2) else np.nan)
        auc_n = abs(auc - 0.5) * 2.0 if np.isfinite(auc) else np.nan
        smd_n = min(abs(smd) / 0.8, 1.0) if np.isfinite(smd) else np.nan
        parts = [p for p in (corr, auc_n, smd_n) if np.isfinite(p)]
        combined = float(np.mean(parts)) if parts else np.nan
        out.append(dict(
            feature=col, kind=kind,
            score=corr, sign=sign,
            auc=auc, auc_norm=auc_n,
            smd=smd, smd_norm=smd_n,
            eta2=eta2, combined=combined
        ))

    return (pd.DataFrame(out)
            .sort_values("combined", ascending=False, na_position="last")
            .reset_index(drop=True))

# 2) Графики
def plot_ranked(summary: pd.DataFrame, by: str = "combined", top_k: int = 25, title: str | None = None):
    sub = summary.sort_values(by, ascending=False).head(top_k).iloc[::-1]
    colors = sub["kind"].map({"numeric": "#5B8FF9", "binary": "#5AD8A6", "categorical": "#F6BD16"}).fillna("#BFBFBF")
    plt.figure(figsize=(10, max(4, 0.4 * len(sub))))
    plt.barh(sub["feature"], sub[by], color=colors)
    if by in {"score", "auc_norm", "smd_norm", "combined"}:
        plt.xlim(0, 1)

    # Подписи чисел у столбцов
    for y, v in enumerate(sub[by].values):
        if np.isfinite(v):
            plt.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=9)

    plt.title(title or f"Топ-{len(sub)} по {by}")
    plt.xlabel(by)
    plt.ylabel("")
    plt.tight_layout()
    plt.show()


def plot_bubble(summary: pd.DataFrame, title: str = "AUC vs SMD (норм.) — подписи всех признаков"):
    sub = summary.copy()
    sub["auc_norm"] = sub["auc_norm"].fillna(0.0)
    sub["smd_norm"] = sub["smd_norm"].fillna(0.0)
    colors = sub["kind"].map({"numeric": "#5B8FF9", "binary": "#5AD8A6", "categorical": "#F6BD16"}).fillna("#BFBFBF")

    # фиксированный размер для пузырьков (чтобы минимальные было все равно видно)
    size = 160

    plt.figure(figsize=(11, 7))
    plt.scatter(sub["auc_norm"], sub["smd_norm"], s=size, c=colors, alpha=0.85, edgecolor="white", linewidth=0.6)

    # подписи для всех точек
    for _, r in sub.iterrows():
        plt.text(r["auc_norm"] + 0.01, r["smd_norm"] + 0.01, str(r["feature"]), fontsize=9)

    plt.xlabel("|AUC - 0.5| × 2")
    plt.ylabel("|SMD| / 0.8")
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.3, linewidth=0.6)
    plt.title(title)
    plt.tight_layout()
    plt.show()
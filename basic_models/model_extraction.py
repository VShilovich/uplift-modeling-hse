from typing import List, Dict
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, clone

from catboost import CatBoostClassifier, CatBoostRegressor
from sklift.models import TwoModels

T_SOLVER_LOGREG_BEST_PARAMS: Dict[str, object] = {
    'C': 0.027301853380688412,
    'class_weight': None,
    'dual': False,
    'fit_intercept': True,
    'intercept_scaling': 1,
    'l1_ratio': None,
    'max_iter': 2000,
    'multi_class': 'deprecated',
    'n_jobs': -1,
    'penalty': 'l2',
    'random_state': None,
    'solver': 'lbfgs',
    'tol': 0.0001,
    'verbose': 0,
    'warm_start': False,
}


XL_OUTCOME_CATBOOST_PARAMS: Dict[str, object] = {
    'iterations': 147,
    'learning_rate': 0.03633155899517177,
    'depth': 5,
    'l2_leaf_reg': 5.860350130719548,
    'random_seed': 42,
    'verbose': 100,
    'allow_writing_files': False,
}

XL_EFFECT_CATBOOST_PARAMS: Dict[str, object] = {
    'iterations': 147,
    'learning_rate': 0.03633155899517177,
    'depth': 5,
    'l2_leaf_reg': 5.860350130719548,
    'loss_function': 'RMSE',
    'random_seed': 42,
    'verbose': 100,
    'allow_writing_files': False,
}

XL_PROPENSITY_CATBOOST_PARAMS: Dict[str, object] = {
    'iterations': 100,
    'learning_rate': 0.1,
    'depth': 4,
    'random_seed': 42,
    'verbose': 0,
    'allow_writing_files': False,
}

S_SOLVER_CATBOOST_PARAMS: Dict[str, object] = {
    'nan_mode': 'Min',
    'eval_metric': 'AUC',
    'iterations': 1000,
    'grow_policy': 'SymmetricTree',
    'l2_leaf_reg': 3,
    'subsample': 0.8,
    'use_best_model': True,
    'class_names': [0, 1],
    'random_seed': 67,
    'depth': 6,
    'border_count': 254,
    'loss_function': 'Logloss',
    'learning_rate': 0.111006,
    'bootstrap_type': 'MVS',
    'max_leaves': 64,
}

BASELINE_CATBOOST_PARAMS: Dict[str, object] = {
    'nan_mode': 'Min',
    'eval_metric': 'AUC',
    'iterations': 1000,
    'grow_policy': 'SymmetricTree',
    'l2_leaf_reg': 3,
    'subsample': 0.8,
    'use_best_model': True,
    'class_names': [0, 1],
    'random_seed': 67,
    'depth': 6,
    'border_count': 254,
    'loss_function': 'Logloss',
    'learning_rate': 0.111006,
    'bootstrap_type': 'MVS',
    'max_leaves': 64,
}

def build_baseline_catboost(cat_features: List[int]) -> CatBoostClassifier:
    params = BASELINE_CATBOOST_PARAMS.copy()
    model = CatBoostClassifier(
        **params,
        cat_features=cat_features,
    )
    return model

def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_logreg_pipeline(params: Dict[str, object], num_cols: List[str], cat_cols: List[str]) -> Pipeline:
    pre = build_preprocessor(num_cols, cat_cols)
    clf = LogisticRegression(**params)
    return Pipeline([("preprocess", pre), ("clf", clf)])


def build_t_learner_logreg(num_cols: List[str], cat_cols: List[str]) -> TwoModels:
    base_pipe = build_logreg_pipeline(T_SOLVER_LOGREG_BEST_PARAMS, num_cols, cat_cols)
    est_trmnt = clone(base_pipe)
    est_ctrl = clone(base_pipe)

    return TwoModels(
        estimator_trmnt=est_trmnt,
        estimator_ctrl=est_ctrl,
        method="vanilla"
    )


def build_s_learner_catboost(cat_features: List[int]) -> CatBoostClassifier:
    params = S_SOLVER_CATBOOST_PARAMS.copy()
    return CatBoostClassifier(
        **params,
        cat_features=cat_features,
    )


def predict_uplift_s_learner(model: CatBoostClassifier, X: pd.DataFrame, treatment_col: str):
    X_treat = X.copy()
    X_ctrl = X.copy()

    X_treat[treatment_col] = 1
    X_ctrl[treatment_col] = 0

    p1 = model.predict_proba(X_treat)[:, 1]
    p0 = model.predict_proba(X_ctrl)[:, 1]

    return p1 - p0


class MyXLearner(BaseEstimator):

    def __init__(self, outcome_learner, effect_learner, propensity_learner):
        self.outcome_learner = outcome_learner

        self.effect_learner = effect_learner
        self.propensity_learner = propensity_learner

        self.model_mu_0 = None
        self.model_mu_1 = None
        self.model_tau_0 = None
        self.model_tau_1 = None
        self.model_propensity = None

    def fit(self, X, y, treatment):
        y = np.asarray(y)
        t = np.asarray(treatment)

        X_c = X[t == 0]
        y_c = y[t == 0]
        X_t = X[t == 1]
        y_t = y[t == 1]

        # outcome models
        self.model_mu_0 = clone(self.outcome_learner)
        self.model_mu_0.fit(X_c, y_c)

        self.model_mu_1 = clone(self.outcome_learner)
        self.model_mu_1.fit(X_t, y_t)

        # propensity
        self.model_propensity = clone(self.propensity_learner)
        self.model_propensity.fit(X, t)

        # pseudo-effects
        mu1_on_c = self.model_mu_1.predict_proba(X_c)[:, 1]
        mu0_on_t = self.model_mu_0.predict_proba(X_t)[:, 1]

        D0 = mu1_on_c - y_c
        D1 = y_t - mu0_on_t

        # effect models
        self.model_tau_0 = clone(self.effect_learner)
        self.model_tau_0.fit(X_c, D0)

        self.model_tau_1 = clone(self.effect_learner)
        self.model_tau_1.fit(X_t, D1)

        return self

    def predict(self, X):
        tau0 = self.model_tau_0.predict(X)
        tau1 = self.model_tau_1.predict(X)

        g = self.model_propensity.predict_proba(X)[:, 1]

        return g * tau0 + (1 - g) * tau1


def build_x_learner_catboost(cat_features: List[str]) -> MyXLearner:
    outcome_est = CatBoostClassifier(
        **XL_OUTCOME_CATBOOST_PARAMS,
        cat_features=cat_features,
    )
    effect_est = CatBoostRegressor(
        **XL_EFFECT_CATBOOST_PARAMS,
        cat_features=cat_features,
    )
    propensity_est = CatBoostClassifier(
        **XL_PROPENSITY_CATBOOST_PARAMS,
        cat_features=cat_features,
    )

    return MyXLearner(
        outcome_learner=outcome_est,
        effect_learner=effect_est,
        propensity_learner=propensity_est,
    )

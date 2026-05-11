from typing import List, Dict, Any
import numpy as np
import pandas as pd
import copy

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, clone

from catboost import CatBoostClassifier, CatBoostRegressor
from sklift.models import TwoModels
from causalml.inference.tree import UpliftRandomForestClassifier

"""
Классические ML-подходы в моделировании, используемые в экспериментах.
"""

T_SOLVER_LOGREG_BEST_PARAMS: Dict[str, object] = {
    'C': 0.027301853380688412,
    'class_weight': None,
    'dual': False,
    'fit_intercept': True,
    'intercept_scaling': 1,
    'l1_ratio': None,
    'max_iter': 2000,
    'n_jobs': -1,
    'penalty': 'l2',
    'random_state': 42,
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

UPLIFT_RF_PARAMS: Dict[str, object] = {
    'control_name': 'control',
    'evaluationFunction': 'KL',
    'n_estimators': 100,
    'max_depth': 6,
    'min_samples_leaf': 200,
    'min_samples_treatment': 50,
    'n_jobs': -1,
    'random_state': 42
}

def _get_calibrated_params(base_params: Dict[str, Any]) -> Dict[str, Any]:
    params = base_params.copy()
    params['use_best_model'] = False
    params.pop('eval_metric', None)
    return params

def _extract_cat_features(estimator):
    """
    Достаем cat_features даже если estimator завернут в CalibratedClassifierCV.
    """
    if estimator is None:
        return []

    if hasattr(estimator, 'get_params'):
        params = estimator.get_params(deep=False)
        if 'cat_features' in params and params['cat_features'] is not None:
            return list(params['cat_features'])

    # sklearn calibration wrapper
    if hasattr(estimator, 'estimator'):
        return _extract_cat_features(estimator.estimator)

    if hasattr(estimator, 'base_estimator'):
        return _extract_cat_features(estimator.base_estimator)

    return []


def _sanitize_catboost_input(X, cat_features):
    """
    Для CatBoost все categorical columns приводим к строке
    и заменяем NaN на специальный токен.
    """
    if not isinstance(X, pd.DataFrame):
        return X

    if not cat_features:
        return X

    X_out = X.copy()

    for col in cat_features:
        if col in X_out.columns:
            X_out[col] = X_out[col].astype('object')
            X_out[col] = X_out[col].where(X_out[col].notna(), '__nan__')
            X_out[col] = X_out[col].astype(str)

    return X_out

def build_baseline_catboost(cat_features: List[int], use_calibration: bool = False) -> CatBoostClassifier:
    params = BASELINE_CATBOOST_PARAMS.copy()
    if use_calibration:
        params = _get_calibrated_params(params)
    
    return CatBoostClassifier(
        **params,
        cat_features=cat_features,
    )

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

def build_s_learner_catboost(cat_features: List[int], use_calibration: bool = False) -> CatBoostClassifier:
    params = S_SOLVER_CATBOOST_PARAMS.copy()
    if use_calibration:
        params = _get_calibrated_params(params)

    cat_features = tuple(cat_features)

    return CatBoostClassifier(
        **params,
        cat_features=cat_features,
    )

def build_uplift_random_forest(control_name: str = 'control') -> UpliftRandomForestClassifier:
    params = UPLIFT_RF_PARAMS.copy()
    params['control_name'] = control_name
        
    return UpliftRandomForestClassifier(**params)

def predict_uplift_s_learner(model: CatBoostClassifier, X: pd.DataFrame, treatment_col: str):
    X_treat = X.copy()
    X_ctrl = X.copy()

    X_treat[treatment_col] = 1
    X_ctrl[treatment_col] = 0

    cat_features = _extract_cat_features(model)
    X_treat = _sanitize_catboost_input(X_treat, cat_features)
    X_ctrl = _sanitize_catboost_input(X_ctrl, cat_features)

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

        outcome_cat_features = _extract_cat_features(self.outcome_learner)
        effect_cat_features = _extract_cat_features(self.effect_learner)
        propensity_cat_features = _extract_cat_features(self.propensity_learner)

        X_outcome = _sanitize_catboost_input(X, outcome_cat_features)
        X_effect = _sanitize_catboost_input(X, effect_cat_features)
        X_propensity = _sanitize_catboost_input(X, propensity_cat_features)

        X_c_out = X_outcome[t == 0]
        y_c = y[t == 0]
        X_t_out = X_outcome[t == 1]
        y_t = y[t == 1]

        X_c_eff = X_effect[t == 0]
        X_t_eff = X_effect[t == 1]

        # outcome models
        self.model_mu_0 = copy.deepcopy(self.outcome_learner)
        self.model_mu_0.fit(X_c_out, y_c)

        self.model_mu_1 = copy.deepcopy(self.outcome_learner)
        self.model_mu_1.fit(X_t_out, y_t)

        # propensity
        self.model_propensity = copy.deepcopy(self.propensity_learner)
        self.model_propensity.fit(X_propensity, t)

        # pseudo-effects
        mu1_on_c = self.model_mu_1.predict_proba(X_c_out)[:, 1]
        mu0_on_t = self.model_mu_0.predict_proba(X_t_out)[:, 1]

        D0 = mu1_on_c - y_c
        D1 = y_t - mu0_on_t

        # effect models
        self.model_tau_0 = copy.deepcopy(self.effect_learner)
        self.model_tau_0.fit(X_c_eff, D0)

        self.model_tau_1 = copy.deepcopy(self.effect_learner)
        self.model_tau_1.fit(X_t_eff, D1)

        return self

    def predict(self, X):
        X_tau = _sanitize_catboost_input(X, _extract_cat_features(self.model_tau_0))
        X_prop = _sanitize_catboost_input(X, _extract_cat_features(self.model_propensity))

        tau0 = self.model_tau_0.predict(X_tau)
        tau1 = self.model_tau_1.predict(X_tau)

        g = self.model_propensity.predict_proba(X_prop)[:, 1]

        return g * tau0 + (1 - g) * tau1


def build_x_learner_catboost(cat_features: List[str], use_calibration: bool = False) -> "MyXLearner":
    outcome_params = XL_OUTCOME_CATBOOST_PARAMS.copy()
    propensity_params = XL_PROPENSITY_CATBOOST_PARAMS.copy()

    if use_calibration:
        outcome_params = _get_calibrated_params(outcome_params)
        propensity_params = _get_calibrated_params(propensity_params)

    cat_features = tuple(cat_features)

    outcome_est = CatBoostClassifier(
        **outcome_params,
        cat_features=cat_features,
    )
    effect_est = CatBoostRegressor(
        **XL_EFFECT_CATBOOST_PARAMS,
        cat_features=cat_features,
    )
    propensity_est = CatBoostClassifier(
        **propensity_params,
        cat_features=cat_features,
    )

    return MyXLearner(
        outcome_learner=outcome_est,
        effect_learner=effect_est,
        propensity_learner=propensity_est,
    )
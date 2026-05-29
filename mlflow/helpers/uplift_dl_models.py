from __future__ import annotations

"""
DL-модели для uplift-моделирования на табличных данных.

- вспомогательные функции для приведения входов к pandas / numpy и фиксации random seed;
- базовые torch-модули для бинарной классификации и mixed tabular input;
- нейросетевые uplift-модели:
    1) T-Learner NN
    2) S-Learner NN
    3) Classic TARNet
    4) Attention TARNet
- общий интерфейс fit / predict / predict_components для использования в sklearn-like пайплайне.

Основная идея:
- T-Learner NN обучает две отдельные модели вероятности отклика:
  одну на treatment, вторую на control.
- S-Learner NN обучает одну модель, которая получает treatment как входной признак.
- TARNet-модели учат общее представление клиента и затем два outcome-head:
  для treatment и для control.
"""

import copy
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset


def set_global_seed(seed: int = 42) -> None:
    """
    Фиксирует random seed во всех основных генераторах случайности.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _as_dataframe(X: pd.DataFrame | np.ndarray, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Приводитт вход к pandas DataFrame.
    - если на вход уже пришел DataFrame, возвращается его копия;
    - если пришел numpy-массив, он преобразуется в DataFrame;
    - при наличии columns сохраняются имена признаков.

    Нужно для того, чтобы downstream-код всегда работал с колонками по именам.
    """
    if isinstance(X, pd.DataFrame):
        return X.copy()
    if columns is None:
        return pd.DataFrame(X)
    return pd.DataFrame(X, columns=list(columns))


def _as_numpy_1d(y: Sequence) -> np.ndarray:
    """
    Преобразует входную последовательность в одномерный numpy-массив.

    Используется для y и treatment-флагов, чтобы:
    - убрать лишние размерности;
    - унифицировать дальнейшие операции индексации и маскирования.
    """
    arr = np.asarray(y).reshape(-1)
    return arr


def _make_ohe():
    """
    Создает OneHotEncoder, совместимый с разными версиями scikit-learn.
    """
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)


class EarlyStopping:
    """
    Ранняя остановка обучения по значению validation loss.

    - хранит лучшее значение функции потерь на валидации;
    - запоминает лучшие веса модели;
    - останавливает обучение, если val_loss достаточно долго не улучшается.
    """
    def __init__(self, patience: Optional[int] = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.best_state = None

    def step(self, current_loss: Optional[float], model: nn.Module) -> bool:
        """
        Проверяет, стало ли качество на валидации лучше.
        """
        if self.patience is None or current_loss is None or np.isnan(current_loss):
            return False

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = float(current_loss)
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False

        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module) -> None:
        """
        Восстанавливает в модель веса лучшей эпохи.
        """
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class MixedTabularDataset(Dataset):
    """
    PyTorch Dataset для табличных mixed-features:
    - числовая часть x_num;
    - категориальная часть x_cat;
    - опционально y и treatment t.

    Нужен как унифицированный контейнер данных для S-Learner NN и TARNet-моделей,
    где числовые и категориальные признаки обрабатываются по-разному.
    """
    def __init__(self, x_num: np.ndarray, x_cat: np.ndarray, y: Optional[np.ndarray] = None, t: Optional[np.ndarray] = None):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        self.t = torch.tensor(t, dtype=torch.float32) if t is not None else None

    def __len__(self) -> int:
        return len(self.x_num)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {'x_num': self.x_num[idx], 'x_cat': self.x_cat[idx]}
        if self.y is not None:
            item['y'] = self.y[idx]
        if self.t is not None:
            item['t'] = self.t[idx]
        return item


class BinaryMLP(nn.Module):
    """
    Базовый MLP для бинарной классификации.

    Архитектура:
    - несколько полносвязных слоев;
    - BatchNorm;
    - ReLU;
    - Dropout;
    - выход через sigmoid.

    Используется как внутренняя сеть для BinaryMLPClassifierTorch, а через него - внутри T-Learner NN.
    """
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (64, 32), dropout: float = 0.2):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(x)).squeeze(-1)


class BinaryMLPClassifierTorch(BaseEstimator, ClassifierMixin):
    """
    Sklearn-like обертка над простым бинарным MLP на PyTorch.

    Нужна для того, чтобы:
    - обучать обычную бинарную нейросеть с интерфейсом fit / predict / predict_proba;
    - использовать эту нейросеть как базовый классификатор внутри T-Learner NN.

    Делает:
    - train loop на BCE loss;
    - early stopping по eval_set;
    - predict_proba в sklearn-формате;
    - predict по порогу 0.5.

    Сама по себе не является uplift-моделью.
    Она используется как блок для оценки вероятности отклика внутри treatment и control-веток.
    """
    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.22338843352395796,
        lr: float = 0.00032488253588586755,
        batch_size: int = 512,
        epochs: int = 60,
        weight_decay: float = 1e-4,
        patience: Optional[int] = 8,
        min_delta: float = 1e-4,
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_state = random_state
        self.verbose = verbose
        self.model_: Optional[BinaryMLP] = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        set_global_seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = _as_numpy_1d(y).astype(np.float32)

        if self.input_dim is None:
            self.input_dim_ = X.shape[1]
        else:
            self.input_dim_ = self.input_dim

        self.model_ = BinaryMLP(
            input_dim=self.input_dim_,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True
        )

        val_tensors = None
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = _as_numpy_1d(y_val).astype(np.float32)
            val_tensors = (
                torch.tensor(X_val, dtype=torch.float32, device=self.device),
                torch.tensor(y_val, dtype=torch.float32, device=self.device)
            )

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCELoss()
        early_stopping = EarlyStopping(self.patience, self.min_delta)

        for epoch in range(self.epochs):
            self.model_.train()
            batch_losses = []
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                preds = self.model_(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            val_loss = None
            if val_tensors is not None:
                self.model_.eval()
                with torch.no_grad():
                    val_preds = self.model_(val_tensors[0])
                    val_loss = criterion(val_preds, val_tensors[1]).item()

            if self.verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch == self.epochs - 1):
                msg = f'[BinaryMLP] epoch {epoch + 1}/{self.epochs} | train_loss={np.mean(batch_losses):.4f}'
                if val_loss is not None:
                    msg += f' | val_loss={val_loss:.4f}'
                print(msg)

            should_stop = early_stopping.step(val_loss, self.model_)
            if should_stop:
                break

        early_stopping.restore(self.model_)
        return self

    def predict_proba(self, X):
        if self.model_ is None:
            raise ValueError('Model is not fitted.')

        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)

        self.model_.eval()
        with torch.no_grad():
            preds = self.model_(X_tensor).detach().cpu().numpy().reshape(-1, 1)

        return np.hstack([1.0 - preds, preds])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class MixedInputMLP(nn.Module):
    """
    MLP для mixed tabular input:
    - числовые признаки подаются напрямую;
    - категориальные признаки преобразуются в embedding-векторы;
    - treatment может подаваться как отдельный входной признак.

    Используется в S-Learner NN.
    """
    def __init__(
        self,
        numerical_dim: int,
        categorical_cardinalities: Sequence[int],
        cat_embed_dim: int = 8,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.2,
        use_treatment_input: bool = True,
    ):
        super().__init__()
        self.numerical_dim = numerical_dim
        self.use_treatment_input = use_treatment_input
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality + 1, cat_embed_dim) for cardinality in categorical_cardinalities
        ])

        cat_total_dim = len(categorical_cardinalities) * cat_embed_dim
        input_dim = numerical_dim + cat_total_dim + (1 if use_treatment_input else 0)

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, treatment: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Строит итоговый входной вектор из:
        - числовых признаков,
        - embedding-представлений категориальных признаков,
        - treatment-признака (если включен),
        и возвращает вероятность положительного отклика.
        """
        if x_cat.size(1) > 0:
            cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x_cat_emb = torch.cat(cat_embs, dim=1)
        else:
            x_cat_emb = torch.zeros((x_num.size(0), 0), dtype=x_num.dtype, device=x_num.device)

        pieces = [x_num, x_cat_emb]
        if self.use_treatment_input:
            if treatment is None:
                raise ValueError('Treatment tensor is required for this model.')
            pieces.append(treatment.unsqueeze(1) if treatment.ndim == 1 else treatment)

        x = torch.cat(pieces, dim=1)
        return torch.sigmoid(self.network(x)).squeeze(-1)


class BaseMixedFeatures:
    """
    Базовый класс с preprocessing-логикой для mixed tabular моделей.

    Что делает:
    - отдельно хранит списки числовых и категориальных признаков;
    - для числовых признаков обучает imputer + scaler;
    - для категориальных признаков строит словари кодирования в integer id;
    - преобразует входной DataFrame в пару матриц:
        x_num : float32
        x_cat : int64

    Используется как общий миксинг для S-Learner NN и TARNet-моделей.
    """
    def __init__(self, num_cols: Sequence[str], cat_cols: Sequence[str]):
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)


    def _sanitize_cat_series(self, s: pd.Series) -> pd.Series:
        """
        Приводит категориальную колонку к строковому виду:
        - пропуски заменяет на специальный токен '__nan__';
        - все значения переводит в str.
        """
        s = s.astype('object')
        s = s.where(s.notna(), '__nan__')
        return s.astype(str)

    def _fit_mixed_preprocessor(self, X: pd.DataFrame) -> None:
        """
        Обучает preprocessing для mixed-features:
        - числовые признаки: median imputation + standard scaling;
        - категориальные признаки: построение словаря value -> integer id.

        После вызова сохраняются:
        - num_imputer_
        - scaler_
        - category_maps_
        - cardinalities_
        """
        X = _as_dataframe(X)

        self.num_imputer_ = SimpleImputer(strategy='median')
        self.scaler_ = StandardScaler()

        if len(self.num_cols) > 0:
            x_num = self.num_imputer_.fit_transform(X[self.num_cols])
            self.scaler_.fit(x_num)

        self.category_maps_: Dict[str, Dict[str, int]] = {}
        self.cardinalities_: List[int] = []

        for col in self.cat_cols:
            vals = self._sanitize_cat_series(X[col])
            uniques = pd.Index(vals).drop_duplicates().tolist()
            mapping = {value: idx + 1 for idx, value in enumerate(uniques)}
            self.category_maps_[col] = mapping
            self.cardinalities_.append(len(uniques))

    def _transform_mixed(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Преобразует DataFrame в две матрицы:
        - x_num для числовых признаков;
        - x_cat для категориальных признаков.

        Категориальные признаки кодируются целыми индексами,
        которые далее будут подаваться в embedding-слои.
        """
        X = _as_dataframe(X)

        if len(self.num_cols) > 0:
            x_num = self.num_imputer_.transform(X[self.num_cols])
            x_num = self.scaler_.transform(x_num).astype(np.float32)
        else:
            x_num = np.zeros((len(X), 0), dtype=np.float32)

        if len(self.cat_cols) > 0:
            x_cat = np.zeros((len(X), len(self.cat_cols)), dtype=np.int64)
            for i, col in enumerate(self.cat_cols):
                mapping = self.category_maps_[col]
                vals = self._sanitize_cat_series(X[col]).tolist()
                x_cat[:, i] = np.array([mapping.get(v, 0) for v in vals], dtype=np.int64)
        else:
            x_cat = np.zeros((len(X), 0), dtype=np.int64)

        return x_num, x_cat



class SLearnerNNUplift(BaseEstimator, ClassifierMixin, BaseMixedFeatures):
    """
    Нейросетевой S-Learner для uplift-моделирования.

    Идея:
    - обучается одна модель отклика P(y=1 | X, t),
    где treatment t подается как обычный входной признак;
    - uplift считается как разность двух прогонов одной и той же модели:
        uplift(x) = p_treatment(x) - p_control(x)

    Реализация:
    - числовые признаки масштабируются;
    - категориальные признаки переводятся в embeddings;
    - treatment подается отдельным входом;
    - используется early stopping по validation loss.
    """
    def __init__(
        self,
        num_cols: Sequence[str],
        cat_cols: Sequence[str],
        cat_embed_dim: int = 8,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.20295116715006706,
        lr: float = 0.0009865977951798554,
        batch_size: int = 512,
        epochs: int = 30,
        weight_decay: float = 1e-4,
        patience: Optional[int] = 8,
        min_delta: float = 1e-4,
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
    ):
        BaseMixedFeatures.__init__(self, num_cols, cat_cols)
        self.cat_embed_dim = cat_embed_dim
        self.hidden_dims = list(hidden_dims)
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_state = random_state
        self.verbose = verbose
        self.model_: Optional[MixedInputMLP] = None
        self.classes_ = np.array([0, 1])

    def fit(
        self,
        X,
        y,
        t,
        eval_set: Optional[Tuple[pd.DataFrame, np.ndarray, np.ndarray]] = None,
    ):
        set_global_seed(self.random_state)

        X = _as_dataframe(X)
        y = _as_numpy_1d(y).astype(np.float32)
        t = _as_numpy_1d(t).astype(np.float32)

        self._fit_mixed_preprocessor(X)
        x_num, x_cat = self._transform_mixed(X)

        if (t == 1).sum() == 0 or (t == 0).sum() == 0:
            raise ValueError(f'{self.__class__.__name__}.fit() requires both treatment and control samples.')

        val_data = None
        if eval_set is not None:
            X_val, y_val, t_val = eval_set
            X_val = _as_dataframe(X_val)
            y_val = _as_numpy_1d(y_val).astype(np.float32)
            t_val = _as_numpy_1d(t_val).astype(np.float32)
            x_num_val, x_cat_val = self._transform_mixed(X_val)
            val_data = (x_num_val, x_cat_val, y_val, t_val)

        self.model_ = MixedInputMLP(
            numerical_dim=len(self.num_cols),
            categorical_cardinalities=self.cardinalities_,
            cat_embed_dim=self.cat_embed_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            use_treatment_input=True,
        ).to(self.device)

        dataset = MixedTabularDataset(x_num, x_cat, y=y, t=t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.BCELoss()
        early_stopping = EarlyStopping(self.patience, self.min_delta)

        for epoch in range(self.epochs):
            self.model_.train()
            train_losses = []

            for batch in loader:
                x_num_b = batch['x_num'].to(self.device)
                x_cat_b = batch['x_cat'].to(self.device)
                y_b = batch['y'].to(self.device)
                t_b = batch['t'].to(self.device)

                optimizer.zero_grad()
                preds = self.model_(x_num_b, x_cat_b, t_b)
                loss = criterion(preds, y_b)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            val_loss = None
            if val_data is not None:
                self.model_.eval()
                with torch.no_grad():
                    x_num_val_t = torch.tensor(val_data[0], dtype=torch.float32, device=self.device)
                    x_cat_val_t = torch.tensor(val_data[1], dtype=torch.long, device=self.device)
                    y_val_t = torch.tensor(val_data[2], dtype=torch.float32, device=self.device)
                    t_val_t = torch.tensor(val_data[3], dtype=torch.float32, device=self.device)

                    preds_val = self.model_(x_num_val_t, x_cat_val_t, t_val_t)
                    val_loss = criterion(preds_val, y_val_t).item()

            if self.verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch == self.epochs - 1):
                msg = f'[SLearnerNN] epoch {epoch + 1}/{self.epochs} | train_loss={np.mean(train_losses):.4f}'
                if val_loss is not None:
                    msg += f' | val_loss={val_loss:.4f}'
                print(msg)

            should_stop = early_stopping.step(val_loss, self.model_)
            if should_stop:
                break

        early_stopping.restore(self.model_)
        return self

    def predict_components(self, X) -> Tuple[np.ndarray, np.ndarray]:
        if self.model_ is None:
            raise ValueError('Model is not fitted.')

        X = _as_dataframe(X)
        x_num, x_cat = self._transform_mixed(X)

        x_num_t = torch.tensor(x_num, dtype=torch.float32, device=self.device)
        x_cat_t = torch.tensor(x_cat, dtype=torch.long, device=self.device)
        ones = torch.ones((len(X),), dtype=torch.float32, device=self.device)
        zeros = torch.zeros((len(X),), dtype=torch.float32, device=self.device)

        self.model_.eval()
        with torch.no_grad():
            p_t = self.model_(x_num_t, x_cat_t, ones).detach().cpu().numpy()
            p_c = self.model_(x_num_t, x_cat_t, zeros).detach().cpu().numpy()

        return p_t, p_c

    def predict(self, X) -> np.ndarray:
        p_t, p_c = self.predict_components(X)
        return p_t - p_c


class TLearnerNNUplift(BaseEstimator, ClassifierMixin):
    """
    Нейросетевой T-Learner для uplift-моделирования.

    Идея:
    - обучаются две независимые модели отклика:
        1) на treatment-группе,
        2) на control-группе;
    - uplift считается как разность их предсказаний:
        uplift(x) = p_treatment_model(x) - p_control_model(x)

    Реализация:
    - preprocessing строится через sklearn ColumnTransformer;
    - числовые признаки: median imputation + scaling;
    - категориальные признаки: imputation + one-hot encoding;
    - каждая из двух веток обучается как отдельный BinaryMLPClassifierTorch.
    """
    def __init__(
        self,
        num_cols: Sequence[str],
        cat_cols: Sequence[str],
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.22338843352395796,
        lr: float = 0.00032488253588586755,
        batch_size: int = 512,
        epochs: int = 60,
        weight_decay: float = 1e-4,
        patience: Optional[int] = 8,
        min_delta: float = 1e-4,
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.hidden_dims = list(hidden_dims)
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_state = random_state
        self.verbose = verbose
        self.classes_ = np.array([0, 1])

    def _build_preprocessor(self) -> ColumnTransformer:
        """
        Создает sklearn-preprocessor для plain tabular features:
        - числовые колонки -> median imputer + scaler;
        - категориальные колонки -> most frequent imputer + one-hot encoder.
        """
        transformers = []
        if len(self.num_cols) > 0:
            transformers.append((
                'num',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]),
                self.num_cols
            ))
        if len(self.cat_cols) > 0:
            transformers.append((
                'cat',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ohe', _make_ohe())
                ]),
                self.cat_cols
            ))
        return ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0.0)

    def fit(
        self,
        X,
        y,
        t,
        eval_set: Optional[Tuple[pd.DataFrame, np.ndarray, np.ndarray]] = None,
    ):
        set_global_seed(self.random_state)

        X = _as_dataframe(X)
        y = _as_numpy_1d(y).astype(np.float32)
        t = _as_numpy_1d(t).astype(int)

        self.preprocessor_ = self._build_preprocessor()
        x_processed = self.preprocessor_.fit_transform(X)
        if hasattr(x_processed, 'toarray'):
            x_processed = x_processed.toarray()
        x_processed = np.asarray(x_processed, dtype=np.float32)

        x_val_processed = y_val = t_val = None
        if eval_set is not None:
            X_val, y_val, t_val = eval_set
            X_val = _as_dataframe(X_val)
            y_val = _as_numpy_1d(y_val).astype(np.float32)
            t_val = _as_numpy_1d(t_val).astype(int)

            x_val_processed = self.preprocessor_.transform(X_val)
            if hasattr(x_val_processed, 'toarray'):
                x_val_processed = x_val_processed.toarray()
            x_val_processed = np.asarray(x_val_processed, dtype=np.float32)

        self.treated_model_ = BinaryMLPClassifierTorch(
            input_dim=x_processed.shape[1],
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            lr=self.lr,
            batch_size=self.batch_size,
            epochs=self.epochs,
            weight_decay=self.weight_decay,
            patience=self.patience,
            min_delta=self.min_delta,
            device=self.device,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self.control_model_ = BinaryMLPClassifierTorch(
            input_dim=x_processed.shape[1],
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            lr=self.lr,
            batch_size=self.batch_size,
            epochs=self.epochs,
            weight_decay=self.weight_decay,
            patience=self.patience,
            min_delta=self.min_delta,
            device=self.device,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        train_treat_mask = (t == 1)
        train_ctrl_mask = (t == 0)

        eval_treat = eval_ctrl = None
        if x_val_processed is not None:
            eval_treat_mask = (t_val == 1)
            eval_ctrl_mask = (t_val == 0)
            eval_treat = (
                x_val_processed[eval_treat_mask],
                y_val[eval_treat_mask]
            ) if eval_treat_mask.sum() > 0 else None
            eval_ctrl = (
                x_val_processed[eval_ctrl_mask],
                y_val[eval_ctrl_mask]
            ) if eval_ctrl_mask.sum() > 0 else None

        if train_treat_mask.sum() == 0 or train_ctrl_mask.sum() == 0:
            raise ValueError('Both treatment and control samples are required for TLearnerNNUplift.fit().')

        self.treated_model_.fit(
            x_processed[train_treat_mask],
            y[train_treat_mask],
            eval_set=eval_treat
        )
        self.control_model_.fit(
            x_processed[train_ctrl_mask],
            y[train_ctrl_mask],
            eval_set=eval_ctrl
        )

        return self

    def predict_components(self, X) -> Tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, 'preprocessor_'):
            raise ValueError('Model is not fitted.')
        X = _as_dataframe(X)
        x_processed = self.preprocessor_.transform(X)
        if hasattr(x_processed, 'toarray'):
            x_processed = x_processed.toarray()
        x_processed = np.asarray(x_processed, dtype=np.float32)

        p_t = self.treated_model_.predict_proba(x_processed)[:, 1]
        p_c = self.control_model_.predict_proba(x_processed)[:, 1]
        return p_t, p_c

    def predict(self, X) -> np.ndarray:
        p_t, p_c = self.predict_components(X)
        return p_t - p_c


class ClassicTARNet(nn.Module):
    """
    Torch-реализация классического TARNet.

    Архитектура:
    - embeddings для категориальных признаков;
    - общая shared representation для всех клиентов;
    - две отдельные головы:
        1) treatment head
        2) control head

    Возвращает две вероятности:
    - p_treatment
    - p_control

    Сама по себе не содержит sklearn-like интерфейса. Для обучения в пайплайне используется через ClassicTARNetUplift.
    """
    def __init__(
        self,
        n_num_features: int,
        cardinalities: Sequence[int],
        cat_emb_dim: int = 8,
        shared_hidden: Sequence[int] = (128, 64),
        head_hidden: Sequence[int] = (32, 16),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality + 1, cat_emb_dim) for cardinality in cardinalities
        ])

        input_dim = n_num_features + len(cardinalities) * cat_emb_dim

        shared_layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in shared_hidden:
            shared_layers.append(nn.Linear(prev_dim, h_dim))
            shared_layers.append(nn.BatchNorm1d(h_dim))
            shared_layers.append(nn.Mish())
            shared_layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.shared_representation = nn.Sequential(*shared_layers)

        self.treatment_head = self._make_head(prev_dim, head_hidden, dropout)
        self.control_head = self._make_head(prev_dim, head_hidden, dropout)

    @staticmethod
    def _make_head(input_dim: int, hidden_dims: Sequence[int], dropout: float) -> nn.Sequential:
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Mish())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x_cat.size(1) > 0:
            cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x_cat_emb = torch.cat(cat_embs, dim=1)
            x = torch.cat([x_num, x_cat_emb], dim=1)
        else:
            x = x_num

        rep = self.shared_representation(x)
        p_treatment = torch.sigmoid(self.treatment_head(rep)).squeeze(-1)
        p_control = torch.sigmoid(self.control_head(rep)).squeeze(-1)
        return p_treatment, p_control


class NumericTokenizer(nn.Module):
    """
    Токенизатор числовых признаков для attention-модели.

    Каждый числовой признак переводится в отдельный токен размерности d_model.
    Используется в AttentionTARNet как аналог токенайзера для табличных числовых полей.
    """
    def __init__(self, n_num_features: int, d_model: int):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_num_features, d_model))
        self.biases = nn.Parameter(torch.zeros(n_num_features, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weights + self.biases


class CategoricalTokenizer(nn.Module):
    """
    Токенизатор категориальных признаков для attention-модели.

    Каждая категориальная колонка переводится в embedding-токен размерности d_model.
    Используется в AttentionTARNet вместе с NumericTokenizer.
    """
    def __init__(self, cardinalities: Sequence[int], d_model: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality + 1, d_model) for cardinality in cardinalities
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 0:
            return torch.zeros((x.size(0), 0, 0), device=x.device)
        return torch.stack([emb(x[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)


class AttentionTARNet(nn.Module):
    """
    Torch-реализация TARNet с self-attention над табличными токенами.

    Архитектура:
    - числовые признаки превращаются в токены;
    - категориальные признаки превращаются в embedding-токены;
    - добавляется CLS-token;
    - токены проходят через TransformerEncoder;
    - из CLS-представления строятся две головы:
        1) treatment head
        2) control head

    Идея:
    attention-механизм позволяет модели выучивать более сложные взаимодействия
    между признаками клиента, чем обычный MLP.
    """
    def __init__(
        self,
        n_num_features: int,
        cardinalities: Sequence[int],
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        mlp_hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_num_features = n_num_features
        self.n_cat_features = len(cardinalities)
        self.d_model = d_model

        self.num_tokenizer = NumericTokenizer(n_num_features, d_model)
        self.cat_tokenizer = CategoricalTokenizer(cardinalities, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.treatment_head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )
        self.control_head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_tokens = self.num_tokenizer(x_num)

        if x_cat.size(1) > 0:
            cat_tokens = self.cat_tokenizer(x_cat)
            tokens = torch.cat([num_tokens, cat_tokens], dim=1)
        else:
            tokens = num_tokens

        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        shared = self.transformer(tokens)
        cls_repr = shared[:, 0, :]

        p_treatment = torch.sigmoid(self.treatment_head(cls_repr)).squeeze(-1)
        p_control = torch.sigmoid(self.control_head(cls_repr)).squeeze(-1)
        return p_treatment, p_control


class _BaseTARNetUplift(BaseEstimator, ClassifierMixin, BaseMixedFeatures):
    """
    Sklearn-like wrapper для TARNet-подходов.

    Что делает:
    - хранит общую train / validation логику;
    - использует BaseMixedFeatures для preprocessing;
    - обучает модель по factual loss:
        treatment-объекты оптимизируют treatment head,
        control-объекты оптимизируют control head;
    - реализует общий интерфейс fit / predict / predict_components.
    """
    def __init__(
        self,
        num_cols: Sequence[str],
        cat_cols: Sequence[str],
        lr: float,
        n_epochs: int,
        batch_size: int,
        weight_decay: float = 1e-4,
        patience: Optional[int] = 8,
        min_delta: float = 1e-4,
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
    ):
        BaseMixedFeatures.__init__(self, num_cols, cat_cols)
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.patience = patience
        self.min_delta = min_delta
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_state = random_state
        self.verbose = verbose
        self.classes_ = np.array([0, 1])

    def _build_model(self) -> nn.Module:
        """
        Должен вернуть конкретную torch-модель TARNet-архитектуры.
        Переопределяется в наследниках.
        """
        raise NotImplementedError

    def fit(
        self,
        X,
        y,
        t,
        eval_set: Optional[Tuple[pd.DataFrame, np.ndarray, np.ndarray]] = None,
    ):
        set_global_seed(self.random_state)

        X = _as_dataframe(X)
        y = _as_numpy_1d(y).astype(np.float32)
        t = _as_numpy_1d(t).astype(np.float32)

        self._fit_mixed_preprocessor(X)
        x_num, x_cat = self._transform_mixed(X)

        val_data = None
        if eval_set is not None:
            X_val, y_val, t_val = eval_set
            X_val = _as_dataframe(X_val)
            y_val = _as_numpy_1d(y_val).astype(np.float32)
            t_val = _as_numpy_1d(t_val).astype(np.float32)
            x_num_val, x_cat_val = self._transform_mixed(X_val)
            val_data = (x_num_val, x_cat_val, y_val, t_val)

        self.model_ = self._build_model().to(self.device)

        dataset = MixedTabularDataset(x_num, x_cat, y=y, t=t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        early_stopping = EarlyStopping(self.patience, self.min_delta)

        for epoch in range(self.n_epochs):
            self.model_.train()
            losses = []

            for batch in loader:
                x_num_b = batch['x_num'].to(self.device)
                x_cat_b = batch['x_cat'].to(self.device)
                y_b = batch['y'].to(self.device)
                t_b = batch['t'].to(self.device)

                p_t, p_c = self.model_(x_num_b, x_cat_b)

                treat_mask = (t_b == 1)
                ctrl_mask = (t_b == 0)

                loss_t = torch.tensor(0.0, device=self.device)
                loss_c = torch.tensor(0.0, device=self.device)

                if treat_mask.any():
                    loss_t = F.binary_cross_entropy(p_t[treat_mask], y_b[treat_mask])
                if ctrl_mask.any():
                    loss_c = F.binary_cross_entropy(p_c[ctrl_mask], y_b[ctrl_mask])

                loss = loss_t + loss_c

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            val_loss = None
            if val_data is not None:
                self.model_.eval()
                with torch.no_grad():
                    x_num_val_t = torch.tensor(val_data[0], dtype=torch.float32, device=self.device)
                    x_cat_val_t = torch.tensor(val_data[1], dtype=torch.long, device=self.device)
                    y_val_t = torch.tensor(val_data[2], dtype=torch.float32, device=self.device)
                    t_val_t = torch.tensor(val_data[3], dtype=torch.float32, device=self.device)

                    p_t_val, p_c_val = self.model_(x_num_val_t, x_cat_val_t)

                    val_treat_mask = (t_val_t == 1)
                    val_ctrl_mask = (t_val_t == 0)

                    loss_t_val = torch.tensor(0.0, device=self.device)
                    loss_c_val = torch.tensor(0.0, device=self.device)

                    if val_treat_mask.any():
                        loss_t_val = F.binary_cross_entropy(p_t_val[val_treat_mask], y_val_t[val_treat_mask])
                    if val_ctrl_mask.any():
                        loss_c_val = F.binary_cross_entropy(p_c_val[val_ctrl_mask], y_val_t[val_ctrl_mask])

                    val_loss = (loss_t_val + loss_c_val).item()

            if self.verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch == self.n_epochs - 1):
                msg = f'[{self.__class__.__name__}] epoch {epoch + 1}/{self.n_epochs} | train_loss={np.mean(losses):.4f}'
                if val_loss is not None:
                    msg += f' | val_loss={val_loss:.4f}'
                print(msg)

            should_stop = early_stopping.step(val_loss, self.model_)
            if should_stop:
                break

        early_stopping.restore(self.model_)
        return self

    def predict_components(self, X) -> Tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, 'model_') or self.model_ is None:
            raise ValueError('Model is not fitted.')
        X = _as_dataframe(X)
        x_num, x_cat = self._transform_mixed(X)

        dataset = MixedTabularDataset(x_num, x_cat)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model_.eval()
        p_ts, p_cs = [], []
        with torch.no_grad():
            for batch in loader:
                x_num_b = batch['x_num'].to(self.device)
                x_cat_b = batch['x_cat'].to(self.device)
                p_t, p_c = self.model_(x_num_b, x_cat_b)
                p_ts.append(p_t.cpu().numpy())
                p_cs.append(p_c.cpu().numpy())

        p_t_all = np.concatenate(p_ts)
        p_c_all = np.concatenate(p_cs)
        return p_t_all, p_c_all

    def predict(self, X) -> np.ndarray:
        p_t, p_c = self.predict_components(X)
        return p_t - p_c


class ClassicTARNetUplift(_BaseTARNetUplift):
    """
    Sklearn-like обертка над классическим TARNet на MLP.
    """
    def __init__(
        self,
        num_cols: Sequence[str],
        cat_cols: Sequence[str],
        shared_hidden: Sequence[int] = (128, 64),
        head_hidden: Sequence[int] = (32, 16),
        cat_emb_dim: int = 8,
        dropout: float = 0.2,
        lr: float = 1e-3,
        n_epochs: int = 40,
        batch_size: int = 512,
        weight_decay: float = 1e-4,
        patience: Optional[int] = 8,
        min_delta: float = 1e-4,
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
    ):
        super().__init__(
            num_cols=num_cols,
            cat_cols=cat_cols,
            lr=lr,
            n_epochs=n_epochs,
            batch_size=batch_size,
            weight_decay=weight_decay,
            patience=patience,
            min_delta=min_delta,
            device=device,
            random_state=random_state,
            verbose=verbose,
        )
        self.shared_hidden = list(shared_hidden)
        self.head_hidden = list(head_hidden)
        self.cat_emb_dim = cat_emb_dim
        self.dropout = dropout

    def _build_model(self) -> nn.Module:
        return ClassicTARNet(
            n_num_features=len(self.num_cols),
            cardinalities=self.cardinalities_,
            cat_emb_dim=self.cat_emb_dim,
            shared_hidden=self.shared_hidden,
            head_hidden=self.head_hidden,
            dropout=self.dropout,
        )


class AttentionTARNetUplift(_BaseTARNetUplift):
    """
    Sklearn-like обертка над TARNet с self-attention.

    Используется как более гибкая альтернатива классическому TARNet,
    когда важно дать модели возможность учить сложные взаимодействия
    между числовыми и категориальными признаками.
    """
    def __init__(
        self,
        num_cols: Sequence[str],
        cat_cols: Sequence[str],
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        mlp_hidden: int = 64,
        dropout: float = 0.2,
        lr: float = 5e-4,
        n_epochs: int = 50,
        batch_size: int = 1024,
        weight_decay: float = 1e-4,
        patience: Optional[int] = 8,
        min_delta: float = 1e-4,
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
    ):
        super().__init__(
            num_cols=num_cols,
            cat_cols=cat_cols,
            lr=lr,
            n_epochs=n_epochs,
            batch_size=batch_size,
            weight_decay=weight_decay,
            patience=patience,
            min_delta=min_delta,
            device=device,
            random_state=random_state,
            verbose=verbose,
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_hidden = mlp_hidden
        self.dropout = dropout

    def _build_model(self) -> nn.Module:
        return AttentionTARNet(
            n_num_features=len(self.num_cols),
            cardinalities=self.cardinalities_,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            mlp_hidden=self.mlp_hidden,
            dropout=self.dropout,
        )


def get_default_dl_model_configs() -> Dict[str, dict]:
    """
    Возвращает словарь с дефолтными конфигами DL-моделей, приведенными к единому формату.

    Что лежит внутри:
    - t_learner_nn
    - s_learner_nn
    - tarnet_classic
    - tarnet_attention
    """
    return {
        't_learner_nn': {
            'hidden_dims': [64, 32],
            'dropout': 0.22338843352395796,
            'lr': 0.00032488253588586755,
            'batch_size': 512,
            'epochs': 60,
        },
        's_learner_nn': {
            'cat_embed_dim': 8,
            'hidden_dims': [64, 32],
            'dropout': 0.20295116715006706,
            'lr': 0.0009865977951798554,
            'batch_size': 512,
            'epochs': 30,
        },
        'tarnet_classic': {
            'shared_hidden': [128, 64],
            'head_hidden': [32, 16],
            'cat_emb_dim': 8,
            'dropout': 0.2,
            'lr': 1e-3,
            'n_epochs': 40,
            'batch_size': 512,
        },
        'tarnet_attention': {
            'd_model': 32,
            'n_heads': 4,
            'n_layers': 2,
            'mlp_hidden': 64,
            'dropout': 0.2,
            'lr': 5e-4,
            'n_epochs': 50,
            'batch_size': 1024,
        },
    }


__all__ = [
    'set_global_seed',
    'BinaryMLPClassifierTorch',
    'TLearnerNNUplift',
    'SLearnerNNUplift',
    'ClassicTARNetUplift',
    'AttentionTARNetUplift',
    'get_default_dl_model_configs',
]

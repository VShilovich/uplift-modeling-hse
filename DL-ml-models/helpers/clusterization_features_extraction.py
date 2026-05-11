import numpy as np
import pandas as pd
import umap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class UmapClusterTransformer(BaseEstimator, TransformerMixin):
    """
    Класс для кластеризации фичей: UMAP -> K-Means (фиксированное число кластеров).
    Генерирует расстояния до центроидов как новые числовые фичи + ID кластера.
    """
    def __init__(self, num_cols, n_clusters=5, n_neighbors=30, min_dist=0.1, random_state=42):
        self.num_cols = num_cols
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.reducer = umap.UMAP(
            n_neighbors=self.n_neighbors, 
            min_dist=self.min_dist, 
            random_state=self.random_state,
            n_jobs=1
        )
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)

    def fit(self, X, y=None):
        # 1. Подготовка данных
        X_num = X[self.num_cols].copy().fillna(0)
        
        # 2. Обучение проекции и кластеризации
        X_scaled = self.scaler.fit_transform(X_num)
        X_umap = self.reducer.fit_transform(X_scaled)
        self.kmeans.fit(X_umap)
        
        return self

    def transform(self, X):
        X_out = X.copy()
        X_num = X[self.num_cols].copy().fillna(0)
        
        # 1. Проекция
        X_scaled = self.scaler.transform(X_num)
        X_umap = self.reducer.transform(X_scaled)
        
        # 2. Расстояния до центроидов (новые числовые фичи)
        centroids = self.kmeans.cluster_centers_
        dist_matrix = cdist(X_umap, centroids, metric='euclidean')
        
        dist_cols = [f"dist_to_centroid_{i}" for i in range(self.n_clusters)]
        for i, col in enumerate(dist_cols):
            X_out[col] = dist_matrix[:, i]
            
        # 3. ID кластера (категориальная фича)
        X_out['cluster_id'] = self.kmeans.predict(X_umap).astype(str)
            
        return X_out

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
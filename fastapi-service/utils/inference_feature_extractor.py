import numpy as np
import pandas as pd

class UpliftFeatureExtractorInference:
    """
    Feature extractor для uplift-моделирования на основе данных X5
    Включает все этапы предобработки и генерации признаков из EDA
    """
    def __init__(self, drop_redundant=True):
        self.feature_names = []
        self.drop_redundant = drop_redundant


    def safe_div(self, a, b):
        """Исправленная версия"""
        result = np.where(b == 0, 0, a / b)
        if hasattr(a, 'index') and hasattr(b, 'index'):
            return pd.Series(result, index=a.index)
        return pd.Series(result)
    

    def preprocess_clients_inference(self, clients_df):
        """Inference: предобработка клиентов"""
        df_clients = clients_df.copy()
        
        # Обработка возраста (тот же код)
        valid_mask = (df_clients['age'] >= 15) & (df_clients['age'] <= 100)
        valid_ages = df_clients.loc[valid_mask, 'age']
        
        q1, q3 = valid_ages.quantile([0.25, 0.75])
        mean_q1 = valid_ages[(valid_ages >= 15) & (valid_ages <= q1)].mean()
        mean_q4 = valid_ages[(valid_ages > q3) & (valid_ages <= 100)].mean()
        
        def replace_age_with_quartile_mean(age):
            if 15 <= age <= 100:
                return age
            if age < 15:
                return mean_q1
            elif 100 < age <= 200:
                return mean_q4
            else:
                return valid_ages.mean()
        
        df_clients['age'] = df_clients['age'].apply(replace_age_with_quartile_mean)
        df_clients['gender'] = df_clients['gender'].astype('category')
        df_clients['is_activated'] = np.where(df_clients['first_redeem_date'].notna(), 1, 0)
        
        return df_clients.set_index('client_id')
    

    def preprocess_purchases(self, purchases_df):
        """Предобработка транзакционных данных"""
        purchases = purchases_df.copy()
        
        # Заполнение пропусков
        purchases['trn_sum_from_red'] = purchases['trn_sum_from_red'].fillna(purchases['trn_sum_from_iss'])
        
        # Преобразование потраченных баллов
        purchases['regular_points_spent'] = purchases['regular_points_spent'].abs()
        purchases['express_points_spent'] = purchases['express_points_spent'].abs()
        
        return purchases
    

    def generate_behavioral_features(self, purchases_df):
        """Генерация поведенческих признаков"""
        df = purchases_df.copy()
        
        # Уникальные транзакции
        trans_cols = ['client_id', 'transaction_id', 'transaction_datetime', 
                     'regular_points_received', 'express_points_received',
                     'regular_points_spent', 'express_points_spent', 
                     'purchase_sum', 'store_id']
        unique_trans = df[trans_cols].drop_duplicates('transaction_id')
        
        # Продуктовые данные
        product_cols = ['client_id', 'transaction_id', 'product_id', 
                       'product_quantity', 'trn_sum_from_iss', 'trn_sum_from_red']
        product_data = df[product_cols]
        
        features = {}
        
        # Базовые транзакционные фичи
        client_trans = unique_trans.groupby('client_id')
        features['total_transactions'] = client_trans.size()
        features['total_purchase_sum'] = client_trans['purchase_sum'].sum()
        features['avg_transaction_amount'] = client_trans['purchase_sum'].mean()
        features['std_transaction_amount'] = client_trans['purchase_sum'].std()
        features['max_transaction_amount'] = client_trans['purchase_sum'].max()
        features['min_transaction_amount'] = client_trans['purchase_sum'].min()
        
        # Квантили транзакций
        for q in [0.25, 0.5, 0.75]:
            features[f'transaction_amount_q{q}'] = client_trans['purchase_sum'].quantile(q)
        
        # Фичи по баллам
        features['total_regular_points_received'] = client_trans['regular_points_received'].sum()
        features['total_express_points_received'] = client_trans['express_points_received'].sum()
        features['total_regular_points_spent'] = client_trans['regular_points_spent'].sum()
        features['total_express_points_spent'] = client_trans['express_points_spent'].sum()
        features['avg_regular_points_per_transaction'] = client_trans['regular_points_received'].mean()
        features['avg_express_points_per_transaction'] = client_trans['express_points_received'].mean()
        
        # Отношение заработанных к потраченным баллам
        features['points_earned_to_spent_ratio'] = (
            (client_trans['regular_points_received'].sum() + 
             client_trans['express_points_received'].sum()) / 
            (client_trans['regular_points_spent'].sum() + 
             client_trans['express_points_spent'].sum() + 1)
        )
        
        # Продуктовые фичи
        client_products = product_data.groupby('client_id')
        features['total_products_purchased'] = client_products['product_quantity'].sum()
        features['unique_products_count'] = client_products['product_id'].nunique()
        features['total_trn_sum_from_iss'] = client_products['trn_sum_from_iss'].sum()
        features['total_trn_sum_from_red'] = client_products['trn_sum_from_red'].sum()
        features['avg_product_quantity'] = client_products['product_quantity'].mean()
        
        # Временные фичи
        unique_trans['transaction_datetime'] = pd.to_datetime(unique_trans['transaction_datetime'])
        client_trans_time = unique_trans.groupby('client_id')
        
        first_date = client_trans_time['transaction_datetime'].min()
        last_date = client_trans_time['transaction_datetime'].max()
        
        features['first_transaction_date'] = first_date
        features['last_transaction_date'] = last_date
        features['transaction_period_days'] = (last_date - first_date).dt.days
        
        # Квартал первой транзакции
        first_date_df = first_date.to_frame(name='first_date_tmp')
        first_date_df['first_transaction_quarter'] = (
            first_date_df['first_date_tmp'].dt.year.astype(str) + 
            'Q' + (((first_date_df['first_date_tmp'].dt.month - 1) // 3) + 1).astype(str)
        )
        first_date_df['first_transaction_year_quarter_idx'] = (
            first_date_df['first_date_tmp'].dt.year * 4 + 
            ((first_date_df['first_date_tmp'].dt.month - 1) // 3 + 1)
        )
        
        features['first_transaction_quarter'] = first_date_df['first_transaction_quarter']
        features['first_transaction_year_quarter_idx'] = first_date_df['first_transaction_year_quarter_idx']
        
        # День недели и время суток
        unique_trans['transaction_weekday'] = unique_trans['transaction_datetime'].dt.dayofweek
        unique_trans['transaction_hour'] = unique_trans['transaction_datetime'].dt.hour
        
        features['most_frequent_weekday'] = unique_trans.groupby('client_id')['transaction_weekday'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else -1
        )
        features['most_frequent_hour'] = unique_trans.groupby('client_id')['transaction_hour'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else -1
        )
        
        # Частота транзакций
        features['transactions_per_day'] = (
            client_trans.size() / (features['transaction_period_days'] + 1)
        )
        
        # Фичи по магазинам
        features['unique_stores_visited'] = client_trans['store_id'].nunique()
        features['most_frequent_store'] = client_trans['store_id'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else -1
        )
        features['store_loyalty_ratio'] = client_trans['store_id'].agg(
            lambda x: x.value_counts().iloc[0] / len(x) if len(x) > 0 else 0
        )
        
        return pd.DataFrame(features)
    

    def generate_static_features(self, clients_df):
        """Генерация статических признаков"""
        df = clients_df.copy()
        features = pd.DataFrame(index=df.index)
        
        first_issue = pd.to_datetime(df['first_issue_date'], errors='coerce')
        first_redeem = pd.to_datetime(df['first_redeem_date'], errors='coerce')
        
        features['first_issue_month'] = first_issue.dt.month
        features['first_issue_weekday'] = first_issue.dt.dayofweek
        features['first_issue_quarter'] = (first_issue.dt.year.astype(str) + 
                                         'Q' + (((first_issue.dt.month - 1) // 3) + 1).astype(str))
        features['first_issue_year_quarter_idx'] = (first_issue.dt.year * 4 + 
                                                   ((first_issue.dt.month - 1) // 3 + 1))
        
        # Лаг между выпуском и первым использованием
        features['redeem_lag_days'] = (first_redeem - first_issue).dt.days
        
        return features
    

    def create_business_features(self, behavioral_df, static_df):
        """Создание бизнес-признаков на основе EDA"""
        df = static_df.join(behavioral_df)
        
        # Основные бизнес-метрики
        df['avg_purchase_per_day'] = self.safe_div(
            df['total_purchase_sum'], 
            df['transaction_period_days'].clip(lower=1)
        )
        
        df['spend_per_transaction'] = self.safe_div(
            df['total_purchase_sum'],
            df['total_transactions'].clip(lower=1)
        )
        
        df['transactions_per_month'] = self.safe_div(
            df['total_transactions'],
            (df['transaction_period_days'].clip(lower=1) / 30)
        )
        
        df['points_earn_ratio'] = self.safe_div(
            df['total_regular_points_received'] + df['total_express_points_received'],
            df['total_transactions'].clip(lower=1)
        )
        
        df['points_spend_ratio'] = self.safe_div(
            df['total_regular_points_spent'] + df['total_express_points_spent'],
            df['total_transactions'].clip(lower=1)
        )
        
        df['points_balance_ratio'] = self.safe_div(
            (df['total_regular_points_received'] + df['total_express_points_received']),
            (df['total_regular_points_spent'] + df['total_express_points_spent'] + 1)
        )
        
        df['avg_points_per_purchase'] = self.safe_div(
            (df['total_regular_points_received'] + df['total_express_points_received']),
            df['total_transactions'].clip(lower=1)
        )
        
        df['loyal_store_flag'] = np.where(df['store_loyalty_ratio'] >= 0.9, 1, 0)
        
        df['unique_store_intensity'] = self.safe_div(
            df['unique_stores_visited'],
            df['total_transactions'].clip(lower=1)
        )
        
        df['activity_density'] = self.safe_div(
            df['total_transactions'],
            df['transaction_period_days'].clip(lower=1)
        )
        
        df['log_total_purchase_sum'] = np.log1p(df['total_purchase_sum'])
        
        # Сезонность
        def quarter_to_season(q_idx):
            if pd.isna(q_idx):
                return 0
            q = int(q_idx) % 4
            return {1: 1, 2: 2, 3: 3, 0: 4}[q]
        
        df['seasonal_quarter_code'] = df['first_transaction_year_quarter_idx'].apply(quarter_to_season)
        
        # Дополнительные бизнес-метрики из EDA
        df['avg_items_per_transaction'] = self.safe_div(
            df['total_products_purchased'],
            df['total_transactions'].clip(lower=1)
        )
        
        df['spend_points_per_transaction'] = self.safe_div(
            df['total_regular_points_spent'],
            df['total_transactions'].clip(lower=1)
        )
        
        df['transaction_value_density'] = self.safe_div(
            df['log_total_purchase_sum'],
            df['transaction_period_days'].clip(lower=1)
        )
        
        df['is_super_loyal'] = np.where(df['store_loyalty_ratio'] >= 0.9, 1, 0)
        
        return df
    

    def remove_redundant_features(self, df):
        """Удаление избыточных признаков на основе корреляционного анализа"""
        cols_to_drop = [
            'first_issue_date', 'first_redeem_date', 'first_transaction_date',
            'last_transaction_date', 'redeem_lag_days', 'std_transaction_amount',
            'most_frequent_hour', 'most_frequent_weekday', 'most_frequent_store',
            'activity_density', 'transactions_per_day', 'spend_per_transaction',
            'transaction_amount_q0.25', 'transaction_amount_q0.5', 'transaction_amount_q0.75',
            'total_purchase_sum', 'total_trn_sum_from_red', 'total_trn_sum_from_iss',
            'avg_regular_points_per_transaction', 'points_earn_ratio', 
            'avg_points_per_purchase', 'total_regular_points_received',
            'total_regular_points_spent', 'total_products_purchased',
            'loyal_store_flag', 'first_issue_quarter'
        ]
        
        return df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    

    def calculate_features(self, clients_df, purchases_df):
        """
        INFERENCE: только clients_df + purchases_df
        """
        # Предобработка
        processed_clients = self.preprocess_clients_inference(clients_df)
        processed_purchases = self.preprocess_purchases(purchases_df)

        # Генерация признаков
        behavioral_features = self.generate_behavioral_features(processed_purchases)
        static_features = self.generate_static_features(processed_clients)
        
        # Объединение и бизнес-признаки
        final_df = self.create_business_features(behavioral_features, static_features)

        # Демография
        demo_features = processed_clients[['age', 'gender', 'is_activated']]
        final_df = final_df.join(demo_features)
        
        # Удаление избыточных
        if self.drop_redundant:
            final_df = self.remove_redundant_features(final_df)
        
        # Обработка NaN/inf
        for col in final_df.select_dtypes(include=[np.number]).columns:
            s = final_df[col]
            s = s.fillna(0.0)  # inference: заполняем 0
            s = s.replace([np.inf, -np.inf], 1e9)
            s = np.clip(s, -1e9, 1e9)
            final_df[col] = s.round(6)
        
        self.feature_names = final_df.columns.tolist()
        return final_df
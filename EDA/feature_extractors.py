import pandas as pd

class BehavioralFeatureGenerator:
    """
    Класс-шаблон для генерации агрегированных поведенческих признаков.
    На основе транзакционных данных (таблица purchases).
    """
    def __init__(self, purchases_df):
        self.df = purchases_df.copy()

        cols_to_keep = ['client_id', 'transaction_id', 'transaction_datetime', 
                'regular_points_received', 'express_points_received',
                'regular_points_spent', 'express_points_spent', 
                'purchase_sum', 'store_id']
        self.df_unique_transactions = self.df[cols_to_keep].drop_duplicates('transaction_id')

        product_cols = ['client_id', 'transaction_id', 'product_id', 
                       'product_quantity', 'trn_sum_from_iss', 'trn_sum_from_red']
        self.df_product_data = self.df[product_cols]

    # Уникальные транзакции (без дублирования по product_id)
    def _get_unique_transactions(self):
        return self.df_unique_transactions

    # Данные на уровне продуктов
    def _get_product_level_data(self):
        return self.df_product_data

    # Генерация всех фич
    def generate_features(self):
        features = {}
        
        # Базовые фичи по транзакциям
        features.update(self._transaction_features())
        
        # Фичи по баллам
        features.update(self._points_features())
        
        # Фичи по продуктам
        features.update(self._product_features())
        
        # Временные фичи
        features.update(self._time_features())
        
        # Фичи по магазинам
        features.update(self._store_features())
        
        return pd.DataFrame(features)

    # Фичи по транзакциям
    def _transaction_features(self):
        trans_df = self._get_unique_transactions()
        client_trans = trans_df.groupby('client_id')
        features = {
            'total_transactions': client_trans.size(),
            'total_purchase_sum': client_trans['purchase_sum'].sum(),
            'avg_transaction_amount': client_trans['purchase_sum'].mean(),
            'std_transaction_amount': client_trans['purchase_sum'].std(),
            'max_transaction_amount': client_trans['purchase_sum'].max(),
            'min_transaction_amount': client_trans['purchase_sum'].min(),
        }
        
        # Квантили
        for q in [0.25, 0.5, 0.75]:
            features[f'transaction_amount_q{q}'] = client_trans['purchase_sum'].quantile(q)
        
        return features

    # Фичи по баллам
    def _points_features(self):
        trans_df = self._get_unique_transactions()
        client_trans = trans_df.groupby('client_id')
        features = {
            'total_regular_points_received': client_trans['regular_points_received'].sum(),
            'total_express_points_received': client_trans['express_points_received'].sum(),
            'total_regular_points_spent': client_trans['regular_points_spent'].sum(),
            'total_express_points_spent': client_trans['express_points_spent'].sum(),
            'avg_regular_points_per_transaction': client_trans['regular_points_received'].mean(),
            'avg_express_points_per_transaction': client_trans['express_points_received'].mean(),
            'points_earned_to_spent_ratio': (client_trans['regular_points_received'].sum() + 
                                           client_trans['express_points_received'].sum()) / 
                                          (client_trans['regular_points_spent'].sum() + 
                                           client_trans['express_points_spent'].sum() + 1)  # +1 чтобы избежать деления на 0
        }
        
        return features

    # Фичи по продуктам
    def _product_features(self):
        product_df = self._get_product_level_data()
        client_products = product_df.groupby('client_id')
        features = {
            'total_products_purchased': client_products['product_quantity'].sum(),
            'unique_products_count': client_products['product_id'].nunique(),
            'total_trn_sum_from_iss': client_products['trn_sum_from_iss'].sum(),
            'total_trn_sum_from_red': client_products['trn_sum_from_red'].sum(),
            'avg_product_quantity': client_products['product_quantity'].mean(),
        }
        
        return features

    # Временные фичи
    def _time_features(self):
        trans_df = self._get_unique_transactions()
        trans_df['transaction_datetime'] = pd.to_datetime(trans_df['transaction_datetime'])
        client_trans = trans_df.groupby('client_id')
        first_date=client_trans['transaction_datetime'].min()
        last_date=client_trans['transaction_datetime'].max()
        features = {
            'first_transaction_date': first_date,
            'last_transaction_date': last_date,
            'transaction_period_days': (last_date - 
                                      first_date).dt.days,
        }
        
        features['transactions_per_day'] = (client_trans.size() / (features['transaction_period_days'] + 1))
        first_date_df = first_date.to_frame(name='first_date_tmp')

        # Квартал строим в формате год+Q+месяц, чтобы учесть "старость" клиентов
        first_date_df['first_transaction_quarter'] = (first_date_df['first_date_tmp'].dt.year.astype(str) 
                                                      + 'Q' + (((first_date_df['first_date_tmp'].dt.month - 1) // 3) + 1).astype(str))

        first_date_df['first_transaction_year_quarter_idx'] = (
             first_date_df['first_date_tmp'].dt.year * 4
             + ((first_date_df['first_date_tmp'].dt.month - 1) // 3 + 1)
             )

        features['first_transaction_quarter'] = first_date_df['first_transaction_quarter']
        features['first_transaction_year_quarter_idx'] = first_date_df['first_transaction_year_quarter_idx']
        
        # День недели и время суток
        trans_df['transaction_weekday'] = trans_df['transaction_datetime'].dt.dayofweek
        trans_df['transaction_hour'] = trans_df['transaction_datetime'].dt.hour
        
        # Самый частый день недели и час
        features['most_frequent_weekday'] = trans_df.groupby('client_id')['transaction_weekday'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else -1
        )
        features['most_frequent_hour'] = trans_df.groupby('client_id')['transaction_hour'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else -1
        )
        
        return features

    # Фичи по магазинам
    def _store_features(self):
        trans_df = self._get_unique_transactions()
        client_trans = trans_df.groupby('client_id')
        features = {
            'unique_stores_visited': client_trans['store_id'].nunique(),
            'most_frequent_store': client_trans['store_id'].agg(
                lambda x: x.mode()[0] if len(x.mode()) > 0 else -1
            ),
            'store_loyalty_ratio': client_trans['store_id'].agg(
                lambda x: x.value_counts().iloc[0] / len(x) if len(x) > 0 else 0
            )
        }
        
        return features

class StaticFeatureGenerator:
    """
    Шаблон класса для извлечения и генерации статических признаков.
    На основе данных о клиентах (таблица df_clients).
    """
    def __init__(self, clients_info_df):
        self.df = clients_info_df.copy()

    def generate_features(self):
        df_indexed = self.df.set_index('client_id')
        features = pd.DataFrame(index=df_indexed.index)

        first_issue = pd.to_datetime(df_indexed['first_issue_date'], errors='coerce')
        first_redeem = pd.to_datetime(df_indexed['first_redeem_date'], errors='coerce')

        # базовые календарные признаки
        features['first_issue_month'] = first_issue.dt.month
        features['first_issue_weekday'] = first_issue.dt.dayofweek

        # квартал и когорта (опять же, чтобы учесть "старость" клиентов, можно было давить и ранее, но выделим отдельно
        # так как это не транзакционная/поведенческая характеристика)
        features['first_issue_quarter'] = (first_issue.dt.year.astype(str)
            + 'Q'+ (((first_issue.dt.month - 1) // 3) + 1).astype(str))

        features['first_issue_year_quarter_idx'] = (
            first_issue.dt.year * 4
            + ((first_issue.dt.month - 1) // 3 + 1)
        )

        # лаг между выпуском карты и первым использованием
        features['redeem_lag_days'] = (first_redeem - first_issue).dt.days

        return features
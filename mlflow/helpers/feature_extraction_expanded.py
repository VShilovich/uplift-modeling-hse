import gc
import numpy as np
import pandas as pd


class UpliftFeatureExtractorExpanded:
    """
    Memory-safe expanded feature extractor для uplift-моделирования.
    """

    def __init__(self, drop_redundant=True):
        self.feature_names = []
        self.drop_redundant = drop_redundant

    # =========================
    # utils
    # =========================
    def safe_div(self, a, b):
        if hasattr(a, 'index') and hasattr(b, 'index'):
            result = np.where(b == 0, 0, a / b)
            return pd.Series(result, index=a.index)
        return np.where(b == 0, 0, a / b)

    def _mode_or_default(self, s, default=-1):
        m = s.mode()
        return m.iloc[0] if len(m) > 0 else default

    def _top_share(self, s):
        if len(s) == 0:
            return 0.0
        return s.value_counts(normalize=True).iloc[0]

    def _downcast_numeric(self, df):
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            s = pd.to_numeric(df[col], errors='coerce')
            s = s.replace([np.inf, -np.inf], np.nan)
            s = s.clip(-1e9, 1e9)

            if pd.api.types.is_float_dtype(s):
                df[col] = pd.to_numeric(s, downcast='float')
            else:
                df[col] = pd.to_numeric(s, downcast='integer')
        return df

    # clients
    def preprocess_clients(self, clients_df, train_df, treatment_df, target_df):
        # concat без лишних копий
        df_clients = pd.concat(
            [train_df, treatment_df, target_df],
            axis=1
        )
        df_clients = df_clients.merge(clients_df, on='client_id', how='left')

        # age cleaning — векторно
        valid_mask = df_clients['age'].between(15, 100)
        valid_ages = df_clients.loc[valid_mask, 'age']

        if len(valid_ages) == 0:
            global_age_mean = 40.0
            mean_q1 = 30.0
            mean_q4 = 55.0
        else:
            q1, q3 = valid_ages.quantile([0.25, 0.75])
            mean_q1 = valid_ages[(valid_ages >= 15) & (valid_ages <= q1)].mean()
            mean_q4 = valid_ages[(valid_ages > q3) & (valid_ages <= 100)].mean()
            global_age_mean = valid_ages.mean()

            if pd.isna(mean_q1):
                mean_q1 = global_age_mean
            if pd.isna(mean_q4):
                mean_q4 = global_age_mean

        age = df_clients['age'].copy()

        age = np.where((age >= 15) & (age <= 100), age, age)
        age = np.where(age < 15, mean_q1, age)
        age = np.where((age > 100) & (age <= 200), mean_q4, age)
        age = np.where(pd.isna(age), global_age_mean, age)

        df_clients['age'] = pd.to_numeric(age, downcast='float')
        df_clients['gender'] = df_clients['gender'].astype('category')
        df_clients['is_activated'] = np.where(df_clients['first_redeem_date'].notna(), 1, 0).astype('int8')

        return df_clients.set_index('client_id')

    # purchases
    def preprocess_purchases(self, purchases_df):
        # берем только нужные колонки
        needed_cols = [
            'client_id',
            'transaction_id',
            'transaction_datetime',
            'regular_points_received',
            'express_points_received',
            'regular_points_spent',
            'express_points_spent',
            'purchase_sum',
            'store_id',
            'product_id',
            'product_quantity',
            'trn_sum_from_iss',
            'trn_sum_from_red',
        ]

        purchases = purchases_df.loc[:, needed_cols].copy()

        purchases['trn_sum_from_red'] = purchases['trn_sum_from_red'].fillna(purchases['trn_sum_from_iss'])
        purchases['regular_points_spent'] = purchases['regular_points_spent'].abs()
        purchases['express_points_spent'] = purchases['express_points_spent'].abs()

        # downcast
        for col in [
            'regular_points_received',
            'express_points_received',
            'regular_points_spent',
            'express_points_spent',
            'purchase_sum',
            'product_quantity',
            'trn_sum_from_iss',
            'trn_sum_from_red',
        ]:
            purchases[col] = pd.to_numeric(purchases[col], downcast='float')

        return purchases

    # behavioral features
    def generate_behavioral_features(self, purchases_df):
        trans_cols = [
            'client_id',
            'transaction_id',
            'transaction_datetime',
            'regular_points_received',
            'express_points_received',
            'regular_points_spent',
            'express_points_spent',
            'purchase_sum',
            'store_id',
        ]
        product_cols = [
            'client_id',
            'transaction_id',
            'product_id',
            'product_quantity',
            'trn_sum_from_iss',
            'trn_sum_from_red',
        ]

        # транзакционный срез
        unique_trans = purchases_df.loc[:, trans_cols].drop_duplicates('transaction_id').copy()
        unique_trans['transaction_datetime'] = pd.to_datetime(
            unique_trans['transaction_datetime'],
            errors='coerce'
        )
        unique_trans['transaction_weekday'] = unique_trans['transaction_datetime'].dt.dayofweek.astype('float32')
        unique_trans['transaction_hour'] = unique_trans['transaction_datetime'].dt.hour.astype('float32')

        grp_t = unique_trans.groupby('client_id', sort=False)

        trans_basic = grp_t.agg(
            total_transactions=('transaction_id', 'size'),
            total_purchase_sum=('purchase_sum', 'sum'),
            avg_transaction_amount=('purchase_sum', 'mean'),
            std_transaction_amount=('purchase_sum', 'std'),
            max_transaction_amount=('purchase_sum', 'max'),
            min_transaction_amount=('purchase_sum', 'min'),
            total_regular_points_received=('regular_points_received', 'sum'),
            total_express_points_received=('express_points_received', 'sum'),
            total_regular_points_spent=('regular_points_spent', 'sum'),
            total_express_points_spent=('express_points_spent', 'sum'),
            avg_regular_points_per_transaction=('regular_points_received', 'mean'),
            avg_express_points_per_transaction=('express_points_received', 'mean'),
            first_transaction_date=('transaction_datetime', 'min'),
            last_transaction_date=('transaction_datetime', 'max'),
            unique_stores_visited=('store_id', 'nunique'),
        )

        # квантили
        purchase_q = (
            grp_t['purchase_sum']
            .quantile([0.25, 0.5, 0.75])
            .unstack()
            .rename(columns={
                0.25: 'transaction_amount_q0.25',
                0.5: 'transaction_amount_q0.5',
                0.75: 'transaction_amount_q0.75'
            })
        )

        # mode features
        most_frequent_weekday = grp_t['transaction_weekday'].agg(lambda s: self._mode_or_default(s, -1))
        most_frequent_hour = grp_t['transaction_hour'].agg(lambda s: self._mode_or_default(s, -1))
        most_frequent_store = grp_t['store_id'].agg(lambda s: self._mode_or_default(s, -1))

        # ratios / spreads
        store_loyalty_ratio = grp_t['store_id'].agg(self._top_share)
        weekend_purchase_ratio = grp_t['transaction_weekday'].agg(lambda s: s.isin([5, 6]).mean() if len(s) > 0 else 0.0)
        evening_purchase_ratio = grp_t['transaction_hour'].agg(lambda s: s.between(18, 23).mean() if len(s) > 0 else 0.0)
        purchase_time_variance = grp_t['transaction_hour'].std().fillna(0)

        # first transaction quarter
        first_date = trans_basic['first_transaction_date']
        first_transaction_year_quarter_idx = (
            first_date.dt.year * 4 +
            ((first_date.dt.month - 1) // 3 + 1)
        )
        first_transaction_quarter = (
            first_date.dt.year.astype('Int64').astype(str) +
            'Q' +
            (((first_date.dt.month - 1) // 3) + 1).astype('Int64').astype(str)
        ).astype('category')

        # продуктовый срез
        product_data = purchases_df.loc[:, product_cols]

        grp_p = product_data.groupby('client_id', sort=False)
        prod_basic = grp_p.agg(
            total_products_purchased=('product_quantity', 'sum'),
            unique_products_count=('product_id', 'nunique'),
            total_trn_sum_from_iss=('trn_sum_from_iss', 'sum'),
            total_trn_sum_from_red=('trn_sum_from_red', 'sum'),
            avg_product_quantity=('product_quantity', 'mean'),
        )

        # join всех behavioral
        features = trans_basic.join(purchase_q, how='left')
        features['most_frequent_weekday'] = most_frequent_weekday
        features['most_frequent_hour'] = most_frequent_hour
        features['most_frequent_store'] = most_frequent_store
        features['store_loyalty_ratio'] = store_loyalty_ratio
        features['store_concentration'] = store_loyalty_ratio
        features['weekend_purchase_ratio'] = weekend_purchase_ratio
        features['evening_purchase_ratio'] = evening_purchase_ratio
        features['purchase_time_variance'] = purchase_time_variance
        features['first_transaction_quarter'] = first_transaction_quarter
        features['first_transaction_year_quarter_idx'] = first_transaction_year_quarter_idx

        features = features.join(prod_basic, how='left')

        # derived behavioral
        features['transaction_period_days'] = (
            features['last_transaction_date'] - features['first_transaction_date']
        ).dt.days.fillna(0)

        features['transactions_per_day'] = self.safe_div(
            features['total_transactions'],
            features['transaction_period_days'] + 1
        )

        features['points_earned_to_spent_ratio'] = self.safe_div(
            features['total_regular_points_received'] + features['total_express_points_received'],
            features['total_regular_points_spent'] + features['total_express_points_spent'] + 1
        )

        features['product_diversity'] = self.safe_div(
            features['unique_products_count'],
            features['total_products_purchased'] + 1
        )

        features['avg_price_per_product'] = self.safe_div(
            features['total_trn_sum_from_iss'],
            features['total_products_purchased'] + 1
        )

        divisor = (features['total_transactions'] - 1).replace(0, 1)
        features['avg_days_between_purchases'] = self.safe_div(
            features['transaction_period_days'],
            divisor
        )

        features['express_points_ratio'] = self.safe_div(
            features['total_express_points_received'],
            features['total_regular_points_received'] + features['total_express_points_received'] + 1
        )

        features['points_redemption_rate'] = self.safe_div(
            features['total_regular_points_spent'],
            features['total_regular_points_received'] + 1
        )

        total_points = features['total_regular_points_received'] + features['total_express_points_received']
        features['points_efficiency'] = self.safe_div(
            total_points,
            features['total_purchase_sum'] + 1
        )

        # memory cleanup
        del unique_trans, product_data, trans_basic, prod_basic, purchase_q
        del most_frequent_weekday, most_frequent_hour, most_frequent_store
        del store_loyalty_ratio, weekend_purchase_ratio, evening_purchase_ratio, purchase_time_variance
        gc.collect()

        return self._downcast_numeric(features)

    # static features
    def generate_static_features(self, clients_df):
        first_issue = pd.to_datetime(clients_df['first_issue_date'], errors='coerce')
        first_redeem = pd.to_datetime(clients_df['first_redeem_date'], errors='coerce')

        features = pd.DataFrame(index=clients_df.index)
        features['first_issue_month'] = pd.to_numeric(first_issue.dt.month, downcast='integer')
        features['first_issue_weekday'] = pd.to_numeric(first_issue.dt.dayofweek, downcast='integer')
        features['first_issue_quarter'] = (
            first_issue.dt.year.astype('Int64').astype(str) +
            'Q' +
            (((first_issue.dt.month - 1) // 3) + 1).astype('Int64').astype(str)
        ).astype('category')
        features['first_issue_year_quarter_idx'] = pd.to_numeric(
            first_issue.dt.year * 4 + ((first_issue.dt.month - 1) // 3 + 1),
            downcast='integer'
        )
        features['redeem_lag_days'] = pd.to_numeric(
            (first_redeem - first_issue).dt.days,
            downcast='float'
        )

        return features

    # business features
    def create_business_features(self, behavioral_df, static_df):
        df = static_df.join(behavioral_df, how='left')

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
            df['transaction_period_days'].clip(lower=1) / 30
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
            df['total_regular_points_received'] + df['total_express_points_received'],
            df['total_regular_points_spent'] + df['total_express_points_spent'] + 1
        )

        df['avg_points_per_purchase'] = self.safe_div(
            df['total_regular_points_received'] + df['total_express_points_received'],
            df['total_transactions'].clip(lower=1)
        )

        df['loyal_store_flag'] = (df['store_loyalty_ratio'] >= 0.9).astype('int8')

        df['unique_store_intensity'] = self.safe_div(
            df['unique_stores_visited'],
            df['total_transactions'].clip(lower=1)
        )

        df['activity_density'] = self.safe_div(
            df['total_transactions'],
            df['transaction_period_days'].clip(lower=1)
        )

        df['log_total_purchase_sum'] = np.log1p(df['total_purchase_sum'])

        def quarter_to_season(q_idx):
            if pd.isna(q_idx):
                return 0
            q = int(q_idx) % 4
            return {1: 1, 2: 2, 3: 3, 0: 4}[q]

        df['seasonal_quarter_code'] = pd.to_numeric(
            df['first_transaction_year_quarter_idx'].apply(quarter_to_season),
            downcast='integer'
        )

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

        df['is_super_loyal'] = (df['store_loyalty_ratio'] >= 0.9).astype('int8')

        df['purchase_frequency_tier'] = pd.cut(
            df['total_transactions'],
            bins=[0, 3, 10, float('inf')],
            labels=['low', 'medium', 'high']
        ).astype('category')

        df['store_loyalty_tier'] = pd.cut(
            df['store_loyalty_ratio'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['low', 'medium', 'high']
        ).astype('category')

        if 'is_activated' in df.columns and 'store_loyalty_ratio' in df.columns:
            df['is_activated_and_loyal'] = (
                (df['is_activated'] == 1) & (df['store_loyalty_ratio'] >= 0.7)
            ).astype('int8')

        return self._downcast_numeric(df)

    # redundant drop
    def remove_redundant_features(self, df):
        cols_to_drop = [
            'first_issue_date',
            'first_redeem_date',
            'first_transaction_date',
            'last_transaction_date',
            'redeem_lag_days',
            'std_transaction_amount',
            'most_frequent_hour',
            'most_frequent_weekday',
            'most_frequent_store',
            'activity_density',
            'transactions_per_day',
            'spend_per_transaction',
            'transaction_amount_q0.25',
            'transaction_amount_q0.5',
            'transaction_amount_q0.75',
            'total_purchase_sum',
            'total_trn_sum_from_red',
            'total_trn_sum_from_iss',
            'avg_regular_points_per_transaction',
            'points_earn_ratio',
            'avg_points_per_purchase',
            'total_regular_points_received',
            'total_regular_points_spent',
            'total_products_purchased',
            'loyal_store_flag',
            'first_issue_quarter',
        ]
        return df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # main функция
    def calculate_features(self, clients_df, train_df, treatment_df, target_df, purchases_df):
        processed_clients = self.preprocess_clients(
            clients_df=clients_df,
            train_df=train_df,
            treatment_df=treatment_df,
            target_df=target_df
        )

        processed_purchases = self.preprocess_purchases(purchases_df)
        behavioral_features = self.generate_behavioral_features(processed_purchases)

        del processed_purchases
        gc.collect()

        static_features = self.generate_static_features(processed_clients)

        final_df = self.create_business_features(
            behavioral_df=behavioral_features,
            static_df=static_features
        )

        del behavioral_features, static_features
        gc.collect()

        # демографические фичи
        demo_features = processed_clients[['age', 'gender', 'is_activated']]
        final_df = final_df.join(demo_features, how='left')

        final_df['age_group'] = pd.cut(
            final_df['age'],
            bins=[0, 25, 45, 65, 200],
            labels=['young', 'middle', 'senior', 'elderly']
        ).astype('category')

        final_df['treatment_flg'] = processed_clients['treatment_flg'].astype('int8')
        final_df['target'] = processed_clients['target'].astype('int8')

        if self.drop_redundant:
            final_df = self.remove_redundant_features(final_df)

        final_df = self._downcast_numeric(final_df)

        self.feature_names = [
            col for col in final_df.columns
            if col not in ['treatment_flg', 'target']
        ]

        return final_df
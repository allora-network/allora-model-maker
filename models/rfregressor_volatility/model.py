import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from models.base_model import Model
from models.rfregressor_volatility.configs import RfregressorConfig

class RfregressorVolatilityModel(Model):
    """Random Forest Regressor model for volatility prediction."""

    def __init__(self, model_name="rfregressor_volatility", config=RfregressorConfig(), debug=False):
        super().__init__(model_name=model_name, debug=debug)
        self.config = config
        self.model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs
        )
        self.candle_interval = config.candle_interval
        self.volatility_window = config.volatility_window


    def _calculate_non_overlapping_volatility(self, series: pd.Series, window_points: int) -> pd.Series:
        """Calculate volatility using non-overlapping windows"""
        log_returns = np.log(series / series.shift(1))
        volatility = pd.Series(index=series.index, dtype=float)

        for start_idx in range(0, len(series), window_points):
            end_idx = start_idx + window_points
            if end_idx > len(series):
                break
            window_data = log_returns.iloc[start_idx:end_idx]
            vol = window_data.std()
            volatility.iloc[start_idx:end_idx] = vol

        return volatility

    def _calculate_volatility_features(self, df: pd.DataFrame):
        """Calculate volatility features with non-overlapping windows"""
        df = df.copy()
        df = df.fillna(method='ffill').fillna(method='bfill')

        window_points = int(pd.Timedelta(self.volatility_window) / pd.Timedelta(self.candle_interval))
        list_of_features = []

        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        for i in range(1, 5):
            log_return_lag_name = f'log_return_t-{i}'
            list_of_features.append(log_return_lag_name)
            df[log_return_lag_name] = df['log_returns'].shift(i)
            volume_lag_name = f'volume_t-{i}'
            list_of_features.append(volume_lag_name)
            df[volume_lag_name] = df['volume'].shift(i)

        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
        list_of_features.extend(['SMA_10', 'EMA_10'])

        df['current_volatility'] = self._calculate_non_overlapping_volatility(df['close'], window_points)
        list_of_features.append('current_volatility')

        df['target_volatility'] = df['current_volatility'].shift(-5)
        print(f"df.tail(): {df.tail()}")
        df_sampled = df.iloc[::window_points].copy()

        # df_sampled = df.copy()
        print(f"df_sampled.tail(): {df_sampled.tail()}")
        return df_sampled, list_of_features

    def train(self, data: pd.DataFrame):
        """Train the volatility prediction model."""
        df_sampled, list_of_features = self._calculate_volatility_features(data)
        df_sampled = df_sampled.dropna()
        print(f"df_sampled.tail() after dropna: {df_sampled.tail()}")
        list_of_features.append('close')
        features = df_sampled[list_of_features]
        target = df_sampled['target_volatility']

        self.model.fit(features, target)
        self.save()

    def inference(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using the trained model."""
        df = input_data.copy()
        df = df.fillna(method='ffill').fillna(method='bfill')

        window_points = int(pd.Timedelta(self.volatility_window) / pd.Timedelta(self.candle_interval))
        list_of_features = []

        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        for i in range(1, 5):
            log_return_lag_name = f'log_return_t-{i}'
            list_of_features.append(log_return_lag_name)
            df[log_return_lag_name] = df['log_returns'].shift(i)
            volume_lag_name = f'volume_t-{i}'
            list_of_features.append(volume_lag_name)
            df[volume_lag_name] = df['volume'].shift(i)

        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
        list_of_features.extend(['SMA_10', 'EMA_10'])

        df['current_volatility'] = self._calculate_non_overlapping_volatility(df['close'], window_points)
        list_of_features.append('current_volatility')
        list_of_features.append('close')

        features = df[list_of_features].fillna(0)
        predictions = self.model.predict(features)

        return pd.DataFrame({"prediction": predictions}, index=features.index)

    def forecast(self, steps: int) -> pd.DataFrame:
        """Forecast future volatility."""
        return pd.DataFrame({"forecast": [0] * steps})

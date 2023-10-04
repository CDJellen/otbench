from typing import Union

import pandas as pd
import numpy as np
import lightgbm as lgb

from otb.benchmark.models.forecasting.base_model import BaseForecastingModel


class RandomForestForecastingModel(BaseForecastingModel):
    """A model that uses a random forest regressor for direct forecasting."""

    def __init__(self, name: str, target_name: str, window_size: int, forecast_horizon: int, **kwargs):
        super().__init__(name, target_name, window_size, forecast_horizon, **kwargs)
        self._lgb_mdl = lgb.LGBMRegressor(**kwargs)

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Train the underlying LightGBM regressor."""
        self._lgb_mdl.fit(X, y)

    def predict(self, X: 'pd.DataFrame'):
        """Use the underlying LightGBM regressor to generate predictions."""
        return self._lgb_mdl.predict(X)

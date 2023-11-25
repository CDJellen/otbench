from typing import Union

import pandas as pd
import numpy as np

from otbench.benchmark.models.forecasting.base_model import BaseForecastingModel


class ClimatologyForecastingModel(BaseForecastingModel):
    """A model that predicts the mean value of the target variable seen during training."""

    def __init__(self,
                 name: str,
                 target_name: str,
                 window_size: int,
                 forecast_horizon: int,
                 time_col_name: Union[str, None] = None,
                 **kwargs):
        super().__init__(name, target_name, window_size, forecast_horizon, **kwargs)
        self.time_col_name = time_col_name if time_col_name is not None else None
        self.global_mean = np.nan

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Determine the mean value of the target variable seen during training."""
        X = X[[c for c in X.columns if c.startswith(self.target_name)]]

        self.global_mean = np.nanmean(X.values.flatten())

    def predict(self, X: 'pd.DataFrame'):
        """Predict the mean seen during training ."""
        return np.full(len(X), self.global_mean)

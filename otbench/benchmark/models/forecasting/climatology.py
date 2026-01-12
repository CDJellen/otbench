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

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', 'np.ndarray']):
        """Determine the mean value of the target variable seen during training."""
        targets = self.target_name if isinstance(self.target_name, list) else [self.target_name]
        self.global_mean = []

        for t in targets:
            cols = [c for c in X.columns if c.startswith(t)]
            mean_val = np.nanmean(X[cols].values.flatten())
            self.global_mean.append(mean_val)
        
        self.global_mean = np.array(self.global_mean)

    def predict(self, X: 'pd.DataFrame'):
        """Predict the mean seen during training."""
        n_samples = len(X)
        if len(self.global_mean) == 1:
            return np.full(n_samples, self.global_mean[0])
        else:
            return np.tile(self.global_mean, (n_samples, 1))

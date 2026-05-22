# otbench/benchmark/models/forecasting/climatology.py
from typing import Union
import pandas as pd
import numpy as np
from otbench.benchmark.models.forecasting.base_model import BaseForecastingModel

class ClimatologyForecastingModel(BaseForecastingModel):
    """A model that predicts the mean value of the target variable seen during training."""

    def __init__(self, name: str, target_name: str, window_size: int, forecast_horizon: int, **kwargs):
        super().__init__(name, target_name, window_size, forecast_horizon, **kwargs)
        self.global_mean = np.nan

    def _train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', 'np.ndarray']):
        """Determine the mean value of the target variable (or residual) seen during training."""
        # y is already processed (potentially residuals)
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
            
        # Compute mean along samples axis
        # Use nanmean to be safe
        self.global_mean = np.nanmean(y, axis=0)

    def _predict(self, X: 'pd.DataFrame'):
        """Predict the mean seen during training."""
        n_samples = len(X)
        if n_samples == 0:
            if np.ndim(self.global_mean) > 0:
                return np.empty((0, len(self.global_mean)))
            return np.array([])

        # Broadcast mean
        if np.ndim(self.global_mean) == 0:
            return np.full(n_samples, self.global_mean)
        else:
            return np.tile(self.global_mean, (n_samples, 1))
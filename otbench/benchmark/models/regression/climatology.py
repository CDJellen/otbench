from typing import Union

import pandas as pd
import numpy as np

from otbench.benchmark.models.regression.base_model import BaseRegressionModel


class ClimatologyRegressionModel(BaseRegressionModel):
    """A model that predicts the mean value of the target variable seen during training."""

    def __init__(self, name: str, target_name: str, time_col_name: Union[str, None] = None, **kwargs):
        super().__init__(name, target_name, **kwargs)
        self.time_col_name = time_col_name if time_col_name is not None else None
        self.global_mean = np.nan

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Determine the mean value of the target variable seen during training."""
        if isinstance(y, pd.Series):
            y = y.values
        elif isinstance(y, pd.DataFrame):
            y = y.values

        # Compute mean along the 0-th axis (samples) to handle both scalar (1D) and vector (2D) targets
        # resulting self.global_mean will be scalar or vector shape (n_features,)
        self.global_mean = np.nanmean(y, axis=0)

    def predict(self, X: 'pd.DataFrame'):
        """Predict the mean seen during training at the time of day for inference."""
        # If scalar, return shape (n_samples,)
        if np.ndim(self.global_mean) == 0:
            return np.full(len(X), self.global_mean)

        # If vector, return shape (n_samples, n_features)
        return np.tile(self.global_mean, (len(X), 1))

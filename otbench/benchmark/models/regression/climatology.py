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
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name=self.target_name, index=X.index)

        self.global_mean = np.nanmean(y[self.target_name].values.flatten())

    def predict(self, X: 'pd.DataFrame'):
        """Predict the mean seen during training at the time of day for inference."""
        return np.full(len(X), self.global_mean)

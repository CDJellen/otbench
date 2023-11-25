from typing import Union

import numpy as np

from otbench.benchmark.models.forecasting.base_model import BaseForecastingModel


class PersistenceForecastingModel(BaseForecastingModel):
    """A model which predicts the most recent value of the target variable."""

    def __init__(self, name: str, target_name: str, window_size: int, forecast_horizon: int, **kwargs):
        super().__init__(name, target_name, window_size, forecast_horizon, **kwargs)

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Maintain the same interface as the other models."""
        pass

    def predict(self, X: 'pd.DataFrame'):
        persistence = X[self.target_name].values[0]

        # develop a prediction for each row in X
        return np.array([persistence for i in range(len(X))])

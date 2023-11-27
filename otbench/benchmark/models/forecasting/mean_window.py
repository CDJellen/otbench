from typing import Union

import numpy as np

from otbench.benchmark.models.forecasting.base_model import BaseForecastingModel


class MeanWindowForecastingModel(BaseForecastingModel):
    """A model which predicts the mean value of the target variable from the input window."""

    def __init__(self, name: str, target_name: str, window_size: int, forecast_horizon: int, **kwargs):
        super().__init__(name, target_name, window_size, forecast_horizon, **kwargs)

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Maintain the same interface as the other models."""
        pass

    def predict(self, X: 'pd.DataFrame'):
        """Forecast the cn2 using the mean of the lagged values."""
        # predict the mean for each entry in X
        # X contains some number of lagged values of the target variable
        # we will use the mean of these lagged values as our prediction
        X = X[[c for c in X.columns if c.startswith(self.target_name)]]

        # develop a prediction for each row in X
        preds = []
        for i in range(len(X)):
            pred = np.nanmean(X.iloc[i, :].values)
            preds.append(pred)

        return np.array(preds)

from typing import Union

import pandas as pd
import numpy as np

from otb.benchmark.models.forecasting.base_model import BaseForecastingModel


class ClimatologyForecastingModel(BaseForecastingModel):
    """A model that predicts the mean value of the target variable for a given time."""

    def __init__(self, name: str, target_name: str, window_size: int, forecast_horizon: int, time_col_name: Union[str, None] = None, **kwargs):
        super().__init__(name, target_name, window_size, forecast_horizon, **kwargs)
        self.time_col_name = time_col_name if time_col_name is not None else None
        self.means = {}

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        # maintain the same interface as the other models
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name=self.target_name)
        data = X.join(y)

        # compute the mean for each interval in X across all days
        if isinstance(data.index, pd.DatetimeIndex):
            times = data.index.time
        else:
            times = data[self.time_col_name]
        times = np.unique(times)
        for t in times:
            self.means[t] = data.loc[data.index.time == t][self.target_name].mean()

    def predict(self, X: 'pd.DataFrame'):
        # predict the mean for each entry in X
        if isinstance(X.index, pd.DatetimeIndex):
            times = X.index.time
        else:
            times = X[self.time_col_name]
        # apply the forecast horizon to each time
        timedelta = times[1] - times[0]
        times = [pd.Timestamp(t) + timedelta for t in times]
        return np.array([self.means[t.time] for t in times])

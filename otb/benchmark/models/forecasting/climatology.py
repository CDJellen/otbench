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
        X = X[[c for c in X.columns if c.startswith(self.target_name)]]

        # compute the mean for each interval in X across all days
        if isinstance(X.index, pd.DatetimeIndex):
            times = X.index.time
        else:
            times = X[self.time_col_name]
            times = times.apply(lambda x: pd.to_datetime(x).time())
        times = np.unique(times)
        for t in times:
            self.means[t] = np.nanmean(X.loc[X.index.time == t].values)

    def predict(self, X: 'pd.DataFrame'):
        # predict the mean for each entry in X
        if isinstance(X.index, pd.DatetimeIndex):
            times = X.index
        else:
            times = X[self.time_col_name]
            times = times.apply(lambda x: pd.to_datetime(x))
        # apply the forecast horizon to each time
        time_step = X.index[1] - X.index[0]
        timedelta = pd.Timedelta(seconds=self.forecast_horizon * time_step.total_seconds())
        # add the timedelta to each time
        times = [(t + timedelta).time() for t in times]
        return np.array([self.means[t] for t in times])

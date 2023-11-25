from typing import Union

import pandas as pd
import numpy as np

from otbench.benchmark.models.forecasting.base_model import BaseForecastingModel


class MinuteClimatologyForecastingModel(BaseForecastingModel):
    """A model that predicts the mean value of the target variable for a given time seen during training."""

    def __init__(self,
                 name: str,
                 target_name: str,
                 window_size: int,
                 forecast_horizon: int,
                 time_col_name: Union[str, None] = None,
                 **kwargs):
        super().__init__(name, target_name, window_size, forecast_horizon, **kwargs)
        self.time_col_name = time_col_name if time_col_name is not None else None
        self.means = {}
        self.global_mean = np.nan

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Determine the mean value of the target variable seen during training for each time."""
        X = X[[c for c in X.columns if c.startswith(self.target_name)]]

        self.global_mean = np.nanmean(X.values.flatten())

        # compute the mean for each interval in X across all days
        X["time_of_day"] = X.index.time
        X_means = X.groupby("time_of_day").mean()
        # iterate through the rows in X_means
        for i in range(len(X_means)):
            self.means[X_means.index[i]] = np.nanmean(X_means.iloc[i, :].values)

    def predict(self, X: 'pd.DataFrame'):
        """Predict the mean seen during training at the time of day of forecast."""
        times = X.index
        times = pd.to_datetime(times)

        # apply the forecast horizon to each time
        time_step = times[1] - times[0]
        timedelta = pd.Timedelta(seconds=self.forecast_horizon * time_step.total_seconds())

        # add the timedelta to each time
        times = times + timedelta

        # convert to time
        preds = []
        for time in times.time:
            if time in self.means and not np.isnan(self.means[time]):
                preds.append(self.means[time])
            else:
                preds.append(self.global_mean)

        return np.array(preds)

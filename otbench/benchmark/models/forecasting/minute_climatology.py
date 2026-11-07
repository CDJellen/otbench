from typing import Union

import pandas as pd
import numpy as np

from otbench.benchmark.models.forecasting.base_model import BaseForecastingModel


class MinuteClimatologyForecastingModel(BaseForecastingModel):
    """A model that predicts the mean value of the target variable for a given time seen during training."""

    def __init__(self, name: str, target_name: str, window_size: int, forecast_horizon: int, **kwargs):
        super().__init__(name, target_name, window_size, forecast_horizon, **kwargs)
        self.means = {}
        self.global_mean = np.nan

    def _train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Determine the mean value of the target variable seen during training for each time."""
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y, index=X.index)
        elif isinstance(y, pd.Series):
            y = y.to_frame()

        self.global_mean = np.nanmean(y.values, axis=0)

        y_temp = y.copy()
        y_temp["time_of_day"] = X.index.time
        grouped = y_temp.groupby("time_of_day").mean()
        self.means = {t: row.values for t, row in grouped.iterrows()}

    def _predict(self, X: 'pd.DataFrame'):
        """Predict the mean seen during training at the time of day of forecast."""
        if len(X) == 0:
            if np.ndim(self.global_mean) > 0:
                return np.empty((0, len(self.global_mean)))
            return np.array([])

        # shift timestamps by the forecast horizon to determine target time of day
        times = pd.to_datetime(X.index)
        if len(times) > 1:
            diffs = pd.Series(times).diff().dropna()
            positive_diffs = diffs[diffs > pd.Timedelta(0)]
            dt = positive_diffs.median() if len(positive_diffs) > 0 else pd.Timedelta(minutes=1)
            shift = pd.Timedelta(seconds=self.forecast_horizon * dt.total_seconds())
            target_times = (times + shift).time
        else:
            target_times = (times + pd.Timedelta(minutes=self.forecast_horizon)).time

        preds = []
        for t in target_times:
            if t in self.means:
                preds.append(self.means[t])
            else:
                preds.append(self.global_mean)

        return np.array(preds)
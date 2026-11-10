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
            shifted_dt = times + shift
        else:
            shifted_dt = times + pd.Timedelta(minutes=self.forecast_horizon)

        if (self.means and np.all(shifted_dt.second == 0) and np.all(shifted_dt.microsecond == 0)
                and all(t.second == 0 and t.microsecond == 0 for t in self.means)):
            g_mean = np.asarray(self.global_mean, dtype=float)
            table = np.broadcast_to(g_mean, (1440,) + g_mean.shape).copy()
            for t, val in self.means.items():
                val_arr = np.asarray(val, dtype=float)
                if not np.isnan(val_arr).all():
                    table[t.hour * 60 + t.minute] = val_arr
            minute_idx = shifted_dt.hour.to_numpy() * 60 + shifted_dt.minute.to_numpy()
            return table[minute_idx]

        target_times = shifted_dt.time
        preds = []
        for t in target_times:
            if t in self.means:
                preds.append(self.means[t])
            else:
                preds.append(self.global_mean)

        return np.array(preds)
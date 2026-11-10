from typing import Union

import pandas as pd
import numpy as np

from otbench.benchmark.models.regression.base_model import BaseRegressionModel


class MinuteClimatologyRegressionModel(BaseRegressionModel):
    """A model that predicts the mean value of the target variable for a given time seen during training."""

    def __init__(self, name: str, target_name: str, time_col_name: Union[str, None] = None, **kwargs):
        super().__init__(name, target_name, **kwargs)
        self.time_col_name = time_col_name if time_col_name is not None else None
        self.means = {}
        self.global_mean = np.nan

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Determine the mean value of the target variable seen during training for each time."""
        # Ensure y is proper format matching X index
        if isinstance(y, np.ndarray):
            if y.ndim > 1:
                y = pd.DataFrame(y, index=X.index)
            else:
                y = pd.DataFrame(y, columns=[self.target_name], index=X.index)

        if isinstance(y, pd.Series):
            y = y.to_frame()

        y = y.copy()

        # Compute global mean
        if isinstance(y, pd.DataFrame):
            self.global_mean = np.nanmean(y.values, axis=0)
        else:
            self.global_mean = np.nanmean(y.values)

        # compute the mean for each interval in X across all days
        y["time_of_day"] = y.index.time
        y_means = y.groupby("time_of_day").mean()

        # Store means
        self.means = {}
        for t in y_means.index:
            self.means[t] = y_means.loc[t].values

    def predict(self, X: 'pd.DataFrame'):
        """Predict the mean seen during training at the time of day for inference."""
        if len(X) == 0:
            if np.ndim(self.global_mean) > 0:
                return np.empty((0, len(self.global_mean)))
            return np.array([])

        dt_index = pd.to_datetime(X.index)
        if (self.means and np.all(dt_index.second == 0) and np.all(dt_index.microsecond == 0)
                and all(t.second == 0 and t.microsecond == 0 for t in self.means)):
            g_mean = np.asarray(self.global_mean, dtype=float)
            table = np.broadcast_to(g_mean, (1440,) + g_mean.shape).copy()
            for t, val in self.means.items():
                val_arr = np.asarray(val, dtype=float)
                if not np.isnan(val_arr).all():
                    table[t.hour * 60 + t.minute] = val_arr
            minute_idx = dt_index.hour.to_numpy() * 60 + dt_index.minute.to_numpy()
            return table[minute_idx]

        times = dt_index.time
        preds = []
        for time in times:
            if time in self.means:
                val = self.means[time]
                if np.isnan(val).all():
                    preds.append(self.global_mean)
                else:
                    preds.append(val)
            else:
                preds.append(self.global_mean)

        return np.array(preds)

from typing import Union

import pandas as pd
import numpy as np

from otb.benchmark.models.regression.base_model import BaseRegressionModel


class ClimatologyRegressionModel(BaseRegressionModel):
    """A model that predicts the mean value of the target variable for a given time."""

    def __init__(self, name: str, target_name: str, time_col_name: Union[str, None] = None, **kwargs):
        super().__init__(name, target_name, **kwargs)
        self.time_col_name = time_col_name if time_col_name is not None else None
        self.means = {}

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        # maintain the same interface as the other models
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name=self.target_name)
        data = X.join(y)

        # compute the mean for each interval in X across all days
        if isinstance(data.index, pd.DatetimeIndex):
            times = data.index
        else:
            times = data[self.time_col_name]
        times = times.apply(lambda x: pd.to_datetime(x).time())
        # get unique times from datetimes
        times = np.unique(times)
        for t in times:
            self.means[t] = np.nanmean(data.loc[data.index.time == t][self.target_name].values)

    def predict(self, X: 'pd.DataFrame'):
        # predict the mean for each entry in X
        if isinstance(X.index, pd.DatetimeIndex):
            times = X.index.time
        else:
            times = X[self.time_col_name]
            times = times.apply(lambda x: pd.to_datetime(x).time())
        return np.array([self.means[t] for t in times])

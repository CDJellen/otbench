from typing import Union

import pandas as pd
import numpy as np


class ClimatologyRegressionModel:
    """A model that predicts the mean value of the target variable for a given time."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.means = {}

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        # maintain the same interface as the other models
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name='target')
        elif isinstance(y, pd.DataFrame):
            target_name = [c for c in y.columns if c != 'time'][0]
            y = y[target_name].rename('target')
        else:
            y = y.rename('target')
        data = X.join(y)

        # compute the mean for each interval in X across all days
        if isinstance(data.index, pd.DatetimeIndex):
            times = data.index.time
        else:
            times = data['time']
        times = np.unique(times)
        for t in times:
            self.means[t] = data.loc[data.index.time == t]['target'].mean()

    def predict(self, X: 'pd.DataFrame'):
        # predict the mean for each entry in X
        if isinstance(X.index, pd.DatetimeIndex):
            times = X.index.time
        else:
            times = X['time']
        return np.array([self.means[t] for t in times])

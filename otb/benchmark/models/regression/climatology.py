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
        self.global_mean = np.nan

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        # maintain the same interface as the other models
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name=self.target_name, index=X.index)
        self.global_mean = np.nanmean(y.values.flatten())
        # compute the mean for each interval in X across all days
        y["time_of_day"] = y.index.time
        y_means = y.groupby("time_of_day").mean()
        # iterate through the rows in X_means
        for i in range(len(y_means)):
            self.means[y_means.index[i]] = np.nanmean(y_means.iloc[i, :].values)

    def predict(self, X: 'pd.DataFrame'):
        # predict the mean for each entry in X
        times = X.index
        times = pd.to_datetime(times).time
        
        preds = []
        
        for time in times:
            if time in self.means and not np.isnan(self.means[time]):
                preds.append(self.means[time])
            else:
                preds.append(self.global_mean)
        return np.array(preds)

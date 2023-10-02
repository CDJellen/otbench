from typing import Union

import numpy as np

from otb.benchmark.models.regression.base_model import BaseRegressionModel


class PersistanceRegressionModel(BaseRegressionModel):

    def __init__(self, name: str, target_name: str, **kwargs):
        super().__init__(name, target_name, **kwargs)
        self.mean = np.nan

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        # maintain the same interface as the other models
        self.mean = np.nanmean(y)

    def predict(self, X: 'pd.DataFrame'):
        # predict the mean for each entry in X
        return np.full(len(X), self.mean)

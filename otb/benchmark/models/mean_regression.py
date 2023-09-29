from typing import Union

import numpy as np


class MeanRegressionModel:

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.mean = np.nan

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        # maintain the same interface as the other models
        self.mean = np.mean(y)

    def predict(self, X: 'pd.DataFrame'):
        # predict the mean for each entry in X
        return np.full(len(X), self.mean)

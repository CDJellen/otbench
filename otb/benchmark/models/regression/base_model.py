from typing import Union

import numpy as np


class BaseRegressionModel:

    def __init__(self, name: str, target_name: str, **kwargs):
        self.name = name
        self.target_name = target_name

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        # maintain the same interface as the other models
        raise NotImplementedError

    def predict(self, X: 'pd.DataFrame'):
        # predict for each entry in X
        raise NotImplementedError

from typing import Union

import numpy as np

from otbench.benchmark.models.regression.base_model import BaseRegressionModel


class PersistenceRegressionModel(BaseRegressionModel):
    """A model which predicts the most recent value of the target variable."""

    def __init__(self, name: str, target_name: str, **kwargs):
        super().__init__(name, target_name, **kwargs)
        self.persistence = np.nan

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Maintain the same interface as the other models."""
        self.persistence = y.values[-1]

    def predict(self, X: 'pd.DataFrame'):
        # predict the mean for each entry in X
        return np.full(len(X), self.persistence)

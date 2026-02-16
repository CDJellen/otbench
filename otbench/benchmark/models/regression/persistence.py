from typing import Union

import numpy as np

from otbench.benchmark.models.regression.base_model import BaseRegressionModel


class PersistenceRegressionModel(BaseRegressionModel):
    """A model which predicts the most recent value of the target variable."""

    def __init__(self, name: str, target_name: str, **kwargs):
        super().__init__(name, target_name, **kwargs)
        self.persistence = np.nan
        self.output_size = kwargs.get("output_size", 1)

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Maintain the same interface as the other models."""
        if len(y) == 0:
            return
        self.persistence = y.values[-1]

    def predict(self, X: 'pd.DataFrame'):
        if len(X) == 0:
            if self.output_size > 1:
                return np.empty((0, self.output_size))
            return np.array([])

        if np.ndim(self.persistence) > 0:
            return np.tile(self.persistence, (len(X), 1))
        
        # Handle scalar persistence for vector output (broadcast)
        if self.output_size > 1:
             return np.tile(self.persistence, (len(X), self.output_size))
             
        return np.full(len(X), self.persistence)
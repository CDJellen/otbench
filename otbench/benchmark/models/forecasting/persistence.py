from typing import Union

import numpy as np

from otbench.benchmark.models.regression.base_model import BaseRegressionModel


class PersistenceForecastingModel(BaseRegressionModel):
    """A model which predicts the most recent value of the target variable."""
    def __init__(self, name: str, target_name: str, **kwargs):
        super().__init__(name, target_name, **kwargs)
        self.persistence = None
        self.output_size = kwargs.get("output_size", 1)  # Passed by bench_runner

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Store the last observed value(s)."""
        y = np.asarray(y)
        self.persistence = y[-1]  # scalar or vector

    def predict(self, X: 'pd.DataFrame'):
        if self.persistence is None:
            raise RuntimeError("Model must be trained before prediction.")

        n_samples = len(X)
        if np.ndim(self.persistence) == 0 or (np.ndim(self.persistence) == 1 and len(self.persistence) == 1):
            # Scalar
            return np.full(n_samples, self.persistence)
        else:
            # Vector: tile the last observed vector
            return np.tile(self.persistence, (n_samples, 1))

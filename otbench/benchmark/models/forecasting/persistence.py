from typing import Union, List

import numpy as np
import pandas as pd

from otbench.benchmark.models.forecasting.base_model import BaseForecastingModel


class PersistenceForecastingModel(BaseForecastingModel):
    """A model which predicts the most recent value of the target variable."""

    def __init__(self, name: str, target_name: Union[str, List[str]], **kwargs):
        super().__init__(name, target_name, **kwargs)
        self.static_persistence = np.nan
        self.output_size = kwargs.get("output_size", 1)
        self.target_cols = target_name if isinstance(target_name, list) else [target_name]

    def _train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Store the last observed value(s) as a static fallback."""
        y_arr = np.asarray(y)
        if len(y_arr) == 0:
            self.static_persistence = np.nan
            return

        # store the last valid row of the training target
        valid_mask = ~np.isnan(y_arr).any(axis=1) if y_arr.ndim > 1 else ~np.isnan(y_arr)
        if np.any(valid_mask):
            self.static_persistence = y_arr[valid_mask][-1]
        else:
            self.static_persistence = y_arr[-1]

    def _predict(self, X: 'pd.DataFrame'):
        """Predict using dynamic persistence if target features are available, else static fallback."""
        n_samples = len(X)

        if n_samples == 0:
            if self.output_size > 1:
                return np.empty((0, self.output_size))
            return np.array([])

        # in residual mode, persistence predicts zero change from t to t+k
        if self.predict_residuals:
            if self.output_size > 1:
                return np.zeros((n_samples, self.output_size))
            return np.zeros(n_samples)

        try:
            feature_cols = self._find_features_for_targets(X)
            vals = X[feature_cols].values
            if self.use_log10:
                vals = np.log10(np.maximum(vals, 1e-19))
            return vals
        except ValueError:
            if np.all(np.isnan(self.static_persistence)):
                raise RuntimeError("model has not been trained and X does not contain target history.")

            if np.ndim(self.static_persistence) == 0 or (np.ndim(self.static_persistence) == 1
                                                         and len(self.static_persistence) == 1):
                return np.full(n_samples, self.static_persistence)

            return np.tile(self.static_persistence, (n_samples, 1))

    def _find_features_for_targets(self, X: pd.DataFrame) -> List[str]:
        """Identify columns in X that correspond to the target variable(s) at t=0."""
        found_cols = []
        for t in self.target_cols:
            if t in X.columns:
                found_cols.append(t)
            elif f"{t} (t-0)" in X.columns:
                found_cols.append(f"{t} (t-0)")
            else:
                raise ValueError(f"target '{t}' not found in X features.")
        return found_cols

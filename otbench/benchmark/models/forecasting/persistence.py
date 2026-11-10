from typing import List, Union

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

        valid_mask = ~np.isnan(y_arr).any(axis=1) if y_arr.ndim > 1 else ~np.isnan(y_arr)
        self.static_persistence = y_arr[valid_mask][-1] if np.any(valid_mask) else y_arr[-1]

    def _predict(self, X: Union['pd.DataFrame', np.ndarray]):
        """Predict using dynamic persistence if target features are available, else static fallback."""
        n_samples = len(X)
        if n_samples == 0:
            return np.empty((0, self.output_size)) if self.output_size > 1 else np.array([])

        if self.predict_residuals:
            return np.zeros((n_samples, self.output_size)) if self.output_size > 1 else np.zeros(n_samples)

        if isinstance(X, np.ndarray) and X.ndim == 3:
            p_idx = self._persistence_indices_3d or self._find_persistence_indices_3d(X.shape[2], strict=False)
            if p_idx:
                vals = np.asarray(X[:, -1, p_idx], dtype=float).copy()
                return self._maybe_log10_array(vals)
            return self._fallback_static_persistence(n_samples)

        try:
            feature_cols = self._find_features_for_targets(X)
            vals = X[feature_cols].to_numpy(dtype=float).copy()
            return self._maybe_log10_array(vals)
        except ValueError:
            return self._fallback_static_persistence(n_samples)

    def _fallback_static_persistence(self, n_samples: int) -> np.ndarray:
        """Return tiled static persistence fallback or raise if untrained."""
        if np.all(np.isnan(self.static_persistence)):
            raise RuntimeError("model has not been trained and X does not contain target history.")

        if np.ndim(self.static_persistence) == 0 or (np.ndim(self.static_persistence) == 1
                                                     and len(self.static_persistence) == 1):
            return np.full(n_samples, self.static_persistence)

        return np.tile(self.static_persistence, (n_samples, 1))

    def _find_features_for_targets(self, X: pd.DataFrame) -> List[str]:
        """Identify columns in X that correspond to the target variable(s) at t=0."""
        return self._find_persistence_cols(X, pd.DataFrame(columns=self.target_cols))

from typing import Union

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

from otbench.benchmark.models._lgbm_utils import (
    _NON_LGBM_KEYS,
    build_lgbm_estimator,
    filter_valid_target_rows,
)
from otbench.benchmark.models.forecasting.base_model import BaseForecastingModel


class GradientBoostingForecastingModel(BaseForecastingModel):
    """A model that uses gradient boosting regression trees for direct forecasting."""

    _NON_LGBM_KEYS = _NON_LGBM_KEYS

    def __init__(self,
                 name: str,
                 target_name: str,
                 window_size: int,
                 forecast_horizon: int,
                 output_size: int = 1,
                 **kwargs):
        super().__init__(name, target_name, window_size, forecast_horizon, **kwargs)
        self._lgb_mdl, self.output_size = build_lgbm_estimator(output_size, target_name, kwargs)

    @staticmethod
    def _flatten_3d_sequence(X: np.ndarray) -> np.ndarray:
        """Convert a 3D sequence tensor (N, W, D) into a 2D tabular feature matrix.

        For compact windows (W * D <= 400), all lags are flattened newest-first.
        For high-dimensional long-window tensors (W * D > 400), extracts a multi-scale
        logarithmically spaced lag set [0, 1, 2, 4, 9, 19, 39, W-1] concatenated with
        full-window mean, full-window standard deviation, and short-window (10-step) mean.
        """
        n_samples, window_size, n_features = X.shape
        if window_size * n_features <= 400:
            return np.ascontiguousarray(X[:, ::-1, :]).reshape(n_samples, -1)
        x_rev = X[:, ::-1, :].astype(np.float32, copy=False)
        lag_idx = sorted({i for i in (0, 1, 2, 4, 9, 19, 39, window_size - 1) if i < window_size})
        lags = x_rev[:, lag_idx, :].reshape(n_samples, -1)
        mean_all = x_rev.mean(axis=1)
        std_all = x_rev.std(axis=1)
        mean_short = x_rev[:, :min(10, window_size), :].mean(axis=1)
        return np.hstack([lags, mean_all, std_all, mean_short])

    def _train(self, X: Union['pd.DataFrame', np.ndarray], y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Internal training implementation."""
        if len(X) == 0:
            return
        if isinstance(X, np.ndarray) and X.ndim == 3:
            X = self._flatten_3d_sequence(X)
        X, y = filter_valid_target_rows(X, y)
        if len(X) == 0:
            return
        if self.output_size == 1 and isinstance(y, np.ndarray) and y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        self._lgb_mdl.fit(X, y)

    def _predict(self, X: Union['pd.DataFrame', np.ndarray]):
        """Internal prediction implementation."""
        if len(X) == 0:
            if isinstance(self._lgb_mdl, MultiOutputRegressor):
                n_targets = len(self.target_name) if isinstance(self.target_name, list) else self.output_size
                return np.empty((0, n_targets))
            return np.array([])
        if isinstance(X, np.ndarray) and X.ndim == 3:
            X = self._flatten_3d_sequence(X)
        return self._lgb_mdl.predict(X)

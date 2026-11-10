from typing import Union

import numpy as np
import pandas as pd

from otbench.benchmark.models.forecasting.base_model import BaseForecastingModel


class LinearForecastingModel(BaseForecastingModel):
    """A model that fits a line to the lagged values of the target variable per sample."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._training_mean = None  # fallback when target columns absent from X

    def _train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Store the per-target training mean as a fallback for targets not in X."""
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            y_arr = y_arr[:, np.newaxis]
        self._training_mean = np.nanmean(y_arr, axis=0)

    def _predict(self, X: 'pd.DataFrame'):
        """Forecast by fitting a line to the lagged values."""
        targets = self.target_name if isinstance(self.target_name, list) else [self.target_name]
        if len(X) == 0:
            return np.empty((0, len(targets))) if len(targets) > 1 else np.array([])

        all_preds = []
        for i, t in enumerate(targets):
            fallback = (float(self._training_mean[i] if len(self._training_mean) > 1 else self._training_mean[0])
                        if self._training_mean is not None else 0.0)
            X_t = self._extract_target_window(X, t, all_targets=targets)
            if X_t is None:
                all_preds.append(np.full(len(X), fallback))
                continue

            # lag columns are ordered [feat(t-0), feat(t-1), ..., feat(t-(W-1))]
            n_lags = X_t.shape[1]
            x_axis = np.arange(0, -n_lags, -1)
            A = np.vstack([x_axis, np.ones(n_lags)]).T

            if not np.isnan(X_t).any():
                coeffs = np.linalg.lstsq(A, X_t.T, rcond=None)[0]
                preds_t = coeffs[0] * self.forecast_horizon + coeffs[1]
            else:
                preds_t = np.empty(len(X_t), dtype=float)
                for row_i in range(len(X_t)):
                    history = X_t[row_i, :]
                    valid = ~np.isnan(history)
                    n_valid = int(valid.sum())
                    if n_valid == 0:
                        preds_t[row_i] = fallback
                    elif n_valid == 1:
                        preds_t[row_i] = float(history[valid][0])
                    else:
                        m, b = np.linalg.lstsq(A[valid], history[valid], rcond=None)[0]
                        preds_t[row_i] = float(m * self.forecast_horizon + b)

            if self.predict_residuals:
                y_curr_t = X_t[:, 0]
                all_nan_row = np.isnan(X_t).all(axis=1)
                preds_t = np.where(all_nan_row, preds_t, preds_t - y_curr_t)

            all_preds.append(preds_t)

        return np.array(all_preds).T

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
        all_preds = []

        if len(X) == 0:
            if len(targets) > 1:
                return np.empty((0, len(targets)))
            return np.array([])

        for i, t in enumerate(targets):
            cols = [c for c in X.columns if c.startswith(t)]

            if not cols:
                if self._training_mean is not None:
                    fallback = float(self._training_mean[i] if len(self._training_mean) > 1 else self._training_mean[0])
                else:
                    fallback = 0.0
                all_preds.append(np.full(len(X), fallback))
                continue

            X_t = X[cols].values

            # transform target history to log10 space when target is log-transformed
            if self.use_log10:
                X_t = np.log10(np.maximum(X_t, 1e-19))

            # lag columns are ordered [feat(t-0), feat(t-1), ..., feat(t-(W-1))]
            n_lags = X_t.shape[1]
            x_axis = np.arange(0, -n_lags, -1)
            A = np.vstack([x_axis, np.ones(n_lags)]).T

            if not np.isnan(X_t).any():
                coeffs = np.linalg.lstsq(A, X_t.T, rcond=None)[0]
                preds_t = coeffs[0] * self.forecast_horizon + coeffs[1]
            else:
                preds_t = []
                for row_i in range(len(X_t)):
                    history = X_t[row_i, :]
                    m, b = np.linalg.lstsq(A, history, rcond=None)[0]
                    pred = m * self.forecast_horizon + b
                    preds_t.append(pred)

            all_preds.append(preds_t)

        return np.array(all_preds).T

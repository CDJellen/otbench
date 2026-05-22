# otbench/benchmark/models/forecasting/linear.py
from typing import Union
import numpy as np
import pandas as pd
from otbench.benchmark.models.forecasting.base_model import BaseForecastingModel

class LinearForecastingModel(BaseForecastingModel):
    """A model that fits a line to the lagged values of the target variable per sample.

    When target history columns are absent from X (e.g., the target is in the task's
    remove list), the model falls back to the global training mean rather than zero.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._training_mean = None  # fallback when target columns absent from X

    def _train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Store the per-target training mean as a fallback for targets not in X."""
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            y_arr = y_arr[:, np.newaxis]
        self._training_mean = np.nanmean(y_arr, axis=0)  # shape (n_targets,)

    def _predict(self, X: 'pd.DataFrame'):
        """Forecast by fitting a line to the lagged values."""
        targets = self.target_name if isinstance(self.target_name, list) else [self.target_name]
        all_preds = []

        if len(X) == 0:
            if len(targets) > 1: return np.empty((0, len(targets)))
            return np.array([])

        for i, t in enumerate(targets):
            # Identify history columns (fuzzy match startswith)
            cols = [c for c in X.columns if c.startswith(t)]

            # Safety check
            if not cols:
                # Target history not in X; fall back to training mean (not zero)
                if self._training_mean is not None:
                    fallback = float(self._training_mean[i] if len(self._training_mean) > 1
                                     else self._training_mean[0])
                else:
                    fallback = 0.0
                all_preds.append(np.full(len(X), fallback))
                continue

            X_t = X[cols].values
            
            # CRITICAL: Transform History to Log Space if target is Log
            if self.use_log10:
                X_t = np.log10(np.maximum(X_t, 1e-19))

            preds_t = []

            # otbench lag columns are ordered [feat(t-0), feat(t-1), ..., feat(t-(W-1))],
            # so cols[0] = current value and cols[-1] = oldest value.
            # Map x so that x=0 corresponds to the current time-step and negative x
            # corresponds to the past: x = [0, -1, -2, ..., -(W-1)].
            # A positive slope m then means the series is growing, and projecting to
            # x = forecast_horizon correctly lands H steps into the future.
            n_lags = X_t.shape[1]
            x_axis = np.arange(0, -n_lags, -1)  # [0, -1, ..., -(W-1)]

            # Vectorized implementation of per-row linear regression is hard in pure numpy without loop
            # Keeping the loop for safety/clarity as in original
            for row_i in range(len(X_t)):
                history = X_t[row_i, :]

                # Fit line: y = mx + b
                A = np.vstack([x_axis, np.ones(len(history))]).T
                m, b = np.linalg.lstsq(A, history, rcond=None)[0]

                # Project H steps into the future (x = +forecast_horizon).
                pred = m * self.forecast_horizon + b
                preds_t.append(pred)

            all_preds.append(preds_t)

        return np.array(all_preds).T
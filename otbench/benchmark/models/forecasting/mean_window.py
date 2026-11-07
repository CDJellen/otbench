from typing import Union

import numpy as np
import pandas as pd

from otbench.benchmark.models.forecasting.base_model import BaseForecastingModel


class MeanWindowForecastingModel(BaseForecastingModel):
    """A model which predicts the mean value of the target variable from the input window.

    When target history columns are absent from X (e.g., the target is in the task's
    remove list), the model falls back to the global training mean — the same prediction
    as ClimatologyForecastingModel — rather than returning NaN.
    """

    def __init__(self, name: str, target_name: str, window_size: int, forecast_horizon: int, **kwargs):
        super().__init__(name, target_name, window_size, forecast_horizon, **kwargs)
        self._training_mean = None  # fallback when target columns absent from X

    def _train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Store the per-target training mean as a fallback for targets not in X."""
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            y_arr = y_arr[:, np.newaxis]
        self._training_mean = np.nanmean(y_arr, axis=0)  # shape (n_targets,)

    def _predict(self, X: 'pd.DataFrame'):
        """Forecast using the mean of the in-window lagged values.

        Falls back to the global training mean when the target variable is not
        present in X (e.g., task removes the target from features).
        """
        targets = self.target_name if isinstance(self.target_name, list) else [self.target_name]

        # Guard against empty input
        if len(X) == 0:
            return np.empty((0, len(targets)))

        all_preds = []

        for i, t in enumerate(targets):
            cols = [c for c in X.columns if c == t or c.startswith(f"{t} (t-") or c.startswith(f"{t}_t-")]
            if not cols:
                other_targets = [ot for ot in targets if ot != t and ot.startswith(t)]
                cols = [c for c in X.columns if c.startswith(t) and not any(c.startswith(ot) for ot in other_targets)]

            if not cols:
                # Target history not in X; fall back to training mean
                if self._training_mean is not None:
                    fallback = float(self._training_mean[i] if len(self._training_mean) > 1 else self._training_mean[0])
                else:
                    fallback = 0.0
                all_preds.append(np.full(len(X), fallback))
                continue

            p_col_idx = (cols.index(f"{t} (t-0)") if f"{t} (t-0)" in cols else
                         (cols.index(f"{t}_t-0") if f"{t}_t-0" in cols else 0))
            if p_col_idx > 0:
                cols = list(reversed(cols))

            X_t = X[cols].to_numpy(dtype=float).copy()
            if self.use_log10:
                valid_vals = X_t[np.isfinite(X_t)]
                if len(valid_vals) > 0 and np.min(valid_vals) >= 0.0 and np.median(valid_vals) < 1e-5:
                    X_t = np.log10(np.maximum(X_t, 1e-19))

            preds_t = np.nanmean(X_t, axis=1)
            if self.predict_residuals:
                preds_t = preds_t - X_t[:, 0]
            all_preds.append(preds_t)

        return np.array(all_preds).T

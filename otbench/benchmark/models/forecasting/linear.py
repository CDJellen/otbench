# otbench/benchmark/models/forecasting/linear.py
from typing import Union
import numpy as np
import pandas as pd
from otbench.benchmark.models.forecasting.base_model import BaseForecastingModel

class LinearForecastingModel(BaseForecastingModel):
    """A model that fits a line to the lagged values of the target variable per sample."""

    def _train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        pass # Lazy learner

    def _predict(self, X: 'pd.DataFrame'):
        """Forecast by fitting a line to the lagged values."""
        targets = self.target_name if isinstance(self.target_name, list) else [self.target_name]
        all_preds = []
        
        if len(X) == 0:
            if len(targets) > 1: return np.empty((0, len(targets)))
            return np.array([])

        for t in targets:
            # Identify history columns (fuzzy match startswith)
            cols = [c for c in X.columns if c.startswith(t)]
            
            # Safety check
            if not cols:
                # If no history, fallback to 0 (mean)
                all_preds.append(np.zeros(len(X)))
                continue

            X_t = X[cols].values
            
            # CRITICAL: Transform History to Log Space if target is Log
            if self.use_log10:
                X_t = np.log10(np.maximum(X_t, 1e-19))

            preds_t = []
            
            # Pre-compute X-axis for regression: [-W, ..., -1, 0]
            # Assuming cols are usually sorted [t, t-1, t-2...] or [t-W ... t]
            # We assume standard otbench lag order. 
            n_lags = X_t.shape[1]
            x_axis = np.arange(n_lags) 
            
            # Vectorized implementation of per-row linear regression is hard in pure numpy without loop
            # Keeping the loop for safety/clarity as in original
            for i in range(len(X_t)):
                history = X_t[i, :]
                
                # Fit line: y = mx + b
                A = np.vstack([x_axis, np.ones(len(history))]).T
                m, b = np.linalg.lstsq(A, history, rcond=None)[0]

                # Project forward
                # If history is [t-W ... t], then x_axis is 0..W
                # We want t + H.
                # The step size is 1.
                # So we project to W + H.
                pred = m * (n_lags + self.forecast_horizon) + b
                preds_t.append(pred)

            all_preds.append(preds_t)

        return np.array(all_preds).T
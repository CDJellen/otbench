from typing import Union

import numpy as np
import pandas as pd

from otbench.benchmark.models.forecasting.base_model import BaseForecastingModel


class MeanWindowForecastingModel(BaseForecastingModel):
    """A model which predicts the mean value of the target variable from the input window."""

    def __init__(self, name: str, target_name: str, window_size: int, forecast_horizon: int, **kwargs):
        super().__init__(name, target_name, window_size, forecast_horizon, **kwargs)

    def _train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Maintain the same interface as the other models."""
        pass

    def _predict(self, X: 'pd.DataFrame'):
        """Forecast the using the mean of the lagged values."""
        
        # Guard against empty input
        if len(X) == 0:
            targets = self.target_name if isinstance(self.target_name, list) else [self.target_name]
            return np.empty((0, len(targets)))

        # Handle Vector vs Scalar
        targets = self.target_name if isinstance(self.target_name, list) else [self.target_name]
        all_preds = []

        for t in targets:
            cols = [c for c in X.columns if c.startswith(t)]
            
            # If no history columns found (rare), append zeros or NaNs
            if not cols:
                all_preds.append(np.full(len(X), np.nan))
                continue

            X_t = X[cols]

            preds_t = []
            for i in range(len(X_t)):
                pred = np.nanmean(X_t.iloc[i, :].values)
                preds_t.append(pred)

            all_preds.append(preds_t)

        all_preds = np.array(all_preds).T

        return all_preds
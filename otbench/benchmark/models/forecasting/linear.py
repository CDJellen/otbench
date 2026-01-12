from typing import Union

import numpy as np

from otbench.benchmark.models.forecasting.base_model import BaseForecastingModel


class LinearForecastingModel(BaseForecastingModel):
    """A model that fits a line to the lagged values of the target variable."""

    def __init__(self, name: str, target_name: str, window_size: int, forecast_horizon: int, **kwargs):
        super().__init__(name, target_name, window_size, forecast_horizon, **kwargs)

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Maintain the same interface as the other models."""
        pass

    def predict(self, X: 'pd.DataFrame'):
        """Forecast the by fitting a line using the lagged values."""
        # Handle Vector vs Scalar
        targets = self.target_name if isinstance(self.target_name, list) else [self.target_name]
        all_preds = []

        # interpolate X to fill in missing values
        X = X.interpolate(method="time")

        for t in targets:
            # Filter columns just for this target variable
            # We must ensure we don't accidentally pick up other targets if they share prefixes
            # Assuming 'startswith' logic from before, but scoped to single target 't'
            cols = [c for c in X.columns if c.startswith(t)]
            X_t = X[cols]

            preds_t = []
            for i in range(len(X_t)):
                # fit a line to the lagged values
                lagged_values = X_t.iloc[i, :].values
                A = np.vstack([np.arange(len(lagged_values)), np.ones(len(lagged_values))]).T
                m, b = np.linalg.lstsq(A, lagged_values, rcond=None)[0]

                # predict the next value at the forecast horizon
                pred = m * (len(lagged_values) + self.forecast_horizon) + b
                preds_t.append(pred)
            
            all_preds.append(preds_t)

        # Stack to (N, n_targets) or (N,)
        all_preds = np.array(all_preds).T  # Shape: (N, n_targets)
        
        if len(targets) == 1:
            return all_preds.flatten()
        return all_preds

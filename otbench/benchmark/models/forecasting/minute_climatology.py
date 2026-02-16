# otbench/benchmark/models/forecasting/minute_climatology.py
from typing import Union
import pandas as pd
import numpy as np
from otbench.benchmark.models.forecasting.base_model import BaseForecastingModel

class MinuteClimatologyForecastingModel(BaseForecastingModel):
    """A model that predicts the mean value of the target variable for a given time seen during training."""

    def __init__(self, name: str, target_name: str, window_size: int, forecast_horizon: int, **kwargs):
        super().__init__(name, target_name, window_size, forecast_horizon, **kwargs)
        self.means = {}
        self.global_mean = np.nan

    def _train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Determine the mean value of the target variable seen during training for each time."""
        # Ensure y is DataFrame
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y, index=X.index)
        elif isinstance(y, pd.Series):
            y = y.to_frame()
            
        # Global mean fallback
        self.global_mean = np.nanmean(y.values, axis=0)

        # Time-based mean
        # We need time context. X has the index.
        y_temp = y.copy()
        y_temp["time_of_day"] = X.index.time
        
        # Groupby
        grouped = y_temp.groupby("time_of_day").mean()
        
        # Store as dict of numpy arrays
        self.means = {t: row.values for t, row in grouped.iterrows()}

    def _predict(self, X: 'pd.DataFrame'):
        """Predict the mean seen during training at the time of day of forecast."""
        if len(X) == 0:
            if np.ndim(self.global_mean) > 0:
                return np.empty((0, len(self.global_mean)))
            return np.array([])

        # Calculate forecast times
        times = pd.to_datetime(X.index)
        # Approximate freq if missing? Or just assume index is valid time
        # Applying horizon shift
        # Note: We need a timedelta. If frequency is missing, we might need a workaround.
        # But for Paranal, index has freq or we can infer.
        # Safer: Just take X.index time? 
        # The prompt implies we predict for *forecast horizon*.
        # So we shift time by horizon.
        
        # Attempt to infer shift
        if len(times) > 1:
            dt = times[1] - times[0]
            shift = pd.Timedelta(seconds=self.forecast_horizon * dt.total_seconds())
            target_times = (times + shift).time
        else:
            # Fallback if 1 sample: assume 1 min steps if not known
            target_times = (times + pd.Timedelta(minutes=self.forecast_horizon)).time

        preds = []
        for t in target_times:
            if t in self.means:
                preds.append(self.means[t])
            else:
                preds.append(self.global_mean)

        return np.array(preds)
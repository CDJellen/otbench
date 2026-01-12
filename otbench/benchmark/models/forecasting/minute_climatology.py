from typing import Union

import pandas as pd
import numpy as np

from otbench.benchmark.models.forecasting.base_model import BaseForecastingModel


class MinuteClimatologyForecastingModel(BaseForecastingModel):
    """A model that predicts the mean value of the target variable for a given time seen during training."""

    def __init__(self,
                 name: str,
                 target_name: str,
                 window_size: int,
                 forecast_horizon: int,
                 time_col_name: Union[str, None] = None,
                 **kwargs):
        super().__init__(name, target_name, window_size, forecast_horizon, **kwargs)
        self.time_col_name = time_col_name if time_col_name is not None else None
        self.means = {}
        self.global_mean = np.nan

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Determine the mean value of the target variable seen during training for each time."""
        targets = self.target_name if isinstance(self.target_name, list) else [self.target_name]
        
        # Prepare Data
        X = X.copy()
        X["time_of_day"] = X.index.time
        
        # Calculate Global Means (Fallback)
        self.global_mean = []
        for t in targets:
            cols = [c for c in X.columns if c.startswith(t) and c != "time_of_day"]
            self.global_mean.append(np.nanmean(X[cols].values))
        self.global_mean = np.array(self.global_mean)

        # Calculate Time-Specific Means
        # Structure: self.means[time] = np.array([val_t1, val_t2...])
        self.means = {}
        
        # We process each target separately
        for t_idx, t in enumerate(targets):
            cols = [c for c in X.columns if c.startswith(t) and c != "time_of_day"]
            
            # 1. Collapse lags (mean across columns per row)
            # This gives one value per observation per target
            row_means = X[cols].mean(axis=1)
            
            # 2. Group by time of day
            climatology = row_means.groupby(X["time_of_day"]).mean()
            
            # 3. Store
            for time, val in climatology.items():
                if time not in self.means:
                    self.means[time] = np.zeros(len(targets))
                    # Initialize with global means to handle missing targets for this specific time (rare)
                    # self.means[time] = self.global_mean.copy() # Optional safety
                    
                self.means[time][t_idx] = val

    def predict(self, X: 'pd.DataFrame'):
        """Predict the mean seen during training at the time of day of forecast."""
        times = pd.to_datetime(X.index)

        # apply the forecast horizon to each time
        time_step = times[1] - times[0]
        timedelta = pd.Timedelta(seconds=self.forecast_horizon * time_step.total_seconds())

        # add the timedelta to each time
        times = times + timedelta

        # convert to time
        preds = []
        for time in times.time:
            if time in self.means:
                preds.append(self.means[time])
            else:
                preds.append(self.global_mean)

        preds = np.array(preds)
        
        targets = self.target_name if isinstance(self.target_name, list) else [self.target_name]
        if len(targets) == 1:
            return preds.flatten()
        return preds

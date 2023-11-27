from typing import Union

import pandas as pd
import numpy as np


class BaseForecastingModel:
    """A model that predicts the mean value of the target variable for a given time."""

    def __init__(self, name: str, target_name: str, window_size: int, forecast_horizon: int, **kwargs):
        self.name = name
        self.target_name = target_name
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        # use observations of the past `window_size` to predict the strength of optical turbulence at the forecast horizon
        raise NotImplementedError

    def predict(self, X: 'pd.DataFrame'):
        # predict the strength of optical turbulence at the forecast horizon
        raise NotImplementedError

"""
Offshore Macro-Meteorological Model from:

@article{wang2015prediction,
  title={Prediction model of atmospheric refractive index structure parameter in coastal area},
  author={Wang, Hongxing and Li, Bifeng and Wu, Xiaojun and Liu, Chuanhui and Hu, Zhihui and Xu, Pengfei},
  journal={Journal of Modern Optics},
  volume={62},
  number={16},
  pages={1336--1346},
  year={2015},
  publisher={Taylor \& Francis}
}
"""
import sys
from typing import Union

import numpy as np
import pandas as pd

from otbench.benchmark.models.regression.base_model import BaseRegressionModel
from otbench.utils import apply_fried_height_adjustment, add_temporal_hour, add_temporal_hour_weight


class OffshoreMacroMeteorologicalModel(BaseRegressionModel):
    """A model which predicts the Cn2 value using macro-meteorological parameters, tuned for offshore cases."""

    def __init__(self,
                 name: str,
                 target_name: str,
                 timezone: str,
                 obs_lat: float,
                 obs_lon: float,
                 air_temperature_col_name: str,
                 humidity_col_name: str,
                 wind_speed_col_name: str,
                 time_col_name: str,
                 temporal_hour_col_name: str = "temporal_hour",
                 temporal_hour_weight_col_name: str = "temporal_hour_weight",
                 height_of_observation: Union[float, None] = None,
                 enforce_dynamic_range: bool = True,
                 constant_adjustment: bool = True,
                 use_log10: bool = True,
                 **kwargs):
        super().__init__(name, target_name, **kwargs)
        self.timezone = timezone
        self.obs_lat = obs_lat
        self.obs_lon = obs_lon
        self.air_temperature_col_name = air_temperature_col_name
        self.humidity_col_name = humidity_col_name
        self.wind_speed_col_name = wind_speed_col_name
        self.time_col_name = time_col_name
        self.temporal_hour_col_name = temporal_hour_col_name
        self.temporal_hour_weight_col_name = temporal_hour_weight_col_name
        self.height_of_observation = height_of_observation
        self.enforce_dynamic_range = enforce_dynamic_range
        self.constant_adjustment = constant_adjustment
        self.use_log10 = use_log10

    def train(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series, np.ndarray]):
        """Maintain the same interface as the other models."""
        pass

    def predict(self, X: pd.DataFrame):
        """Generate predictions from the model using an input DataFrame."""

        # add temporal hour and temporal hour weight
        if self.temporal_hour_col_name not in X.columns:
            X = add_temporal_hour(X,
                                  name=self.name,
                                  timezone=self.timezone,
                                  obs_lat=self.obs_lat,
                                  obs_lon=self.obs_lon,
                                  time_col_name=self.time_col_name,
                                  temporal_hour_col_name=self.temporal_hour_col_name)
        if self.temporal_hour_weight_col_name not in X.columns:
            X = add_temporal_hour_weight(X,
                                         temporal_hour_col_name=self.temporal_hour_col_name,
                                         temporal_hour_weight_col_name=self.temporal_hour_weight_col_name)

        X_ = X.loc[X[self.air_temperature_col_name].notna() & X[self.humidity_col_name].notna() &
                   X[self.wind_speed_col_name].notna() & X[self.temporal_hour_weight_col_name].notna()].copy()

        T = X_[self.air_temperature_col_name]  # temperature in [C]
        U = X_[self.wind_speed_col_name]  # Wind Speed in [m/s]
        RH = X_[self.humidity_col_name]  # Relative Humidity [%]
        W = X_[self.temporal_hour_weight_col_name]  # Temporal Hour Weight

        w = -1.58e-15  # coefficient for temporal hour weight W
        t = 2.74e-16  # coefficient for temperature T
        rh = 8.3e-17  # coefficient for relative humidity RH
        rh2 = -2.22e-18  # coefficient for relative humidity squared RH^2
        rh3 = 1.42e-20  # coefficient for relative humidity cubed RH^3
        u = 3.37e-16  # coefficient for wind speed U
        u2 = 1.92e-16  # coefficient for wind speed squared U^2
        u3 = -2.8e-17  # coefficient for wind speed cubed U^3
        c = -7.44e-14  # final coefficient

        omm_prediction = pd.Series((w * W + t * T + rh * RH + rh2 * (RH * RH) + rh3 * (RH * RH * RH) + u * U + u2 *
                                    (U * U) + u3 * (U * U * U) + c),
                                   index=X.index,
                                   name=self.name)

        if self.enforce_dynamic_range:
            X[self.name] = omm_prediction
            X.loc[((X[self.wind_speed_col_name] > 15) | ((X[self.air_temperature_col_name] < -2) |
                                                         (X[self.air_temperature_col_name] > 24)) |
                   (X[self.humidity_col_name] < 15)), self.name] = np.nan
            omm_prediction = X[self.name].values

            X.drop(columns=[self.name], inplace=True)

        if len(omm_prediction[~np.isnan(omm_prediction)]) > 0:
            if self.constant_adjustment & (min(omm_prediction[~np.isnan(omm_prediction)]) <= 0):
                constant_adjustment = min(omm_prediction[~np.isnan(omm_prediction)]) * -1

                omm_prediction = (
                    omm_prediction + \
                    constant_adjustment + \
                    sys.float_info.epsilon  # add epsilon to prevent 0 values
                    )
            else:
                omm_prediction[omm_prediction <= 0] = np.nan

        # convert from model reference height to height of observation
        if self.height_of_observation is not None:
            omm_prediction = apply_fried_height_adjustment(cn2=omm_prediction,
                                                           observed=62.0,
                                                           desired=self.height_of_observation)

        if self.use_log10:
            return np.log10(omm_prediction)
        return omm_prediction

"""
Macro-Meteorological Model from:

@article{sadot1992forecasting,
  title={Forecasting optical turbulence strength on the basis of macroscale meteorology and aerosols: models and validation},
  author={Sadot, Dan and Kopeika, Norman S},
  journal={Optical Engineering},
  volume={31},
  number={2},
  pages={200--212},
  year={1992},
  publisher={SPIE}
}
"""
import sys
from typing import Union

import numpy as np
import pandas as pd

from otbench.benchmark.models.regression.base_model import BaseRegressionModel
from otbench.utils import apply_fried_height_adjustment, add_temporal_hour, add_temporal_hour_weight


class MacroMeteorologicalModel(BaseRegressionModel):
    """A model which predicts the Cn2 value using macro-meteorological parameters."""

    _REFERENCE_HEIGHT: float = 15.0
    _TEMP_KELVIN_OFFSET: float = 273.15
    _COEFFS = (3.8e-14, 2e-15, -2.8e-15, 2.9e-17, -1.1e-19, -2.5e-15, 1.2e-15, -8.5e-17, -5.3e-13)

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
        self._trained_constant_offset = None

    def _out_of_range_mask(self, X: pd.DataFrame) -> pd.Series:
        """Identify rows outside the empirical validity envelope of the polynomial."""
        return ((X[self.wind_speed_col_name] > 10) | ((X[self.air_temperature_col_name] < 9) |
                                                      (X[self.air_temperature_col_name] > 35)) |
                ((X[self.humidity_col_name] < 14) | (X[self.humidity_col_name] > 92)))

    def _compute_raw_prediction(self, X: pd.DataFrame) -> np.ndarray:
        """Compute unshifted macro-meteorological polynomial predictions without mutating X."""
        X = X.copy()
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

        T = X_[self.air_temperature_col_name] + self._TEMP_KELVIN_OFFSET
        U = X_[self.wind_speed_col_name]
        RH = X_[self.humidity_col_name]
        W = X_[self.temporal_hour_weight_col_name]

        w, t, rh, rh2, rh3, u, u2, u3, c = self._COEFFS

        mm_series = pd.Series(
            (w * W + t * T + rh * RH + rh2 * (RH * RH) + rh3 * (RH * RH * RH) + u * U + u2 * (U * U) + u3 *
             (U * U * U) + c),
            index=X.index,
            name=self.name,
            dtype=float,
        )

        if self.enforce_dynamic_range:
            mm_series.loc[self._out_of_range_mask(X)] = np.nan

        return mm_series.to_numpy(dtype=float, copy=True)

    def train(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series, np.ndarray]):
        """Record the training-set constant adjustment offset so single-row inference is consistent."""
        if X is not None and len(X) > 0:
            raw = self._compute_raw_prediction(X)
            valid = raw[~np.isnan(raw)]
            if len(valid) > 0 and np.min(valid) <= 0:
                self._trained_constant_offset = float(-np.min(valid))
            else:
                self._trained_constant_offset = 0.0

    def predict(self, X: pd.DataFrame):
        """Generate predictions from the model using an input DataFrame."""
        mm_prediction = self._compute_raw_prediction(X)
        valid = mm_prediction[~np.isnan(mm_prediction)]

        if len(valid) > 0:
            batch_min = float(np.min(valid))
            if self.constant_adjustment:
                offset = (self._trained_constant_offset if self._trained_constant_offset is not None else
                          (-batch_min if batch_min <= 0 else 0.0))
                if offset > 0:
                    mm_prediction = mm_prediction + offset + sys.float_info.epsilon
                elif batch_min <= 0:
                    mm_prediction = mm_prediction + (-batch_min) + sys.float_info.epsilon
                finite_nonpos = ~np.isnan(mm_prediction) & (mm_prediction <= 0)
                if np.any(finite_nonpos):
                    mm_prediction[finite_nonpos] = sys.float_info.epsilon
            else:
                mm_prediction[~np.isnan(mm_prediction) & (mm_prediction <= 0)] = np.nan

        # convert from model reference height to height of observation
        if self.height_of_observation is not None:
            mm_prediction = apply_fried_height_adjustment(cn2=mm_prediction,
                                                          observed=self._REFERENCE_HEIGHT,
                                                          desired=self.height_of_observation)

        if self.use_log10:
            return np.log10(mm_prediction)
        return mm_prediction

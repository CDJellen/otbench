from typing import Union

import pandas as pd
import numpy as np
import lightgbm as lgb

from otbench.benchmark.models.regression.base_model import BaseRegressionModel
from otbench.benchmark.models.regression.air_water_temperature_difference import AWTModel


class HybridAWTRegressionModel(BaseRegressionModel):
    """A model that uses a hybrid of the AWTModel and a random forest regressor to predict the target."""

    # Keys consumed by the framework or AWTModel that must NOT be forwarded to LGBMRegressor.
    _NON_LGBM_KEYS = frozenset({
        "verbose", "predict_residuals", "use_log10",
        "input_size", "output_size", "in_channels",
        "timezone", "obs_lat", "obs_lon", "obs_tz",
        "air_temperature_col_name", "water_temperature_col_name",
        "humidity_col_name", "wind_speed_col_name", "time_col_name",
        "height_of_observation", "enforce_dynamic_range", "constant_adjustment",
        "session_col", "d_model", "nhead", "num_layers", "dropout",
        "batch_size", "n_epochs", "learning_rate", "hidden_size",
        "window_size", "forecast_horizon",
    })

    def __init__(self, name: str, target_name: str, time_col_name: Union[str, None] = None, **kwargs):
        super().__init__(name, target_name, **kwargs)
        self.time_col_name = time_col_name if time_col_name is not None else None
        self._awt_mdl = AWTModel(name=name, target_name=target_name, **kwargs)
        lgbm_kwargs = {k: v for k, v in kwargs.items() if k not in self._NON_LGBM_KEYS}
        self._lgb_mdl = lgb.LGBMRegressor(verbose=-1, **lgbm_kwargs)

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Train the underlying LightGBM regressor to form a hybrid model."""
        y_pred_awt = self._awt_mdl.predict(X)
        target = y[self.target_name].values - y_pred_awt.values
        target[np.isnan(y_pred_awt.values)] = np.nan
        # Drop rows where the AWT model produced NaN (missing temperature inputs).
        # LightGBM rejects NaN in y.
        valid = ~np.isnan(target)
        if not valid.any():
            return
        self._lgb_mdl.fit(X[valid], target[valid])

    def predict(self, X: 'pd.DataFrame'):
        """Use generate AWT predictions and correct with trained LightGBM regressor."""
        y_pred_awt = self._awt_mdl.predict(X)
        hybrid_correction = self._lgb_mdl.predict(X)
        preds = y_pred_awt.values + hybrid_correction
        preds[np.isnan(y_pred_awt.values)] = np.nan
        return preds

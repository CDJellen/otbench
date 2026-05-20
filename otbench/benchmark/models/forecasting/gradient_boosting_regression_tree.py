# otbench/benchmark/models/forecasting/gradient_boosting_regression_tree.py
from typing import Union
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from otbench.benchmark.models.forecasting.base_model import BaseForecastingModel

class GradientBoostingForecastingModel(BaseForecastingModel):
    """A model that uses gradient boosting regression trees for direct forecasting."""

    # Keys consumed by the framework (BaseForecastingModel, bench_runner, notebooks)
    # that must NOT be forwarded to LGBMRegressor.
    _NON_LGBM_KEYS = frozenset({
        "verbose", "predict_residuals", "use_log10",
        "input_size", "output_size", "in_channels",
        "timezone", "obs_lat", "obs_lon", "obs_tz",
        "air_temperature_col_name", "water_temperature_col_name",
        "humidity_col_name", "wind_speed_col_name", "time_col_name",
        "height_of_observation", "enforce_dynamic_range", "constant_adjustment",
        "session_col", "d_model", "nhead", "num_layers", "dropout",
        "batch_size", "n_epochs", "learning_rate", "hidden_size",
    })

    def __init__(self, name: str, target_name: str, window_size: int, forecast_horizon: int, output_size: int = 1, **kwargs):
        super().__init__(name, target_name, window_size, forecast_horizon, **kwargs)

        # Filter to only LGBM-compatible parameters
        lgbm_kwargs = {k: v for k, v in kwargs.items() if k not in self._NON_LGBM_KEYS}
        lgb_estimator = lgb.LGBMRegressor(verbose=-1, **lgbm_kwargs)

        # Handle Vector vs Scalar
        if output_size > 1:
            self._lgb_mdl = MultiOutputRegressor(lgb_estimator)
        else:
            self._lgb_mdl = lgb_estimator

    def _train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Internal training implementation."""
        if len(X) == 0:
            return

        # Drop rows where any target is NaN.  LightGBM handles NaN in X natively
        # but rejects NaN in y; filter here defensively in case the caller skips
        # the standard forecasting pipeline's dropna step.
        if isinstance(y, pd.DataFrame):
            valid = y.notna().all(axis=1).values
        elif isinstance(y, pd.Series):
            valid = y.notna().values
        else:
            y_arr = np.asarray(y)
            valid = (~np.isnan(y_arr).any(axis=1) if y_arr.ndim > 1
                     else ~np.isnan(y_arr))
        if not valid.all():
            X = X[valid]
            y = y[valid] if not isinstance(y, np.ndarray) else y[valid]

        if len(X) == 0:
            return

        self._lgb_mdl.fit(X, y)

    def _predict(self, X: 'pd.DataFrame'):
        """Internal prediction implementation."""
        # Guard against empty prediction data
        if len(X) == 0:
            # Return empty array with correct dimensions
            if isinstance(self._lgb_mdl, MultiOutputRegressor):
                # We can't ask the model for output dim easily if not fitted, 
                # so we rely on the target_name list length if available or fallback
                n_targets = len(self.target_name) if isinstance(self.target_name, list) else 1
                return np.empty((0, n_targets))
            return np.array([])

        return self._lgb_mdl.predict(X)
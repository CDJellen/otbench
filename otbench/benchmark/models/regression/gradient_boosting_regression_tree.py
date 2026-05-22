from typing import Union

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor

from otbench.benchmark.models.regression.base_model import BaseRegressionModel


class GradientBoostingRegressionModel(BaseRegressionModel):
    """A model that uses gradient boosting regression trees to predict the target."""

    # Keys consumed by the framework (BaseRegressionModel, bench_runner, notebooks)
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
        "window_size", "forecast_horizon",
    })

    def __init__(self, name: str, target_name: str, time_col_name: Union[str, None] = None, output_size: int = 1, **kwargs):
        super().__init__(name, target_name, **kwargs)
        self.time_col_name = time_col_name
        self.output_size = output_size

        # Filter to only LGBM-compatible parameters
        lgbm_kwargs = {k: v for k, v in kwargs.items() if k not in self._NON_LGBM_KEYS}
        lgb_estimator = lgb.LGBMRegressor(verbose=-1, **lgbm_kwargs)

        # Handle Vector vs Scalar
        if output_size > 1:
            self._lgb_mdl = MultiOutputRegressor(lgb_estimator)
        else:
            self._lgb_mdl = lgb_estimator

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Train the underlying LightGBM regressor."""
        if len(X) == 0:
            return

        # Drop rows where any target is NaN.  LightGBM handles NaN in X
        # natively (missing-value splitting) but rejects NaN in y.  Filtering
        # here makes train() safe to call directly from notebooks without
        # relying on bench_runner's upstream y-mask.
        if isinstance(y, pd.DataFrame):
            valid = y.notna().all(axis=1)
        elif isinstance(y, pd.Series):
            valid = y.notna()
        else:
            y_arr = np.asarray(y)
            valid = (~np.isnan(y_arr).any(axis=1) if y_arr.ndim > 1
                     else ~np.isnan(y_arr))
        if not valid.all():
            X = X[valid]
            y = y[valid]

        if len(X) == 0:
            return

        self._lgb_mdl.fit(X, y)

    def predict(self, X: 'pd.DataFrame'):
        """Use the underlying LightGBM regressor to generate predictions."""
        if len(X) == 0:
            # Return empty array matching the shape of non-empty predictions:
            # scalar output → (0,); vector output → (0, n_targets)
            if self.output_size > 1:
                return np.empty((0, self.output_size))
            return np.empty((0,))

        return self._lgb_mdl.predict(X)

import inspect
from typing import Any, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

# Framework and cross-model keys that should not override LGBMRegressor defaults.
_NON_LGBM_KEYS = frozenset({
    "verbose",
    "predict_residuals",
    "use_log10",
    "input_size",
    "output_size",
    "in_channels",
    "timezone",
    "obs_lat",
    "obs_lon",
    "obs_tz",
    "air_temperature_col_name",
    "water_temperature_col_name",
    "humidity_col_name",
    "wind_speed_col_name",
    "time_col_name",
    "height_of_observation",
    "enforce_dynamic_range",
    "constant_adjustment",
    "session_col",
    "feature_names",
    "d_model",
    "nhead",
    "num_layers",
    "dropout",
    "batch_size",
    "n_epochs",
    # Excluded because bench_runner broadcasts PyTorch's learning_rate across shared model_kwargs;
    # LightGBM's learning rate can still be configured via the `eta` alias when needed.
    "learning_rate",
    "hidden_size",
    "window_size",
    "forecast_horizon",
    "weight_decay",
    "patience",
    "val_fraction",
    "use_scheduler",
    "normalize_data",
    "cell_type",
    "optimizer",
    "criterion",
    "use_physics_loss",
    "heights",
    "lambda_r0",
    "lambda_theta0",
    "jensen_correction",
})

_LGBM_INIT_PARAMS = frozenset(
    name for name, p in inspect.signature(lgb.LGBMRegressor.__init__).parameters.items()
    if name not in ("self", "verbose") and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY))

_LGBM_BOOSTER_ALIASES = frozenset({
    "eta",
    "feature_fraction",
    "bagging_fraction",
    "bagging_freq",
    "lambda_l1",
    "lambda_l2",
    "min_data_in_leaf",
    "min_sum_hessian_in_leaf",
    "max_bin",
    "seed",
    "deterministic",
    "force_col_wise",
    "force_row_wise",
    "extra_trees",
    "path_smooth",
    "min_gain_to_split",
    "cat_smooth",
    "cat_l2",
    "max_cat_to_onehot",
    "top_rate",
    "other_rate",
    "monotone_constraints",
    "interaction_constraints",
    "linear_tree",
    "tweedie_variance_power",
    "alpha",
    "fair_c",
})

_LGBM_VALID_PARAMS = (_LGBM_INIT_PARAMS | _LGBM_BOOSTER_ALIASES) - _NON_LGBM_KEYS


def filter_lgbm_kwargs(kwargs: dict) -> dict:
    """Filter a keyword dictionary to valid LightGBM parameters, excluding framework keys."""
    return {k: v for k, v in kwargs.items() if k in _LGBM_VALID_PARAMS}


def build_lgbm_estimator(output_size: int, target_name: Any, kwargs: dict) -> Tuple[Any, int]:
    """Construct a scalar LGBMRegressor or vector MultiOutputRegressor(LGBMRegressor)."""
    if output_size == 1 and isinstance(target_name, (list, tuple)) and len(target_name) > 1:
        output_size = len(target_name)
    lgb_estimator = lgb.LGBMRegressor(verbose=-1, **filter_lgbm_kwargs(kwargs))
    if output_size > 1:
        return MultiOutputRegressor(lgb_estimator), output_size
    return lgb_estimator, output_size


def filter_valid_target_rows(
    X: pd.DataFrame,
    y: Union[pd.DataFrame, pd.Series, np.ndarray],
) -> Tuple[pd.DataFrame, Union[pd.DataFrame, pd.Series, np.ndarray]]:
    """Drop rows where any target column in y is NaN while preserving feature NaNs in X."""
    if isinstance(y, pd.DataFrame):
        valid = y.notna().all(axis=1).to_numpy()
    elif isinstance(y, pd.Series):
        valid = y.notna().to_numpy()
    else:
        y_arr = np.asarray(y)
        valid = ~np.isnan(y_arr).any(axis=1) if y_arr.ndim > 1 else ~np.isnan(y_arr)
    if not valid.all():
        return X[valid], y[valid]
    return X, y

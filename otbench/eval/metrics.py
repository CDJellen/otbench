from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import sklearn.metrics as sk_m
from scipy.stats import linregress

__all__ = ["is_implemented_metric", "coefficient_of_determination", "root_mean_square_error", "mean_absolute_error", "mean_absolute_percentage_error"]


def is_implemented_metric(metric_name: str) -> bool:
    """Check that the metric is implemented"""
    if metric_name in __all__:
        return True
    return False


def coefficient_of_determination(y_true: Sequence, y_pred: Sequence, detailed: bool = False) -> dict:
    """Calculates R2 score using `sklearn.metrics.r2_score`."""
    y_true, y_pred = _get_valid_indices(y_true=y_true, y_pred=y_pred)
    if len(y_pred) == 0:
        return _format_metric(np.nan, 0)
    
    if detailed:
        r2_raw = sk_m.r2_score(y_true, y_pred, multioutput="raw_values")
        r2_avg = np.mean(r2_raw)
        res = _format_metric(float(r2_avg), len(y_pred))
        res["detailed_score"] = r2_raw.tolist()
        return res
        
    r2 = sk_m.r2_score(y_true, y_pred, multioutput="uniform_average")
    return _format_metric(float(r2), len(y_pred))


def root_mean_square_error(y_true: Sequence, y_pred: Sequence, detailed: bool = False) -> dict:
    """Calculate RMSE from `sklearn.metrics.mean_squared_error`."""
    y_true, y_pred = _get_valid_indices(y_true=y_true, y_pred=y_pred)
    if len(y_pred) == 0:
        return _format_metric(np.nan, 0)
    
    if detailed:
        rmse_raw = sk_m.mean_squared_error(y_true, y_pred, squared=False, multioutput="raw_values")
        rmse_avg = np.mean(rmse_raw)
        res = _format_metric(float(rmse_avg), len(y_pred))
        res["detailed_score"] = rmse_raw.tolist()
        return res

    return _format_metric(float(sk_m.mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False, multioutput="uniform_average")), len(y_pred))


def mean_absolute_error(y_true: Sequence, y_pred: Sequence, detailed: bool = False) -> dict:
    """An alias for `sklearn.metrics.mean_absolute_error`."""
    y_true, y_pred = _get_valid_indices(y_true=y_true, y_pred=y_pred)
    if len(y_pred) == 0:
        return _format_metric(np.nan, 0)
    
    if detailed:
        mae_raw = sk_m.mean_absolute_error(y_true, y_pred, multioutput="raw_values")
        mae_avg = np.mean(mae_raw)
        res = _format_metric(float(mae_avg), len(y_pred))
        res["detailed_score"] = mae_raw.tolist()
        return res

    return _format_metric(float(sk_m.mean_absolute_error(y_true=y_true, y_pred=y_pred, multioutput="uniform_average")), len(y_pred))


def mean_absolute_percentage_error(y_true: Sequence, y_pred: Sequence, detailed: bool = False) -> dict:
    """An alias for `sklearn.metrics.mean_absolute_percentage_error`."""
    y_true, y_pred = _get_valid_indices(y_true=y_true, y_pred=y_pred)
    if len(y_pred) == 0:
        return _format_metric(np.nan, 0)
    
    if detailed:
        mape_raw = sk_m.mean_absolute_percentage_error(y_true, y_pred, multioutput="raw_values")
        mape_avg = np.mean(mape_raw)
        res = _format_metric(float(mape_avg), len(y_pred))
        res["detailed_score"] = mape_raw.tolist()
        return res

    return _format_metric(float(sk_m.mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred, multioutput="uniform_average")), len(y_pred))


from .utils import _get_valid_indices, _format_metric

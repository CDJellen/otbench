from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import sklearn.metrics as sk_m
from scipy.stats import linregress

__all__ = ["coefficient_of_determination", "root_mean_square_error", "mean_absolute_error", "mean_absolute_percentage_error"]


def is_implemented_metric(metric_name: str) -> bool:
    """Check that the metric is implemented"""
    if metric_name in __all__:
        return True
    return False


def coefficient_of_determination(y_true: Sequence, y_pred: Sequence) -> Tuple[float, int]:
    """Calculates R2 score using `sklearn.metrics.r2_score`."""
    y_true, y_pred = _get_valid_indices(y_true=y_true, y_pred=y_pred)
    if len(y_pred) == 0:
        return _format_metric(np.nan, 0)
    r2 = sk_m.r2_score(y_true, y_pred, multioutput="uniform_average")
    return _format_metric(float(r2), len(y_pred))


def root_mean_square_error(y_true: Sequence, y_pred: Sequence) -> Tuple[float, int]:
    """Calculate RMSE from `sklearn.metrics.mean_squared_error`."""
    y_true, y_pred = _get_valid_indices(y_true=y_true, y_pred=y_pred)
    if len(y_pred) == 0:
        return _format_metric(np.nan, 0)
    return _format_metric(float(sk_m.mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False, multioutput="uniform_average")), len(y_pred))


def mean_absolute_error(y_true: Sequence, y_pred: Sequence) -> Tuple[float, int]:
    """An alias for `sklearn.metrics.mean_absolute_error`."""
    y_true, y_pred = _get_valid_indices(y_true=y_true, y_pred=y_pred)
    if len(y_pred) == 0:
        return _format_metric(np.nan, 0)
    return _format_metric(float(sk_m.mean_absolute_error(y_true=y_true, y_pred=y_pred, multioutput="uniform_average")), len(y_pred))


def mean_absolute_percentage_error(y_true: Sequence, y_pred: Sequence) -> Tuple[float, int]:
    """An alias for `sklearn.metrics.mean_absolute_percentage_error`."""
    y_true, y_pred = _get_valid_indices(y_true=y_true, y_pred=y_pred)
    if len(y_pred) == 0:
        return _format_metric(np.nan, 0)
    return _format_metric(float(sk_m.mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred, multioutput="uniform_average")), len(y_pred))


def _format_metric(metric_value: float, valid_predictions: int) -> dict:
    """Format the metric value and valid predictions into a dict."""
    return {"metric_value": metric_value, "valid_predictions": valid_predictions}


def _get_valid_indices(y_true: Sequence, y_pred: Sequence) -> Tuple[Sequence, Sequence]:
    """Get the valid indices for the supplied sequences."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must have the same length, got {len(y_true)} and {len(y_pred)}")

    y_true = y_true.to_numpy().squeeze()

    # ensure we have numpy arrays in y_pred
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.to_numpy().squeeze()
    else:
        y_pred = np.array(y_pred).squeeze()

    # handle 1D case (scalar target)
    if y_true.ndim == 1:
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    # handle 2D case (vector target)
    else:
        # Check if any value in the row is nan
        mask = ~np.isnan(y_true).all(axis=1) & ~np.isnan(y_pred).all(axis=1)

    return y_true[mask], y_pred[mask]

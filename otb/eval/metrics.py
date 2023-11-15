from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import sklearn.metrics as sk_m

__all__ = ["r2_score", "root_mean_square_error", "mean_absolute_error", "mean_absolute_percentage_error"]


def is_implemented_metric(metric_name: str) -> bool:
    """Check that the metric is implemented"""
    if metric_name in __all__:
        return True
    return False


def r2_score(y_true: Sequence, y_pred: Sequence) -> Tuple[float, int]:
    """An alias for `sklearn.metrics.r2_score`."""
    y_true, y_pred = _get_valid_indices(y_true=y_true, y_pred=y_pred)
    return _format_metric(float(sk_m.r2_score(y_true=y_true, y_pred=y_pred)), len(y_pred))


def root_mean_square_error(y_true: Sequence, y_pred: Sequence) -> Tuple[float, int]:
    """Calculate RMSE from `sklearn.metrics.mean_squared_error`."""
    y_true, y_pred = _get_valid_indices(y_true=y_true, y_pred=y_pred)
    return _format_metric(float(sk_m.mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)), len(y_pred))


def mean_absolute_error(y_true: Sequence, y_pred: Sequence) -> Tuple[float, int]:
    """An alias for `sklearn.metrics.r2_score`."""
    y_true, y_pred = _get_valid_indices(y_true=y_true, y_pred=y_pred)
    return _format_metric(float(sk_m.mean_absolute_error(y_true=y_true, y_pred=y_pred)), len(y_pred))


def mean_absolute_percentage_error(y_true: Sequence, y_pred: Sequence) -> Tuple[float, int]:
    """An alias for `sklearn.metrics.r2_score`."""
    y_true, y_pred = _get_valid_indices(y_true=y_true, y_pred=y_pred)
    return _format_metric(float(sk_m.mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)), len(y_pred))


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

    return y_true[~np.isnan(y_true) & ~np.isnan(y_pred)], y_pred[~np.isnan(y_true) & ~np.isnan(y_pred)]

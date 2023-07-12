from typing import Any, Sequence

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


__all__ = ["r2_score", "root_mean_square_error", "mean_absolute_error", "mean_absolute_percentage_error", "symetric_mean_absolute_percentage_error"]


def is_implemented_metric(metric_name: str) -> bool:
    """Check that the metric is implemented"""
    if metric_name in __all__: return True
    return False

def r2_score(y_true: Sequence, y_pred: Sequence) -> float:
    """An alias for `sklearn.metrics.r2_score`."""
    return r2_score(y_true=y_true, y_pred=y_pred)


def root_mean_square_error(y_true: Sequence, y_pred: Sequence) -> float:
    """Calculate RMSE from `sklearn.metrics.mean_squared_error`."""
    return mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)


def mean_absolute_error(y_true: Sequence, y_pred: Sequence) -> float:
    """An alias for `sklearn.metrics.r2_score`."""
    return mean_absolute_error(y_true=y_true, y_pred=y_pred)


def mean_absolute_percentage_error(y_true: Sequence, y_pred: Sequence) -> float:
    """An alias for `sklearn.metrics.r2_score`."""
    return mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)


def symetric_mean_absolute_percentage_error(y_true: Sequence, y_pred: Sequence) -> float:
    """Calculate the symmetric mean absolute percentage error"""
    smape = 100/len(y_true) * sum(2 * abs(y_pred - y_true) / (abs(y_true) + abs(y_pred)))

    return smape

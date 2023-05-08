from typing import Any

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


def r2_score(y_true: Any, y_pred: Any):
    """An alias for `sklearn.metrics.r2_score`."""
    return r2_score(y_true=y_true, y_pred=y_pred)


def root_mean_square_error(y_true, y_pred):
    """Calculate RMSE from `sklearn.metrics.mean_squared_error`."""
    return mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)


def mean_absolute_error(y_true, y_pred):
    """An alias for `sklearn.metrics.r2_score`."""
    return mean_absolute_error(y_true=y_true, y_pred=y_pred)


def mean_absolute_percentage_error(y_true, y_pred):
    """An alias for `sklearn.metrics.r2_score`."""
    return mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)


def symetric_mean_absolute_percentage_error(y_true, y_pred):
    """Calculate the symmetric mean absolute percentage error"""
    smape = 100/len(y_true) * sum(2 * abs(y_pred - y_true) / (abs(y_true) + abs(y_pred)))

    return smape

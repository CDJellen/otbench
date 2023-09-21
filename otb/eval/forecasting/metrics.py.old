from typing import Tuple, Sequence

import numpy as np

__all__ = ["symmetric_mean_absolute_percentage_error"]


def symmetric_mean_absolute_percentage_error(y_true: Sequence[Sequence], y_pred: Sequence[Sequence]) -> Tuple[float, Sequence[float]]:
    """Calculate the symmetric mean absolute percentage error"""
    smape = []
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
    for i in range(len(y_true)):
        cur_true, cur_pred = y_true[i], y_pred[i]
        if len(cur_true) != len(cur_pred):
            raise ValueError("each forecast window prediction in y_true and y_pred must have the same length")
        smape.append(100/len(cur_true) * sum(2 * abs(cur_pred - cur_true) / (abs(cur_true) + abs(cur_pred))))

    smape = np.array(smape)
    return smape.mean(), smape

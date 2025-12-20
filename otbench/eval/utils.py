
import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Union

def _format_metric(metric_value: float, valid_predictions: int) -> dict:
    """Format the metric value and valid predictions into a dict."""
    return {"metric_value": metric_value, "valid_predictions": valid_predictions}


def _get_valid_indices(y_true: Sequence, y_pred: Sequence) -> Tuple[Sequence, Sequence]:
    """Get the valid indices for the supplied sequences."""
    # Ensure inputs are array-like if they are sequences but not numpy/pandas (e.g. lists)
    if not isinstance(y_true, (np.ndarray, pd.Series, pd.DataFrame)):
        y_true = np.array(y_true)
        
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must have the same length, got {len(y_true)} and {len(y_pred)}")

    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        y_true = y_true.to_numpy()
    
    # Only squeeze if it's (N, 1) to become (N,)
    # Do not squeeze (1, M) as that destroys the sample dimension for vectors
    if y_true.ndim == 2 and y_true.shape[1] == 1:
        y_true = y_true.ravel()

    # ensure we have numpy arrays in y_pred
    if isinstance(y_pred, (pd.Series, pd.DataFrame)):
        y_pred = y_pred.to_numpy()
    else:
        y_pred = np.array(y_pred)
        
    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.ravel()

    # handle 1D case (scalar target)
    if y_true.ndim == 1:
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    # handle 2D case (vector target)
    else:
        # Check if any value in the row is nan
        mask = ~np.isnan(y_true).all(axis=1) & ~np.isnan(y_pred).all(axis=1)

    return y_true[mask], y_pred[mask]

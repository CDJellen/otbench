import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Union

def _format_metric(metric_value: float, valid_predictions: int) -> dict:
    """Format the metric value and valid predictions into a dict."""
    return {"metric_value": metric_value, "valid_predictions": valid_predictions}


def _get_valid_indices(y_true: Sequence, y_pred: Sequence) -> Tuple[Sequence, Sequence]:
    """
    Get the valid indices for the supplied sequences, handling scalar and vector targets robustly.
    
    This function:
    - Converts inputs to numpy arrays
    - Normalizes (N, 1) shapes to (N,) for scalar targets
    - Ensures y_true and y_pred have compatible shapes
    - Creates a mask excluding rows where any (for scalar) or all (for vector) values are NaN
    - Returns masked arrays of identical shape
    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Normalise common (N, 1) -> (N,) cases
    if y_true.ndim == 2 and y_true.shape[1] == 1:
        y_true = y_true.ravel()
    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.ravel()

    # After normalisation, shapes must match
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have compatible shapes after normalisation. "
            f"Got y_true.shape = {y_true.shape} and y_pred.shape = {y_pred.shape}"
        )

    # Determine masking strategy based on ground truth dimensionality
    if y_true.ndim == 1:
        # Scalar target: drop samples where either value is NaN
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    elif y_true.ndim == 2:
        # Vector target: drop samples where *all* values in the row are NaN
        # (i.e. keep samples that have at least one valid value in true and pred)
        mask = ~np.isnan(y_true).all(axis=1) & ~np.isnan(y_pred).all(axis=1)
    else:
        raise ValueError(f"Unsupported dimensionality for y_true: ndim={y_true.ndim}. "
                         f"Expected 1 (scalar) or 2 (vector) dimensions.")

    return y_true[mask], y_pred[mask]

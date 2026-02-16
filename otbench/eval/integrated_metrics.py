import numpy as np
from typing import Sequence, Optional, Union, Tuple
from .utils import _get_valid_indices, _format_metric


def integrated_seeing(
    y_true: Sequence,
    y_pred: Sequence,
    heights: Optional[Sequence[float]] = None,
    wavelength: float = 500e-9,
    detailed: bool = False,
) -> dict:
    """
    Calculates the Integrated Seeing (Fried parameter derived FWHM) from a Cn2 profile.
    
    Formula:
        r0 = [0.423 * (2*pi/lambda)^2 * integral(Cn2(h) dh)]^(-3/5)
        seeing = 0.98 * lambda / r0
        
    Args:
        y_true: True Cn2 profile (samples x layers)
        y_pred: Predicted Cn2 profile (samples x layers)
        heights: Array of heights in meters corresponding to the layers. 
                 If None, assumes uniform unit layers (simple summation).
        wavelength: Wavelength in meters (default 500nm)
        detailed: If True, returns seeing per sample.
        
    Returns:
        Dictionary containing 'metric_value' (RMSE of seeing) and 'valid_predictions'.
        Note: The metric reported is the RMSE between the *Derived True Seeing* and 
        *Derived Predicted Seeing*, NOT the seeing value itself.
    """
    y_true, y_pred = _get_valid_indices(y_true, y_pred)
    if len(y_pred) == 0:
        return _format_metric(np.nan, 0)

    # helper to calc seeing from profile
    def calc_seeing(profile, h):
        # inputs: profile (N, layers), h (layers,)
        k = 2 * np.pi / wavelength



        # Avoid zero or negative Cn2
        profile = np.maximum(profile, 1e-19)

        if h is not None:
            # Integrate Cn2 * dh
            # If heights provided, use trapz or sum with deltas
            J = np.trapz(profile, x=h, axis=1)
        else:
            J = np.sum(profile, axis=1)  # Treat as sum of layers

        # Avoid zero or negative J
        J = np.clip(J, 1e-30, None)

        r0 = (0.423 * k**2 * J)**(-3 / 5)
        epsilon = 0.98 * wavelength / r0
        
        # Convert to arcseconds
        epsilon_asec = epsilon * 206265.0
        
        return epsilon_asec

    # Convert heights to numpy if present
    h_arr = np.array(heights) if heights is not None else None

    # Calculate seeing for true ("DIMM" equivalent) and pred
    seeing_true = calc_seeing(y_true, h_arr)
    seeing_pred = calc_seeing(y_pred, h_arr)

    # RMSE of seeing
    error = seeing_true - seeing_pred
    rmse_seeing = np.sqrt(np.mean(error**2))

    res = _format_metric(float(rmse_seeing), len(y_pred))

    if detailed:
        # Return the per-sample errors
        res["detailed_score"] = error.tolist()
        res["seeing_true"] = seeing_true.tolist()
        res["seeing_pred"] = seeing_pred.tolist()

    return res

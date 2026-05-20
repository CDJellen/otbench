import numpy as np
from typing import Sequence, Optional, Union, Tuple
from .utils import _get_valid_indices, _format_metric

# np.trapezoid was introduced in NumPy 2.0; fall back to np.trapz for older installs.
try:
    _trapezoid = np.trapezoid
except AttributeError:
    _trapezoid = np.trapz


def integrated_seeing(
    y_true: Sequence,
    y_pred: Sequence,
    heights: Optional[Sequence[float]] = None,
    wavelength: float = 500e-9,
    detailed: bool = False,
    values_are_integrals: bool = True,
) -> dict:
    """
    Calculates the Integrated Seeing (Fried parameter derived FWHM) from a turbulence profile.

    Formula:
        r0 = [0.423 * (2*pi/lambda)^2 * J_total]^(-3/5)
        seeing = 0.98 * lambda / r0

    The input profile may contain either:
      - Layer-integrated turbulence strengths J_i (units: m^{1/3}), as provided by
        MASS restoration. In this case the total integral is simply sum(J_i).
      - C_n^2 density values (units: m^{-2/3}), requiring numerical integration
        over height via trapz.

    Args:
        y_true: True profile (samples x layers).
        y_pred: Predicted profile (samples x layers).
        heights: Array of heights in meters corresponding to the layers.
                 Only used for numerical integration when values_are_integrals=False.
        wavelength: Wavelength in meters (default 500 nm).
        detailed: If True, returns seeing per sample.
        values_are_integrals: If True (default), the profile values are already
            layer-integrated J_i (m^{1/3}) and are simply summed. If False,
            the values are C_n^2 density (m^{-2/3}) and are integrated with
            np.trapz using the provided heights.

    Returns:
        Dictionary containing 'metric_value' (RMSE of seeing in arcseconds) and
        'valid_predictions'.  When ``detailed=True``, also contains:
          - 'detailed_score': per-sample signed errors (seeing_true - seeing_pred) in arcseconds
          - 'seeing_true': per-sample integrated seeing derived from y_true
          - 'seeing_pred': per-sample integrated seeing derived from y_pred
    """
    y_true, y_pred = _get_valid_indices(y_true, y_pred)
    if len(y_pred) == 0:
        return _format_metric(np.nan, 0)

    def calc_seeing(profile, h):
        k = 2 * np.pi / wavelength

        # Avoid zero or negative values
        profile = np.maximum(profile, 1e-19)

        if values_are_integrals:
            # Values are J_i (layer integrals in m^{1/3}): just sum
            J = np.sum(profile, axis=1)
        elif h is not None:
            # Values are C_n^2 density: integrate over height
            J = _trapezoid(profile, x=h, axis=1)
        else:
            raise ValueError(
                "heights must be provided when values_are_integrals=False. "
                "C_n^2 density values require numerical integration over height."
            )

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

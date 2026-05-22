import numpy as np
from typing import Sequence, Optional
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

    # Detect and reverse log10 transform.
    # Turbulence integrals J_i are O(1e-15), so log10(J_i) ~ -15.
    # If the median is negative, the inputs are in log10 space and must
    # be exponentiated before physical integration (sum or trapz).
    if values_are_integrals and np.nanmedian(y_true) < 0:
        y_true = np.power(10.0, y_true)
        y_pred = np.power(10.0, y_pred)

    def calc_seeing(profile, h):
        _, r0 = _profile_to_j_and_r0(profile, h, wavelength, values_are_integrals)
        epsilon = 0.98 * wavelength / r0
        # Convert to arcseconds
        return epsilon * 206265.0

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


def _profile_to_j_and_r0(
    profile: np.ndarray,
    heights: Optional[np.ndarray],
    wavelength: float,
    values_are_integrals: bool,
) -> tuple:
    """Shared helper: profile -> (J_total, r0) per sample."""
    k = 2 * np.pi / wavelength
    profile = np.maximum(profile, 1e-19)

    if values_are_integrals:
        J = np.sum(profile, axis=1)
    elif heights is not None:
        J = _trapezoid(profile, x=heights, axis=1)
    else:
        raise ValueError(
            "heights must be provided when values_are_integrals=False."
        )

    J = np.clip(J, 1e-30, None)
    r0 = (0.423 * k**2 * J) ** (-3 / 5)
    return J, r0


def _effective_height(
    profile: np.ndarray,
    heights: np.ndarray,
    values_are_integrals: bool,
) -> np.ndarray:
    """Turbulence-weighted effective height H_eff per sample.

    H_eff = [ sum( J_i * h_i^(5/3) ) / sum( J_i ) ]^(3/5)
    """
    profile = np.maximum(profile, 1e-19)
    h = np.asarray(heights, dtype=float)

    if values_are_integrals:
        weights = profile  # J_i per layer
    else:
        raise NotImplementedError(
            "Effective height from C_n^2 density requires layer boundaries."
        )

    h_53 = np.power(h, 5.0 / 3.0)  # (n_layers,)
    num = np.sum(weights * h_53[np.newaxis, :], axis=1)  # (n_samples,)
    den = np.sum(weights, axis=1)
    den = np.clip(den, 1e-30, None)
    return np.power(num / den, 3.0 / 5.0)


def isoplanatic_angle(
    y_true: Sequence,
    y_pred: Sequence,
    heights: Sequence[float],
    wavelength: float = 500e-9,
    detailed: bool = False,
    values_are_integrals: bool = True,
) -> dict:
    """RMSE of the isoplanatic angle theta_0 (arcseconds).

    theta_0 = 0.314 * (r0 / H_eff)

    where H_eff is the turbulence-weighted effective height.
    theta_0 determines the angular field over which adaptive-optics
    correction remains valid.

    Args:
        y_true: True profile (samples x layers).
        y_pred: Predicted profile (samples x layers).
        heights: Layer heights in meters.
        wavelength: Wavelength in meters (default 500 nm).
        detailed: If True, return per-sample values.
        values_are_integrals: If True, values are J_i (m^{1/3}).
    """
    y_true, y_pred = _get_valid_indices(y_true, y_pred)
    if len(y_pred) == 0:
        return _format_metric(np.nan, 0)

    if values_are_integrals and np.nanmedian(y_true) < 0:
        y_true = np.power(10.0, y_true)
        y_pred = np.power(10.0, y_pred)

    h_arr = np.asarray(heights, dtype=float)

    def _theta0(profile):
        _, r0 = _profile_to_j_and_r0(profile, h_arr, wavelength, values_are_integrals)
        h_eff = _effective_height(profile, h_arr, values_are_integrals)
        h_eff = np.clip(h_eff, 1.0, None)  # avoid division by zero
        theta0_rad = 0.314 * r0 / h_eff
        return theta0_rad * 206265.0  # to arcseconds

    theta0_true = _theta0(y_true)
    theta0_pred = _theta0(y_pred)

    error = theta0_true - theta0_pred
    rmse = np.sqrt(np.mean(error**2))
    res = _format_metric(float(rmse), len(y_pred))

    if detailed:
        res["detailed_score"] = error.tolist()
        res["theta0_true"] = theta0_true.tolist()
        res["theta0_pred"] = theta0_pred.tolist()
    return res


def coherence_time(
    y_true: Sequence,
    y_pred: Sequence,
    heights: Sequence[float],
    wind_speed: float = 10.0,
    wavelength: float = 500e-9,
    detailed: bool = False,
    values_are_integrals: bool = True,
) -> dict:
    """RMSE of the atmospheric coherence time tau_0 (milliseconds).

    tau_0 = 0.314 * r0 / v_bar

    where v_bar is the effective wind speed. When only surface wind is
    available it is used as a scalar approximation.

    Args:
        y_true: True profile (samples x layers).
        y_pred: Predicted profile (samples x layers).
        heights: Layer heights in meters.
        wind_speed: Effective wind speed in m/s (scalar default 10 m/s).
        wavelength: Wavelength in meters (default 500 nm).
        detailed: If True, return per-sample values.
        values_are_integrals: If True, values are J_i (m^{1/3}).
    """
    y_true, y_pred = _get_valid_indices(y_true, y_pred)
    if len(y_pred) == 0:
        return _format_metric(np.nan, 0)

    if values_are_integrals and np.nanmedian(y_true) < 0:
        y_true = np.power(10.0, y_true)
        y_pred = np.power(10.0, y_pred)

    h_arr = np.asarray(heights, dtype=float)
    v = float(wind_speed)

    def _tau0(profile):
        _, r0 = _profile_to_j_and_r0(profile, h_arr, wavelength, values_are_integrals)
        tau0_s = 0.314 * r0 / v
        return tau0_s * 1000.0  # to milliseconds

    tau0_true = _tau0(y_true)
    tau0_pred = _tau0(y_pred)

    error = tau0_true - tau0_pred
    rmse = np.sqrt(np.mean(error**2))
    res = _format_metric(float(rmse), len(y_pred))

    if detailed:
        res["detailed_score"] = error.tolist()
        res["tau0_true"] = tau0_true.tolist()
        res["tau0_pred"] = tau0_pred.tolist()
    return res


def greenwood_frequency(
    y_true: Sequence,
    y_pred: Sequence,
    heights: Sequence[float],
    wind_speed: float = 10.0,
    wavelength: float = 500e-9,
    detailed: bool = False,
    values_are_integrals: bool = True,
) -> dict:
    """RMSE of the Greenwood frequency f_G (Hz).

    f_G = 0.427 * v_bar / r0

    The Greenwood frequency sets the minimum AO loop bandwidth needed to
    keep the servo lag error below an acceptable level.

    Args:
        y_true: True profile (samples x layers).
        y_pred: Predicted profile (samples x layers).
        heights: Layer heights in meters.
        wind_speed: Effective wind speed in m/s (scalar default 10 m/s).
        wavelength: Wavelength in meters (default 500 nm).
        detailed: If True, return per-sample values.
        values_are_integrals: If True, values are J_i (m^{1/3}).
    """
    y_true, y_pred = _get_valid_indices(y_true, y_pred)
    if len(y_pred) == 0:
        return _format_metric(np.nan, 0)

    if values_are_integrals and np.nanmedian(y_true) < 0:
        y_true = np.power(10.0, y_true)
        y_pred = np.power(10.0, y_pred)

    h_arr = np.asarray(heights, dtype=float)
    v = float(wind_speed)

    def _fg(profile):
        _, r0 = _profile_to_j_and_r0(profile, h_arr, wavelength, values_are_integrals)
        return 0.427 * v / r0  # Hz

    fg_true = _fg(y_true)
    fg_pred = _fg(y_pred)

    error = fg_true - fg_pred
    rmse = np.sqrt(np.mean(error**2))
    res = _format_metric(float(rmse), len(y_pred))

    if detailed:
        res["detailed_score"] = error.tolist()
        res["fg_true"] = fg_true.tolist()
        res["fg_pred"] = fg_pred.tolist()
    return res

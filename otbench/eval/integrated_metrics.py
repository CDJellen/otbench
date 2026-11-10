from typing import Sequence, Optional, Union
import warnings

import numpy as np
import pandas as pd

from otbench.eval.utils import _get_valid_indices, _format_metric

# np.trapezoid was introduced in NumPy 2.0; fall back to np.trapz for older installs.
try:
    _trapezoid = np.trapezoid
except AttributeError:
    _trapezoid = np.trapz


def _to_linear_profile(profile: np.ndarray) -> np.ndarray:
    """Convert log10 turbulence profiles (median < 0) to linear physical space."""
    if len(profile) > 0 and np.nanmedian(profile) < 0:
        return np.power(10.0, profile)
    return profile


def _prepare_profile_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Ensure y_true and y_pred are in linear physical space, warning on domain mismatch."""
    yt_is_log = bool(len(y_true) > 0 and np.nanmedian(y_true) < 0)
    yp_is_log = bool(len(y_pred) > 0 and np.nanmedian(y_pred) < 0)
    if yt_is_log != yp_is_log:
        warnings.warn(
            "Domain mismatch detected in integrated metric: one profile has negative median "
            f"(log10 space, y_true_log10={yt_is_log}) while the other has non-negative median "
            f"(linear space, y_pred_log10={yp_is_log}). Converting log10 profile(s) via 10**x.",
            UserWarning,
            stacklevel=3,
        )
    if yt_is_log:
        y_true = np.power(10.0, y_true)
    if yp_is_log:
        y_pred = np.power(10.0, y_pred)
    return y_true, y_pred


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

    # Detect and reverse log10 transform independently on y_true and y_pred.
    y_true, y_pred = _prepare_profile_pair(y_true, y_pred)

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


def _validate_profile_heights(
    profile: np.ndarray,
    heights: Optional[Sequence[float]],
) -> Optional[np.ndarray]:
    """Validate that heights length matches the profile layer dimension and altitudes are non-negative and sorted."""
    if heights is None:
        return None
    h_arr = np.asarray(heights, dtype=float)
    if np.any(np.isnan(h_arr)) or np.any(h_arr < 0.0):
        raise ValueError(f"heights must be finite and non-negative (>= 0 m), got {h_arr.tolist()}.")
    if len(h_arr) > 1 and np.any(np.diff(h_arr) < 0.0):
        raise ValueError(f"heights must be monotonically non-decreasing, got {h_arr.tolist()}.")
    if profile.ndim == 2 and profile.shape[1] > 1 and len(h_arr) != profile.shape[1]:
        raise ValueError(f"heights length ({len(h_arr)}) must match number of profile layers ({profile.shape[1]}).")
    return h_arr


def _compute_quadrature_slab_weights(h_arr: np.ndarray) -> np.ndarray:
    """Compute central-difference trapezoidal quadrature slab thicknesses dh_k (m) for altitude grid h_arr."""
    dh = np.empty_like(h_arr, dtype=float)
    if len(h_arr) == 0:
        return dh
    if len(h_arr) == 1:
        dh[0] = 1.0
        return dh
    dh[0] = 0.5 * (h_arr[1] - h_arr[0])
    dh[-1] = 0.5 * (h_arr[-1] - h_arr[-2])
    if len(h_arr) > 2:
        dh[1:-1] = 0.5 * (h_arr[2:] - h_arr[:-2])
    return np.maximum(dh, 0.0)


def _profile_to_j_and_r0(
    profile: np.ndarray,
    heights: Optional[np.ndarray],
    wavelength: float,
    values_are_integrals: bool,
) -> tuple:
    """Shared helper: profile -> (J_total, r0) per sample."""
    h_arr = _validate_profile_heights(profile, heights)
    k = 2 * np.pi / wavelength
    profile = np.maximum(profile, 1e-19)

    if values_are_integrals:
        J = np.sum(profile, axis=1)
    elif h_arr is not None:
        J = _trapezoid(profile, x=h_arr, axis=1)
    else:
        raise ValueError("heights must be provided when values_are_integrals=False.")

    J = np.clip(J, 1e-30, None)
    r0 = (0.423 * k**2 * J)**(-3 / 5)
    return J, r0


def profile_to_fried_parameter(
    profile: Sequence,
    heights: Optional[Sequence[float]] = None,
    wavelength: float = 500e-9,
    values_are_integrals: bool = True,
) -> tuple:
    """Compute column-integrated turbulence strength J_total (m^{1/3}) and Fried parameter r_0 (m).

    Automatically detects and exponentiates log10-transformed profiles (median < 0).

    Args:
        profile: 2D array or DataFrame of shape (n_samples, n_layers).
        heights: Optional layer altitudes in meters (required when values_are_integrals=False).
        wavelength: Optical wavelength in meters (default 500 nm).
        values_are_integrals: If True, profile entries are layer integrals J_i (m^{1/3});
            if False, entries are C_n^2 densities (m^{-2/3}).

    Returns:
        Tuple of (J_total, r0), each a 1D numpy.ndarray of length n_samples.
    """
    arr = np.asarray(profile, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    arr = _to_linear_profile(arr)
    h_arr = np.asarray(heights, dtype=float) if heights is not None else None
    return _profile_to_j_and_r0(arr, h_arr, wavelength, values_are_integrals)


def _effective_height(
    profile: np.ndarray,
    heights: np.ndarray,
    values_are_integrals: bool,
) -> np.ndarray:
    """Turbulence-weighted effective height H_eff per sample.

    H_eff = [ sum( J_i * h_i^(5/3) ) / sum( J_i ) ]^(3/5)
    """
    if heights is None:
        raise ValueError("heights must be provided to compute effective height H_eff.")
    h = _validate_profile_heights(profile, heights)
    profile = np.maximum(profile, 1e-19)
    h_53 = np.power(h, 5.0 / 3.0)  # (n_layers,)

    if values_are_integrals:
        num = np.sum(profile * h_53[np.newaxis, :], axis=1)  # (n_samples,)
        den = np.sum(profile, axis=1)
    elif h is not None:
        num = _trapezoid(profile * h_53[np.newaxis, :], x=h, axis=1)
        den = _trapezoid(profile, x=h, axis=1)
    else:
        raise ValueError("heights must be provided when values_are_integrals=False.")

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

    y_true, y_pred = _prepare_profile_pair(y_true, y_pred)

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


def _align_profiles_and_wind(
    y_true: Sequence,
    y_pred: Sequence,
    wind_speed: Union[float, Sequence[float]],
    min_wind_speed: float = 1.0,
) -> tuple:
    """Align y_true, y_pred, and scalar or 1D wind_speed, masking NaNs and clamping calm wind."""
    if isinstance(y_true, (pd.Series, pd.DataFrame)) and isinstance(y_pred, (pd.Series, pd.DataFrame)):
        if len(y_true) == len(y_pred) and not y_true.index.equals(y_pred.index):
            raise ValueError("y_true and y_pred are both pandas objects with mismatched indices "
                             f"(first entries: {y_true.index[:3].tolist()} vs {y_pred.index[:3].tolist()}). "
                             "Align indices before evaluation or pass numpy arrays.")

    yt_raw = np.asarray(y_true, dtype=float)
    yp_raw = np.asarray(y_pred, dtype=float)
    if len(yt_raw) == 0:
        return np.array([]), np.array([]), 10.0

    if yt_raw.ndim == 2 and yt_raw.shape[1] == 1:
        yt_raw = yt_raw.ravel()
    if yp_raw.ndim == 2 and yp_raw.shape[1] == 1:
        yp_raw = yp_raw.ravel()
    if yt_raw.shape != yp_raw.shape:
        raise ValueError(f"y_true and y_pred must have compatible shapes. Got {yt_raw.shape} vs {yp_raw.shape}.")

    if yt_raw.ndim == 1:
        mask = ~np.isnan(yt_raw) & ~np.isnan(yp_raw)
    elif yt_raw.ndim == 2:
        mask = ~np.isnan(yt_raw).any(axis=1) & ~np.isnan(yp_raw).any(axis=1)
    else:
        raise ValueError(f"Unsupported dimensionality for y_true: ndim={yt_raw.ndim}.")

    v = np.asarray(wind_speed, dtype=float)
    if v.ndim == 0:
        v_val = float(v)
        if np.isnan(v_val):
            return np.array([]), np.array([]), 10.0
        v_out: Union[float, np.ndarray] = max(v_val, float(min_wind_speed))
    else:
        v_1d = v.ravel()
        if len(v_1d) == len(yt_raw):
            mask = mask & ~np.isnan(v_1d)
            v_out = np.maximum(v_1d[mask], float(min_wind_speed))
        elif len(v_1d) == int(np.sum(mask)):
            valid_v = ~np.isnan(v_1d)
            v_out = np.maximum(v_1d[valid_v], float(min_wind_speed))
            yt_masked = yt_raw[mask][valid_v]
            yp_masked = yp_raw[mask][valid_v]
            return yt_masked, yp_masked, v_out
        else:
            raise ValueError(f"1D wind_speed length ({len(v_1d)}) must match number of samples ({len(yt_raw)}).")

    return yt_raw[mask], yp_raw[mask], v_out


def coherence_time(
    y_true: Sequence,
    y_pred: Sequence,
    heights: Sequence[float],
    wind_speed: Union[float, Sequence[float]] = 10.0,
    wavelength: float = 500e-9,
    detailed: bool = False,
    values_are_integrals: bool = True,
) -> dict:
    """RMSE of the atmospheric coherence time tau_0 (milliseconds).

    tau_0 = 0.314 * r0 / v_bar

    where v_bar is the effective wind speed. When only surface wind is
    available it is used as a scalar approximation or clamped per-sample sequence.

    Args:
        y_true: True profile (samples x layers).
        y_pred: Predicted profile (samples x layers).
        heights: Layer heights in meters.
        wind_speed: Effective wind speed in m/s (scalar default 10 m/s, or 1D per-sample sequence).
        wavelength: Wavelength in meters (default 500 nm).
        detailed: If True, return per-sample values.
        values_are_integrals: If True, values are J_i (m^{1/3}).
    """
    y_true, y_pred, v = _align_profiles_and_wind(y_true, y_pred, wind_speed)
    if len(y_pred) == 0:
        return _format_metric(np.nan, 0)

    y_true, y_pred = _prepare_profile_pair(y_true, y_pred)

    h_arr = np.asarray(heights, dtype=float)

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
    wind_speed: Union[float, Sequence[float]] = 10.0,
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
        wind_speed: Effective wind speed in m/s (scalar default 10 m/s, or 1D per-sample sequence).
        wavelength: Wavelength in meters (default 500 nm).
        detailed: If True, return per-sample values.
        values_are_integrals: If True, values are J_i (m^{1/3}).
    """
    y_true, y_pred, v = _align_profiles_and_wind(y_true, y_pred, wind_speed)
    if len(y_pred) == 0:
        return _format_metric(np.nan, 0)

    y_true, y_pred = _prepare_profile_pair(y_true, y_pred)

    h_arr = np.asarray(heights, dtype=float)

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


def ao_wavefront_error_budget(
    profile: Sequence,
    heights: Sequence[float],
    wind_speed: Union[float, Sequence[float]] = 10.0,
    wavelength: float = 1650e-9,
    actuator_pitch: float = 0.20,
    loop_bandwidth: float = 100.0,
    guide_star_angle: float = 5.0,
    values_are_integrals: bool = True,
) -> dict:
    """Decompose a turbulence profile into Adaptive Optics wavefront error variances and Strehl ratio.

    Implements the standard Kolmogorov / Maréchal closed-loop AO error budget at
    science wavelength ``wavelength`` (default 1.65 um H-band):
      1. DM Fitting variance (rad^2):
         sigma^2_fit = 0.28 * (d / r_0(lambda))^(5/3)
      2. Angular Anisoplanatism variance (rad^2):
         sigma^2_aniso = (theta_GS / theta_0(lambda))^(5/3)
      3. Temporal Servo-Lag variance (rad^2):
         sigma^2_servo = (f_G(lambda) / f_loop)^(5/3)
      4. Total Wavefront Variance & Maréchal Extended Strehl Ratio:
         sigma^2_total = sigma^2_fit + sigma^2_aniso + sigma^2_servo
         S = exp(-sigma^2_total)

    Also returns normalized vertical layer weights ``w_i = J_i / sum(J_k)`` for
    tomographic minimum-variance reconstructor priors.

    Args:
        profile: 2D array or DataFrame of shape (n_samples, n_layers). Automatically
            detects and exponentiates log10-transformed profiles (median < 0).
        heights: Layer heights in meters.
        wind_speed: Effective wind speed in m/s (scalar default 10.0 m/s or 1D per-sample sequence).
        wavelength: Science observation wavelength in meters (default 1.65e-6 m, H-band).
        actuator_pitch: Deformable mirror inter-actuator pitch projected on primary (m, default 0.20 m).
        loop_bandwidth: Closed-loop -3 dB control bandwidth in Hz (default 100.0 Hz).
        guide_star_angle: Angular separation between guide star and science target in arcseconds (default 5.0").
        values_are_integrals: If True, profile entries are layer integrals J_i (m^{1/3}).

    Returns:
        Dictionary containing per-sample 1D numpy arrays (`r0_m`, `theta0_arcsec`, `f_g_hz`,
        `tau0_ms`, `h_eff_m`, `sigma2_fitting`, `sigma2_anisoplanatic`, `sigma2_servo`,
        `sigma2_total`, `opd_rms_nm`, `strehl_ratio`) and 2D array `layer_weights` (n_samples, n_layers).
    """
    arr = np.asarray(profile, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    arr = _to_linear_profile(arr)

    arr = np.maximum(arr, 1e-19)
    h_arr = np.asarray(heights, dtype=float)
    v = np.asarray(wind_speed, dtype=float)
    v_clamped = np.maximum(v, 1.0)

    j_total, r0 = _profile_to_j_and_r0(arr, h_arr, wavelength, values_are_integrals)
    h_eff = np.clip(_effective_height(arr, h_arr, values_are_integrals), 1.0, None)
    theta0_arcsec = (0.314 * r0 / h_eff) * 206265.0
    f_g_hz = 0.427 * v_clamped / r0
    tau0_ms = (0.314 * r0 / v_clamped) * 1000.0

    sigma2_fit = 0.28 * np.power(float(actuator_pitch) / r0, 5.0 / 3.0)
    sigma2_aniso = np.power(float(guide_star_angle) / np.clip(theta0_arcsec, 1e-6, None), 5.0 / 3.0)
    sigma2_servo = np.power(f_g_hz / max(float(loop_bandwidth), 1e-3), 5.0 / 3.0)
    sigma2_total = sigma2_fit + sigma2_aniso + sigma2_servo

    opd_rms_nm = (wavelength / (2.0 * np.pi)) * np.sqrt(sigma2_total) * 1e9
    strehl = np.exp(-sigma2_total)
    if not values_are_integrals and len(h_arr) == arr.shape[1] and len(h_arr) >= 2:
        dh = _compute_quadrature_slab_weights(h_arr)
        layer_j = arr * dh[np.newaxis, :]
        layer_weights = layer_j / np.clip(np.sum(layer_j, axis=1, keepdims=True), 1e-30, None)
    else:
        layer_weights = arr / np.clip(np.sum(arr, axis=1, keepdims=True), 1e-30, None)

    return {
        "j_total": j_total,
        "r0_m": r0,
        "h_eff_m": h_eff,
        "theta0_arcsec": theta0_arcsec,
        "f_g_hz": f_g_hz,
        "tau0_ms": tau0_ms,
        "sigma2_fitting": sigma2_fit,
        "sigma2_anisoplanatic": sigma2_aniso,
        "sigma2_servo": sigma2_servo,
        "sigma2_total": sigma2_total,
        "opd_rms_nm": opd_rms_nm,
        "strehl_ratio": strehl,
        "layer_weights": layer_weights,
    }


def strehl_ratio(
    y_true: Sequence,
    y_pred: Sequence,
    heights: Sequence[float],
    wind_speed: Union[float, Sequence[float]] = 10.0,
    wavelength: float = 1650e-9,
    actuator_pitch: float = 0.20,
    loop_bandwidth: float = 100.0,
    guide_star_angle: float = 5.0,
    detailed: bool = False,
    values_are_integrals: bool = True,
) -> dict:
    """RMSE of the closed-loop Maréchal Strehl ratio S in (0, 1] at science wavelength (default H-band 1.65 um).

    Also computes ``mean_opd_rms_error_nm`` (mean absolute optical path difference error in nm)
    and ``layer_weight_kl_divergence`` (mean Kullback-Leibler divergence D_KL(w_true || w_pred)
    measuring tomographic reconstructor vertical prior mismatch).
    """
    y_true, y_pred, v = _align_profiles_and_wind(y_true, y_pred, wind_speed)
    if len(y_pred) == 0:
        res = _format_metric(np.nan, 0)
        res["mean_opd_rms_error_nm"] = np.nan
        res["layer_weight_kl_divergence"] = np.nan
        return res

    y_true, y_pred = _prepare_profile_pair(y_true, y_pred)

    budget_true = ao_wavefront_error_budget(
        y_true,
        heights=heights,
        wind_speed=v,
        wavelength=wavelength,
        actuator_pitch=actuator_pitch,
        loop_bandwidth=loop_bandwidth,
        guide_star_angle=guide_star_angle,
        values_are_integrals=values_are_integrals,
    )
    budget_pred = ao_wavefront_error_budget(
        y_pred,
        heights=heights,
        wind_speed=v,
        wavelength=wavelength,
        actuator_pitch=actuator_pitch,
        loop_bandwidth=loop_bandwidth,
        guide_star_angle=guide_star_angle,
        values_are_integrals=values_are_integrals,
    )

    s_true = budget_true["strehl_ratio"]
    s_pred = budget_pred["strehl_ratio"]
    error = s_true - s_pred
    rmse = float(np.sqrt(np.mean(error**2)))

    w_true = np.clip(budget_true["layer_weights"], 1e-12, 1.0)
    w_pred = np.clip(budget_pred["layer_weights"], 1e-12, 1.0)
    kl_div = float(np.mean(np.sum(w_true * np.log(w_true / w_pred), axis=1)))
    opd_mae_nm = float(np.mean(np.abs(budget_true["opd_rms_nm"] - budget_pred["opd_rms_nm"])))

    res = _format_metric(rmse, len(y_pred))
    res["mean_opd_rms_error_nm"] = opd_mae_nm
    res["layer_weight_kl_divergence"] = kl_div

    if detailed:
        res["detailed_score"] = error.tolist()
        res["strehl_true"] = s_true.tolist()
        res["strehl_pred"] = s_pred.tolist()
    return res


def ground_layer_fraction(
    y_true: Sequence,
    y_pred: Sequence,
    heights: Optional[Sequence[float]] = None,
    detailed: bool = False,
    values_are_integrals: bool = True,
) -> dict:
    """RMSE of the Ground-Layer Fraction (GLF = J_0 / sum_k J_k in [0, 1]).

    Quantifies the fraction of total column optical turbulence residing in the
    lowest (ground/surface) layer (k = 0, h_0 = 0 m), which directly governs
    Ground-Layer Adaptive Optics (GLAO) wide-field correction efficiency.

    Args:
        y_true: True profile (samples x layers).
        y_pred: Predicted profile (samples x layers).
        heights: Optional layer heights in meters (required when values_are_integrals=False).
        detailed: If True, return per-sample GLF errors and series.
        values_are_integrals: If True (default), values are J_i (m^{1/3}).

    Returns:
        Dictionary with 'metric_value' (RMSE of GLF), 'valid_predictions',
        'mean_glf_true', and 'mean_glf_pred'.
    """
    y_true, y_pred = _get_valid_indices(y_true, y_pred)
    if len(y_pred) == 0:
        res = _format_metric(np.nan, 0)
        res["mean_glf_true"] = np.nan
        res["mean_glf_pred"] = np.nan
        return res

    y_true, y_pred = _prepare_profile_pair(y_true, y_pred)
    h_arr = _validate_profile_heights(y_true, heights)

    def _compute_glf(profile: np.ndarray) -> np.ndarray:
        arr = np.maximum(profile, 1e-19)
        if arr.ndim == 1 or arr.shape[1] == 1:
            return np.ones(len(arr), dtype=float)
        if not values_are_integrals:
            if h_arr is None or len(h_arr) < 2:
                raise ValueError("heights must be provided when values_are_integrals=False.")
            dh = _compute_quadrature_slab_weights(h_arr)
            layer_j = arr * dh[np.newaxis, :]
            return layer_j[:, 0] / np.clip(np.sum(layer_j, axis=1), 1e-30, None)
        return arr[:, 0] / np.clip(np.sum(arr, axis=1), 1e-30, None)

    glf_true = _compute_glf(y_true)
    glf_pred = _compute_glf(y_pred)
    error = glf_true - glf_pred
    rmse = float(np.sqrt(np.mean(error**2)))

    res = _format_metric(rmse, len(y_pred))
    res["mean_glf_true"] = float(np.mean(glf_true))
    res["mean_glf_pred"] = float(np.mean(glf_pred))

    if detailed:
        res["detailed_score"] = error.tolist()
        res["glf_true"] = glf_true.tolist()
        res["glf_pred"] = glf_pred.tolist()
    return res

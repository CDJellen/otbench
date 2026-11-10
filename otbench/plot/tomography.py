import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import cmocean
    HAS_CMOCEAN = True
except ImportError:
    HAS_CMOCEAN = False


def _parse_heights(feature_names: List[str]) -> np.ndarray:
    """Heuristic to extract physical heights from column names like 'cn2_free_atmos_1000'."""
    heights = []
    for f in feature_names:
        # Match the last number in the string
        match = re.search(r'(\d+)$', f)
        if match:
            h = float(match.group(1))
            # Handle the 'Ground' scalar case (often 0 or undefined)
            if 'ground' in f.lower():
                h = 0.0
            heights.append(h)
        else:
            heights.append(0.0)
    return np.array(heights)


def compute_mass_slab_thicknesses(heights: Sequence[float]) -> np.ndarray:
    """Compute effective vertical slab thicknesses dh_k (m) for MASS-DIMM logarithmic altitude grids.

    For the ground layer (h_0 <= 0), uses the nominal 500 m surface-to-first-MASS-slab thickness.
    For free-atmosphere layers (h_k > 0), uses geometric-mean slab boundaries
    [sqrt(h_{k-1} * h_k), sqrt(h_k * h_{k+1})], with half-octave endpoints h_1 / sqrt(2)
    and h_K * sqrt(2). For heights = [0, 500, 1000, 2000, 4000, 8000, 16000] m, this yields
    dh = [500.0, 353.6, 707.1, 1414.2, 2828.4, 5656.9, 11313.7] m.
    """
    h_arr = np.asarray(heights, dtype=float)
    h_pos = np.sort(h_arr[h_arr > 0])
    dh = np.empty_like(h_arr, dtype=float)
    for i, h in enumerate(h_arr):
        if h <= 0:
            dh[i] = 500.0
        else:
            idx = int(np.searchsorted(h_pos, h))
            lo = np.sqrt(h_pos[idx - 1] * h) if idx > 0 else h / np.sqrt(2.0)
            hi = np.sqrt(h * h_pos[idx + 1]) if idx < len(h_pos) - 1 else h * np.sqrt(2.0)
            dh[i] = hi - lo
    return np.maximum(dh, 1.0)


def plot_profile_comparison(y_true: Union[np.ndarray, pd.DataFrame],
                            y_pred: Union[np.ndarray, pd.DataFrame],
                            feature_names: Optional[List[str]] = None,
                            heights: Optional[Sequence[float]] = None,
                            title: str = "Vertical Structure",
                            xlabel: str = r"Turbulence Strength ($\log_{10} C_n^2$)",
                            ylabel: str = "Altitude (m)",
                            ax: Optional[plt.Axes] = None,
                            add_hv_reference: bool = True,
                            convert_to_density: bool = True) -> plt.Axes:
    """
    Plots the vertical profile (Altitude vs Cn2) with physical references.
    
    Args:
        convert_to_density: If True, assumes input is Integral (J) and divides by 
                            layer thickness (dh) to compare with HV-5/7 ($C_n^2$).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))

    # Data Prep
    if isinstance(y_true, (pd.DataFrame, pd.Series)):
        y_true = y_true.to_numpy()
    if isinstance(y_pred, (pd.DataFrame, pd.Series)):
        y_pred = y_pred.to_numpy()

    # 1. Height Resolution
    if heights is None:
        if feature_names:
            heights = _parse_heights(feature_names)
        else:
            heights = np.arange(y_true.shape[1])
    heights = np.asarray(heights, dtype=float)

    # 2. Physics Conversion: Integral (J) -> Density (Cn2)
    # Clamp integral J at the instrument floor (1e-19 m^{1/3}, or -19.0 in log10)
    # BEFORE dividing by slab thickness dh (m) so volumetric density Cn2 = J / dh
    # (units: m^{-2/3}, down to ~1e-23) is not artificially truncated at 1e-19.
    if convert_to_density:
        dh = compute_mass_slab_thicknesses(heights)
        log10_dh = np.log10(dh)

        is_log = np.nanmedian(y_true) < 0
        if is_log:
            y_true = np.maximum(y_true, -19.0) - log10_dh
            y_pred = np.maximum(y_pred, -19.0) - log10_dh
        else:
            y_true = np.log10(np.maximum(y_true, 1e-19)) - log10_dh
            y_pred = np.log10(np.maximum(y_pred, 1e-19)) - log10_dh

    # Sort by height for proper plotting
    sort_idx = np.argsort(heights)
    h_sorted = heights[sort_idx]

    # 3. Statistics (Compute NaN-safe Mean/Std in Log Space)
    mu_true = np.nanmean(y_true, axis=0)[sort_idx]
    std_true = np.nanstd(y_true, axis=0)[sort_idx]
    mu_pred = np.nanmean(y_pred, axis=0)[sort_idx]
    std_pred = np.nanstd(y_pred, axis=0)[sort_idx]

    # 4. Reference Physics (Hufnagel-Valley 5/7)
    if add_hv_reference:
        h_ref = np.linspace(1, max(h_sorted) * 1.1, 200)  # start at 1m to avoid h=0
        # Standard HV-5/7: v_rms = 21 m/s, A = 1.7e-14
        v_rms = 21.0
        hv = (5.94e-53 * (v_rms / 27.0)**2 * h_ref**10 * np.exp(-h_ref / 1000.0) + 2.7e-16 * np.exp(-h_ref / 1500.0) +
              1.7e-14 * np.exp(-h_ref / 100.0))
        ax.plot(np.log10(hv), h_ref, 'k--', alpha=0.4, label='Hufnagel-Valley 5/7 (Theory)')

    # 5. Plotting
    # True Data (Mean + Shading)
    ax.plot(mu_true, h_sorted, 'o-', color='#00356B', lw=2, label='Measured (Mean)')
    ax.fill_betweenx(h_sorted,
                     mu_true - std_true,
                     mu_true + std_true,
                     color='#00356B',
                     alpha=0.14,
                     label=r'Measured Variability ($\pm 1\sigma$)')

    # Predicted Data (Mean + Shading)
    ax.plot(mu_pred, h_sorted, 's--', color='#C90016', lw=2, label='Forecast (Mean)')
    ax.fill_betweenx(h_sorted,
                     mu_pred - std_pred,
                     mu_pred + std_pred,
                     color='#C90016',
                     alpha=0.12,
                     label=r'Forecast Variability ($\pm 1\sigma$)')

    # Formatting
    ax.set_facecolor("#faf9f6")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.set_ylim(0, max(h_sorted) * 1.05)

    return ax


def plot_time_series_heatmap(data: Union[np.ndarray, pd.DataFrame],
                             feature_names: Optional[List[str]] = None,
                             heights: Optional[Sequence[float]] = None,
                             title: str = "Turbulence Evolution",
                             xlabel: Optional[str] = "Time Step",
                             ylabel: str = "Altitude (m)",
                             cmap: str = "magma",
                             vmin: Optional[float] = None,
                             vmax: Optional[float] = None,
                             ax: Optional[plt.Axes] = None,
                             add_cbar: bool = True,
                             cbar_label: str = "Magnitude") -> plt.Axes:
    """
    Plots a tomogram with optional colorbar and clean axis labeling.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    if isinstance(data, pd.DataFrame):
        if feature_names is None:
            feature_names = data.columns.tolist()
        data_arr = data.to_numpy()
    else:
        data_arr = data

    if heights is None:
        heights = _parse_heights(feature_names) if feature_names else np.arange(data_arr.shape[1])
    heights = np.asarray(heights, dtype=float)

    sort_idx = np.argsort(heights)
    h_sorted = heights[sort_idx]
    data_sorted = data_arr[:, sort_idx]

    # Map array indices to physical units [x0, x1, y0, y1]
    extent = [0, data_sorted.shape[0], h_sorted[0], h_sorted[-1]]

    if HAS_CMOCEAN and cmap in dir(cmocean.cm):
        cmap = getattr(cmocean.cm, cmap)

    im = ax.imshow(data_sorted.T,
                   aspect='auto',
                   origin='lower',
                   cmap=cmap,
                   extent=extent,
                   interpolation='nearest',
                   vmin=vmin,
                   vmax=vmax)

    ax.set_title(title)
    ax.set_ylabel(ylabel)

    # Conditionally set X-label (prevents duplication in subplots)
    if xlabel:
        ax.set_xlabel(xlabel)
    else:
        # Hide tick labels if no label provided (cleaner for stacking)
        ax.tick_params(labelbottom=False)

    if add_cbar:
        # Magic fraction/pad to make colorbar height match the plot height
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)

    return ax


def plot_horizon_decay(
    horizon_df: pd.DataFrame,
    metric: str = "root_mean_square_error",
    ax: Optional[plt.Axes] = None,
    title: str = "Forecast Predictability Horizon Decay",
    ylabel: Optional[str] = None,
    xlabel: str = "Forecast Horizon (steps ahead)",
) -> plt.Axes:
    """Plot forecast error or skill score as a function of lead time H from :meth:`ForecastingTask.evaluate_horizon_curve`."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    if horizon_df.empty or metric not in horizon_df.columns:
        raise ValueError(f"Metric '{metric}' not found in horizon_df columns: {list(horizon_df.columns)}")

    palette = ["#00356B", "#C90016", "#2E7D32", "#6A1B9A", "#EF6C00", "#37474F"]
    for idx, (model_name, grp) in enumerate(horizon_df.groupby("model", sort=False)):
        grp_sorted = grp.sort_values("horizon")
        ax.plot(
            grp_sorted["horizon"].to_numpy(),
            grp_sorted[metric].to_numpy(),
            marker="o",
            linewidth=2.0,
            color=palette[idx % len(palette)],
            label=str(model_name),
        )

    if metric == "skill_score":
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_facecolor("#faf9f6")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel is not None else metric)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True)
    return ax


def plot_profile_error_heatmap(
    per_layer_scores: Union[pd.DataFrame, Dict[str, Sequence[float]]],
    layer_labels: Optional[Sequence[str]] = None,
    title: str = "Layer-Wise Forecast Error by Model and Altitude",
    cbar_label: str = r"Per-Layer RMSE ($\log_{10}\ \mathrm{m}^{1/3}$)",
    cmap: str = "YlOrRd",
    annotate: bool = True,
    fmt: str = ".3f",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a 2D Model x Altitude Layer heatmap of per-layer RMSE or Persistence Skill Score."""
    if isinstance(per_layer_scores, dict):
        cols = list(layer_labels) if layer_labels is not None else None
        df = pd.DataFrame.from_dict(per_layer_scores, orient="index", columns=cols)
    elif isinstance(per_layer_scores, pd.DataFrame):
        df = per_layer_scores.copy()
        if layer_labels is not None and len(layer_labels) == df.shape[1]:
            df.columns = list(layer_labels)
    else:
        raise TypeError("per_layer_scores must be a pandas DataFrame or dict mapping model names to layer arrays.")

    if df.empty:
        raise ValueError("per_layer_scores must not be empty.")

    if ax is None:
        _, ax = plt.subplots(figsize=(9.0, max(3.5, 0.55 * len(df) + 1.5)))

    mat = df.to_numpy(dtype=float)
    im = ax.imshow(mat, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels([str(c) for c in df.columns], rotation=25, ha="right")
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels([str(idx) for idx in df.index])
    ax.set_xlabel("Altitude Layer")
    ax.set_title(title)

    if annotate:
        vmin = float(np.nanmin(mat))
        vmax = float(np.nanmax(mat))
        mid = 0.5 * (vmin + vmax)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if np.isnan(val):
                    continue
                text_color = "white" if val > mid else "#111111"
                ax.text(j, i, format(val, fmt), ha="center", va="center", color=text_color, fontsize=8.5)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    return ax


def plot_paranal_climatology(
        profile_df: pd.DataFrame,
        heights: Sequence[float] = (0.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0),
        seeing_series: Optional[Union[pd.Series, np.ndarray]] = None,
        figsize: Tuple[float, float] = (13.0, 9.5),
) -> plt.Figure:
    """Plot a 4-panel vertical optical turbulence climatology summary for ESO Cerro Paranal.

    Panels:
        (a) Empirical CDFs of log10(J_i [m^{1/3}]) across all 7 altitude layers.
        (b) Vertical Cn2(h) [m^{-2/3}] median & interquartile profile vs. Hufnagel-Valley 5/7.
        (c) Distribution of Ground-Layer (J_0 / J_tot) vs. Free-Atmosphere turbulence fraction.
        (d) Nocturnal evolution of ground vs. free-atmosphere integrated turbulence across UTC night hours.
    """
    if profile_df.empty:
        raise ValueError("profile_df must not be empty.")

    arr = profile_df.to_numpy(dtype=float)
    valid_mask = ~np.isnan(arr).any(axis=1)
    arr_valid = arr[valid_mask]
    if len(arr_valid) == 0:
        raise ValueError("profile_df contains no complete non-NaN profile rows.")

    is_log = bool(np.nanmedian(arr_valid) < 0)
    log_j = arr_valid if is_log else np.log10(np.maximum(arr_valid, 1e-19))
    lin_j = 10.0**log_j

    h_arr = np.asarray(heights, dtype=float)
    layer_Palette = ["#00356B", "#1565C0", "#00838F", "#2E7D32", "#EF6C00", "#C90016", "#6A1B9A"]
    layer_names = [f"{int(h)} m (GL)" if h <= 0 else f"{int(h)} m" for h in h_arr]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_cdf, ax_prof, ax_frac, ax_diurnal = axes.ravel()

    # Panel (a): Empirical CDFs per layer
    for k in range(log_j.shape[1]):
        col_sorted = np.sort(log_j[:, k])
        cdf = np.linspace(0.0, 1.0, len(col_sorted), endpoint=True)
        ax_cdf.plot(col_sorted, cdf, lw=1.8, color=layer_Palette[k % len(layer_Palette)], label=layer_names[k])
    ax_cdf.set_facecolor("#faf9f6")
    ax_cdf.set_xlabel(r"Layer Turbulence Integral $\log_{10}(J_i\ [\mathrm{m}^{1/3}])$")
    ax_cdf.set_ylabel("Cumulative Probability")
    ax_cdf.set_title("(a) Layer-Wise Empirical CDFs")
    ax_cdf.grid(True, ls="--", alpha=0.35)
    ax_cdf.legend(loc="upper left", fontsize=8, frameon=True)

    # Panel (b): Vertical Cn2(h) profile vs Hufnagel-Valley 5/7
    dh = compute_mass_slab_thicknesses(h_arr)
    log_cn2 = np.maximum(log_j, -19.0) - np.log10(dh)
    q10, q25, q50, q75, q90 = np.percentile(log_cn2, [10, 25, 50, 75, 90], axis=0)
    h_km = h_arr / 1000.0

    h_ref = np.linspace(10.0, max(h_arr) * 1.02, 250)
    v_rms = 21.0
    hv = (5.94e-53 * (v_rms / 27.0)**2 * h_ref**10 * np.exp(-h_ref / 1000.0) + 2.7e-16 * np.exp(-h_ref / 1500.0) +
          1.7e-14 * np.exp(-h_ref / 100.0))
    ax_prof.plot(np.log10(hv), h_ref / 1000.0, "k--", lw=1.6, alpha=0.65, label="Hufnagel-Valley 5/7")
    ax_prof.plot(q50, h_km, "o-", color="#00356B", lw=2.2, label="Paranal Median")
    ax_prof.fill_betweenx(h_km, q25, q75, color="#00356B", alpha=0.25, label="Interquartile (P25-P75)")
    ax_prof.fill_betweenx(h_km, q10, q90, color="#00356B", alpha=0.10, label="Interdecile (P10-P90)")
    ax_prof.set_facecolor("#faf9f6")
    ax_prof.set_xlabel(r"Turbulence Density $\log_{10}(C_n^2(h)\ [\mathrm{m}^{-2/3}])$")
    ax_prof.set_ylabel("Altitude AGL (km)")
    ax_prof.set_title(r"(b) Vertical $C_n^2(h)$ Climatology vs. HV-5/7")
    ax_prof.grid(True, ls="--", alpha=0.35)
    ax_prof.legend(loc="upper right", fontsize=8, frameon=True)

    # Panel (c): Ground Layer vs Free Atmosphere Fraction
    j_tot = np.sum(lin_j, axis=1)
    f_gl = np.clip(lin_j[:, 0] / np.maximum(j_tot, 1e-19), 0.0, 1.0)
    f_fa = 1.0 - f_gl
    ax_frac.hist(f_gl,
                 bins=40,
                 density=True,
                 color="#00356B",
                 alpha=0.65,
                 label=f"Ground Layer (med={np.median(f_gl):.2f})")
    ax_frac.hist(f_fa,
                 bins=40,
                 density=True,
                 color="#C90016",
                 alpha=0.45,
                 label=f"Free Atmos (med={np.median(f_fa):.2f})")
    ax_frac.axvline(float(np.median(f_gl)), color="#00356B", ls="--", lw=1.6)
    ax_frac.axvline(float(np.median(f_fa)), color="#C90016", ls="--", lw=1.6)
    ax_frac.set_facecolor("#faf9f6")
    ax_frac.set_xlabel(r"Column Turbulence Fraction ($J_{\mathrm{sub}} / J_{\mathrm{total}}$)")
    ax_frac.set_ylabel("Probability Density")
    ax_frac.set_title("(c) Ground-Layer vs. Free-Atmosphere Partition")
    ax_frac.grid(True, ls="--", alpha=0.35)
    ax_frac.legend(loc="upper center", fontsize=8, frameon=True)

    # Panel (d): Nocturnal evolution by UTC hour (if DatetimeIndex available)
    ax_diurnal.set_facecolor("#faf9f6")
    valid_index = profile_df.index[valid_mask]
    log_j_fa = np.log10(np.maximum(np.sum(lin_j[:, 1:], axis=1), 1e-19))
    if isinstance(valid_index, pd.DatetimeIndex):
        hours = valid_index.hour.to_numpy()
        min_bin_count = max(1, int(0.001 * len(hours)))
        raw_h = [int(h) for h in np.unique(hours) if int(np.sum(hours == h)) >= min_bin_count]
        if not raw_h:
            raw_h = [int(h) for h in np.unique(hours)]
        # Order nocturnal hours chronologically across UTC midnight (e.g. 23, 00, 01, ..., 10 UTC)
        has_daytime_gap = any(h not in raw_h for h in range(12, 20))
        unique_h = sorted(raw_h, key=lambda h: (h - 12) % 24) if has_daytime_gap else sorted(raw_h)
        x_pos = np.arange(len(unique_h))
        gl_med = [float(np.median(log_j[hours == h, 0])) for h in unique_h]
        gl_q25 = [float(np.percentile(log_j[hours == h, 0], 25)) for h in unique_h]
        gl_q75 = [float(np.percentile(log_j[hours == h, 0], 75)) for h in unique_h]
        fa_med = [float(np.median(log_j_fa[hours == h])) for h in unique_h]
        fa_q25 = [float(np.percentile(log_j_fa[hours == h], 25)) for h in unique_h]
        fa_q75 = [float(np.percentile(log_j_fa[hours == h], 75)) for h in unique_h]
        ax_diurnal.plot(x_pos, gl_med, "o-", color="#00356B", lw=2.0, label=r"Ground Layer $\log_{10} J_0$")
        ax_diurnal.fill_between(x_pos, gl_q25, gl_q75, color="#00356B", alpha=0.18)
        ax_diurnal.plot(x_pos,
                        fa_med,
                        "s--",
                        color="#C90016",
                        lw=2.0,
                        label=r"Free Atmos $\log_{10} \sum_{i\geq 1} J_i$")
        ax_diurnal.fill_between(x_pos, fa_q25, fa_q75, color="#C90016", alpha=0.18)
        ax_diurnal.set_xticks(x_pos)
        ax_diurnal.set_xticklabels([f"{h:02d}:00" for h in unique_h], rotation=25, ha="right")
        ax_diurnal.set_xlabel("Nocturnal Observation Hour (UTC)")
    else:
        bins = np.array_split(np.arange(len(log_j)), min(10, len(log_j)))
        x_idx = np.arange(len(bins))
        gl_med = [float(np.median(log_j[b, 0])) for b in bins]
        fa_med = [float(np.median(log_j_fa[b])) for b in bins]
        ax_diurnal.plot(x_idx, gl_med, "o-", color="#00356B", lw=2.0, label=r"Ground Layer $\log_{10} J_0$")
        ax_diurnal.plot(x_idx,
                        fa_med,
                        "s--",
                        color="#C90016",
                        lw=2.0,
                        label=r"Free Atmos $\log_{10} \sum_{i\geq 1} J_i$")
        ax_diurnal.set_xlabel("Normalized Nocturnal Decile")
    ax_diurnal.set_ylabel(r"Integrated Turbulence $\log_{10}(J\ [\mathrm{m}^{1/3}])$")
    ax_diurnal.set_title("(d) Nocturnal Boundary-Layer vs. Free-Atmosphere Evolution")
    ax_diurnal.grid(True, ls="--", alpha=0.35)
    ax_diurnal.legend(loc="best", fontsize=8, frameon=True)

    fig.tight_layout()
    return fig


def plot_burst_case_study(
        y_true: Union[np.ndarray, pd.DataFrame],
        predictions: Dict[str, Union[np.ndarray, pd.DataFrame]],
        heights: Sequence[float] = (0.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0),
        timestamps: Optional[Sequence] = None,
        primary_model: Optional[str] = None,
        figsize: Tuple[float, float] = (12.5, 9.0),
) -> plt.Figure:
    """Plot a 3-panel turbulence burst diagnostic (Tomogram, Integrated Seeing, Isoplanatic Angle)."""
    yt = y_true.to_numpy(dtype=float) if isinstance(y_true,
                                                    (pd.DataFrame, pd.Series)) else np.asarray(y_true, dtype=float)
    if yt.ndim != 2 or len(yt) == 0:
        raise ValueError("y_true must be a non-empty 2D profile array (N, n_layers).")

    if not predictions:
        raise ValueError("predictions dictionary must contain at least one model entry.")

    if primary_model is None or primary_model not in predictions:
        primary_model = next(iter(predictions.keys()))

    is_log = bool(np.nanmedian(yt) < 0)
    yt_log = yt if is_log else np.log10(np.maximum(yt, 1e-19))
    yt_lin = 10.0**yt_log

    h_arr = np.asarray(heights, dtype=float)
    wavelength = 500e-9

    from otbench.eval.integrated_metrics import _effective_height, _profile_to_j_and_r0

    def _calc_seeing_theta0(j_lin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _, r0 = _profile_to_j_and_r0(j_lin, h_arr, wavelength=wavelength, values_are_integrals=True)
        seeing = (0.98 * wavelength / r0) * 206265.0
        h_eff = np.clip(_effective_height(j_lin, h_arr, values_are_integrals=True), 1.0, None)
        theta0 = (0.314 * r0 / h_eff) * 206265.0
        return seeing, theta0

    true_seeing, true_theta0 = _calc_seeing_theta0(yt_lin)
    x_axis = np.arange(len(yt)) if timestamps is None else np.asarray(timestamps)

    fig, (ax_tomo, ax_see, ax_iso) = plt.subplots(3, 1, figsize=figsize, sharex=False)

    # Panel (a): Ground-Truth Observed Tomogram
    x_is_num = np.issubdtype(x_axis.dtype, np.number)
    tomo_extent = [float(x_axis[0]), float(x_axis[-1]), -0.5,
                   len(h_arr) - 0.5] if x_is_num and len(x_axis) > 1 else None
    im = ax_tomo.imshow(yt_log.T,
                        aspect="auto",
                        origin="lower",
                        cmap="magma",
                        interpolation="nearest",
                        extent=tomo_extent)
    ax_tomo.set_yticks(np.arange(len(h_arr)))
    ax_tomo.set_yticklabels([f"{int(h)} m" for h in h_arr])
    ax_tomo.set_ylabel("Altitude Layer")
    ax_tomo.set_title(r"(a) Observed Vertical Turbulence Evolution $\log_{10} J_i(t)$ Across MASS-DIMM Slabs")
    cbar = plt.colorbar(im, ax=ax_tomo, fraction=0.025, pad=0.02)
    cbar.set_label(r"$\log_{10} J_i$")

    # Panel (b) & (c): Seeing and Isoplanatic Angle Trajectories
    palette = ["#C90016", "#2E7D32", "#6A1B9A", "#EF6C00", "#1565C0"]
    ax_see.plot(x_axis, true_seeing, color="#00356B", lw=2.2, label="Observed (MASS-DIMM)")
    ax_iso.plot(x_axis, true_theta0, color="#00356B", lw=2.2, label="Observed (MASS-DIMM)")

    for idx, (m_name, m_pred) in enumerate(predictions.items()):
        mp_arr = m_pred.to_numpy(
            dtype=float) if isinstance(m_pred, (pd.DataFrame, pd.Series)) else np.asarray(m_pred, dtype=float)
        mp_log = mp_arr if bool(np.nanmedian(mp_arr) < 0) else np.log10(np.maximum(mp_arr, 1e-19))
        mp_see, mp_iso = _calc_seeing_theta0(10.0**mp_log)
        ls = "--" if "Persistence" in m_name else "-"
        lw = 2.0 if m_name == primary_model else 1.65
        col = palette[idx % len(palette)]
        ax_see.plot(x_axis, mp_see, ls=ls, lw=lw, color=col, alpha=0.92, label=m_name)
        ax_iso.plot(x_axis, mp_iso, ls=ls, lw=lw, color=col, alpha=0.92, label=m_name)

    if x_is_num and len(x_axis) > 1:
        ax_tomo.set_xlim(float(x_axis[0]), float(x_axis[-1]))
        ax_see.set_xlim(float(x_axis[0]), float(x_axis[-1]))
        ax_iso.set_xlim(float(x_axis[0]), float(x_axis[-1]))

    ax_see.set_facecolor("#faf9f6")
    ax_see.set_ylabel(r"Seeing $\varepsilon_0$ (arcsec)")
    ax_see.set_title(r"(b) Integrated Seeing $\varepsilon_0(t)$ Trajectory During Turbulence Burst")
    ax_see.grid(True, ls="--", alpha=0.35)
    ax_see.legend(loc="upper right", fontsize=8.5, frameon=True)

    ax_iso.set_facecolor("#faf9f6")
    ax_iso.set_xlabel("Observation Epoch (1-min steps)")
    ax_iso.set_ylabel(r"Isoplanatic Angle $\theta_0$ (arcsec)")
    ax_iso.set_title(r"(c) Upper-Tropospheric Isoplanatic Angle $\theta_0(t)$ Trajectory")
    ax_iso.grid(True, ls="--", alpha=0.35)
    ax_iso.legend(loc="upper right", fontsize=8.5, frameon=True)

    fig.tight_layout()
    return fig

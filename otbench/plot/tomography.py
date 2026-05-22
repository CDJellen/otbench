import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Sequence, List, Union, Optional
import re

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
    # We must apply this BEFORE log/mean to get physically correct density
    if convert_to_density:
        # Ground layer (h=0) uses a nominal 500 m thickness.
        # Paranal MASS Layer 0 integrates from the surface to ~500 m.
        h_arr = np.array(heights, dtype=float)
        h_sorted_tmp = np.sort(h_arr[h_arr > 0])  # positive heights only
        dh = np.empty_like(h_arr)
        for i, h in enumerate(h_arr):
            if h <= 0:
                dh[i] = 500.0
            else:
                idx = np.searchsorted(h_sorted_tmp, h)
                lo = np.sqrt(h_sorted_tmp[idx - 1] * h) if idx > 0 else h / np.sqrt(2)
                hi = np.sqrt(h * h_sorted_tmp[idx + 1]) if idx < len(h_sorted_tmp) - 1 else h * np.sqrt(2)
                dh[i] = hi - lo

        dh = np.maximum(dh, 1.0)

        # If data is log10-scale (values predominantly < 0), unlog first
        is_log = np.nanmedian(y_true) < 0
        if is_log:
            y_true = 10**y_true
            y_pred = 10**y_pred

        y_true = y_true / dh
        y_pred = y_pred / dh

        # Return to Log10 for plotting
        y_true = np.log10(np.maximum(y_true, 1e-19))
        y_pred = np.log10(np.maximum(y_pred, 1e-19))

    # Sort by height for proper plotting
    sort_idx = np.argsort(heights)
    h_sorted = heights[sort_idx]

    # 3. Statistics (Compute Mean/Std in Log Space)
    mu_true = np.mean(y_true, axis=0)[sort_idx]
    std_true = np.std(y_true, axis=0)[sort_idx]
    mu_pred = np.mean(y_pred, axis=0)[sort_idx]

    # 4. Reference Physics (Hufnagel-Valley 5/7)
    if add_hv_reference:
        h_ref = np.linspace(1, max(h_sorted) * 1.1, 200)  # start at 1m to avoid h=0
        # Standard HV-5/7: v_rms = 21 m/s, A = 1.7e-14
        v_rms = 21.0
        hv = (5.94e-53 * (v_rms / 27.0)**2 * h_ref**10 * np.exp(-h_ref / 1000.0) +
              2.7e-16 * np.exp(-h_ref / 1500.0) +
              1.7e-14 * np.exp(-h_ref / 100.0))
        ax.plot(np.log10(hv), h_ref, 'k--', alpha=0.4, label='Hufnagel-Valley 5/7 (Theory)')

    # 5. Plotting
    # True Data (Mean + Shading)
    ax.plot(mu_true, h_sorted, 'o-', color='#00356B', lw=2, label='Measured (Mean)')
    ax.fill_betweenx(h_sorted,
                     mu_true - std_true,
                     mu_true + std_true,
                     color='#00356B',
                     alpha=0.1,
                     label=r'Atmospheric Variability ($1\sigma$)')

    # Predicted Data
    ax.plot(mu_pred, h_sorted, 's--', color='#C90016', lw=2, label='Forecast')

    # Formatting
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

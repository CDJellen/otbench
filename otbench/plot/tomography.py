
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Sequence, List, Union, Optional

try:
    import cmocean
    HAS_CMOCEAN = True
except ImportError:
    HAS_CMOCEAN = False

def plot_profile_comparison(y_true: Union[np.ndarray, pd.DataFrame], 
                            y_pred: Union[np.ndarray, pd.DataFrame], 
                            heights: Optional[Sequence[float]] = None,
                            feature_names: Optional[List[str]] = None,
                            title: str = r"$\log_{10} C_n^2$ Profile Comparison",
                            xlabel: str = r'$\log_{10} C_n^2$',
                            ylabel: str = 'Height (m)',
                            ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plots the average vertical profile of True vs Predicted values.
    
    Args:
        y_true: True values (N, layers)
        y_pred: Predicted values (N, layers)
        heights: Altitudes for Y-axis. If None, uses layer index.
        feature_names: Names of layers/features.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        ax: Matplotlib axes to plot on.
        
    Returns:
        ax: The axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
        
    # Convert to numpy
    if isinstance(y_true, (pd.DataFrame, pd.Series)):
        y_true = y_true.to_numpy()
    if isinstance(y_pred, (pd.DataFrame, pd.Series)):
        y_pred = y_pred.to_numpy()
        
    # Calculate means
    mean_true = np.mean(y_true, axis=0)
    mean_pred = np.mean(y_pred, axis=0)
    
    # Check if we should log scale for plot (Cn2 is often log scale)
    # If the values are < 0 (log transformed data), we plot as is.
    # If values are > 0 and tiny, we might want to log them or semilogx.
    
    if heights is None:
        # Try to parse heights from feature_names if available
        # formats like 'cn2_1000', 'temp_profile_100'
        if feature_names:
            try:
                # heuristic: extract last number
                heights = []
                for f in feature_names:
                    # extract digits from end
                    import re
                    match = re.search(r'(\d+)$', f)
                    if match:
                        heights.append(float(match.group(1)))
                    else:
                        heights.append(0)
            except:
                heights = np.arange(len(mean_true))
        else:
            heights = np.arange(len(mean_true))
            
    # sort by height
    ht_arr = np.array(heights)
    idx = np.argsort(ht_arr)
    
    ax.plot(mean_true[idx], ht_arr[idx], label='True', marker='o', linestyle='-')
    ax.plot(mean_pred[idx], ht_arr[idx], label='Predicted', marker='x', linestyle='--')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", ls="-", alpha=0.5)
    
    return ax

def plot_time_series_heatmap(data: Union[np.ndarray, pd.DataFrame], 
                             feature_names: Optional[List[str]] = None,
                             time_index: Optional[Sequence] = None,
                             heights: Optional[Sequence[float]] = None,
                             title: str = "Turbulence Evolution",
                             xlabel: str = 'Time Step',
                             ylabel: str = 'Height (m)',
                             cmap: str = "viridis",
                             vmin: Optional[float] = None,  # <--- Added
                             vmax: Optional[float] = None,  # <--- Added
                             ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plots a time-series heatmap (Time x Height) of the profile evolution.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        
    if isinstance(data, pd.DataFrame):
        data_arr = data.to_numpy()
        if feature_names is None:
            feature_names = data.columns.tolist()
        if time_index is None:
            time_index = data.index
    else:
        data_arr = data
        
    # Heuristic for heights if missing
    if heights is None and feature_names:
        try:
             import re
             heights = []
             for f in feature_names:
                 match = re.search(r'(\d+)$', f)
                 if match:
                     heights.append(float(match.group(1)))
                 else:
                     heights.append(0)
        except:
             heights = np.arange(data_arr.shape[1])
             
    if heights is None:
        heights = np.arange(data_arr.shape[1])
        
    # Sort data by height for plotting
    ht_arr = np.array(heights)
    sort_idx = np.argsort(ht_arr)
    data_sorted = data_arr[:, sort_idx]
    ht_sorted = ht_arr[sort_idx]

    # Transpose for Time on X, Height on Y
    img_data = data_sorted.T
    
    # Extent [x_min, x_max, y_min, y_max]
    extent = [0, data_sorted.shape[0], ht_sorted[0], ht_sorted[-1]]
    
    if HAS_CMOCEAN and cmap == "thermal":
        cmap = cmocean.cm.thermal
    elif HAS_CMOCEAN and cmap == "deep":
        cmap = cmocean.cm.deep

    im = ax.imshow(img_data, aspect='auto', origin='lower', cmap=cmap, 
                   extent=extent, interpolation='nearest',
                   vmin=vmin, vmax=vmax)
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Magnitude')
    
    return ax

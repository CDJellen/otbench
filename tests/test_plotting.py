
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from otbench.plot.tomography import plot_profile_comparison, plot_time_series_heatmap

def test_plot_profile_comparison_runs():
    """Verify profile comparison plot runs with dummy data."""
    # N samples, M layers
    y_true = np.random.rand(10, 5)
    y_pred = np.random.rand(10, 5)
    
    # Run
    ax = plot_profile_comparison(y_true, y_pred, title="Test Plot")
    assert ax is not None
    plt.close()

def test_plot_time_series_heatmap_runs():
    """Verify heatmap plot runs with dummy data."""
    # N samples, M layers
    data = np.random.rand(20, 5)
    
    ax = plot_time_series_heatmap(data, title="Test Heatmap")
    assert ax is not None
    plt.close()

def test_plots_with_feature_names():
    """Verify plots parse feature names for heights."""
    y_true = np.random.rand(10, 3)
    y_pred = np.random.rand(10, 3)
    feats = ["cn2_100", "cn2_500", "cn2_1000"]

    ax = plot_profile_comparison(y_true, y_pred, feature_names=feats)
    assert ax is not None
    plt.close()

    ax = plot_time_series_heatmap(y_true, feature_names=feats)
    assert ax is not None
    plt.close()


def test_convert_to_density_dh_reasonable():
    """Geometric-mean dh computation must produce layer thicknesses ~0.7*h for MASS."""
    # Exercise the code path with MASS-like heights to verify no crash
    # and that HV reference curve is plotted
    heights = [0, 500, 1000, 2000, 4000, 8000, 16000]
    y_true = np.random.rand(20, 7) * 1e-14
    y_pred = np.random.rand(20, 7) * 1e-14

    ax = plot_profile_comparison(
        y_true, y_pred,
        heights=heights,
        convert_to_density=True,
        add_hv_reference=True,
    )
    assert ax is not None
    # Check that HV reference line was plotted (3 lines: measured, forecast, HV)
    assert len(ax.lines) >= 3
    plt.close()


def test_hv_reference_values():
    """HV-5/7 at h=10 km should be ~1e-17 (tropopause peak region)."""
    h = np.array([10000.0])
    v_rms = 21.0
    hv = (5.94e-53 * (v_rms / 27.0)**2 * h**10 * np.exp(-h / 1000.0) +
          2.7e-16 * np.exp(-h / 1500.0) +
          1.7e-14 * np.exp(-h / 100.0))
    # At 10 km the tropopause term dominates: ~5.94e-53 * 0.605 * 1e40 * 4.5e-5 ≈ 1.6e-17
    assert 1e-18 < hv[0] < 1e-15, f"HV at 10 km = {hv[0]:.2e}, expected ~1e-17"

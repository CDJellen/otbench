
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

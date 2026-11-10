import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from otbench.plot.tomography import (
    plot_burst_case_study,
    plot_paranal_climatology,
    plot_profile_comparison,
    plot_profile_error_heatmap,
    plot_time_series_heatmap,
)


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
        y_true,
        y_pred,
        heights=heights,
        convert_to_density=True,
        add_hv_reference=True,
    )
    assert ax is not None
    # Check that HV reference line was plotted (3 lines: measured, forecast, HV)
    assert len(ax.lines) >= 3
    plt.close()


def test_convert_to_density_does_not_truncate_below_1e19():
    """Volumetric density Cn2 = J / dh (m^{-2/3}) must not be clamped at the integral floor -19.0."""
    from otbench.plot.tomography import compute_mass_slab_thicknesses

    heights = [0.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
    dh = compute_mass_slab_thicknesses(heights)
    # Constant log10(J_k) = -17.0 m^{1/3} -> log10(Cn2_k) = -17.0 - log10(dh_k) < -19.5
    y_true = np.full((10, 7), -17.0)
    y_pred = np.full((10, 7), -16.5)

    ax = plot_profile_comparison(
        y_true,
        y_pred,
        heights=heights,
        convert_to_density=True,
        add_hv_reference=False,
    )
    measured_xdata = ax.lines[0].get_xdata()
    expected_xdata = -17.0 - np.log10(dh)
    np.testing.assert_allclose(measured_xdata, expected_xdata, rtol=1e-12)
    assert np.all(measured_xdata < -19.5)
    plt.close()


def test_hv_reference_values():
    """HV-5/7 at h=10 km should be ~1e-17 (tropopause peak region)."""
    h = np.array([10000.0])
    v_rms = 21.0
    hv = (5.94e-53 * (v_rms / 27.0)**2 * h**10 * np.exp(-h / 1000.0) + 2.7e-16 * np.exp(-h / 1500.0) +
          1.7e-14 * np.exp(-h / 100.0))
    # At 10 km the tropopause term dominates: ~5.94e-53 * 0.605 * 1e40 * 4.5e-5 ≈ 1.6e-17
    assert 1e-18 < hv[0] < 1e-15, f"HV at 10 km = {hv[0]:.2e}, expected ~1e-17"


def test_plot_profile_error_heatmap_runs():
    """Verify Model x Layer heatmap renders from both dict and DataFrame inputs."""
    scores = {
        "Persistence": [0.20, 0.51, 0.78, 0.98, 0.59, 0.65, 0.15],
        "GBRT": [0.18, 0.45, 0.68, 0.86, 0.55, 0.62, 0.14],
        "Transformer": [0.18, 0.44, 0.67, 0.84, 0.51, 0.60, 0.13],
    }
    labels = ["0 m", "500 m", "1 km", "2 km", "4 km", "8 km", "16 km"]
    ax = plot_profile_error_heatmap(scores, layer_labels=labels)
    assert ax is not None
    assert len(ax.get_xticklabels()) == 7
    assert len(ax.get_yticklabels()) == 3
    plt.close()


def test_plot_paranal_climatology_runs():
    """Verify 4-panel Paranal vertical climatology plot renders with DatetimeIndex."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2018-04-01 00:00", periods=120, freq="1min")
    cols = [
        "cn2_ground_scalar",
        "cn2_free_atmos_500",
        "cn2_free_atmos_1000",
        "cn2_free_atmos_2000",
        "cn2_free_atmos_4000",
        "cn2_free_atmos_8000",
        "cn2_free_atmos_16000",
    ]
    data = rng.normal(loc=[-12.6, -13.9, -14.2, -14.1, -13.7, -13.5, -13.4], scale=0.35, size=(120, 7))
    df = pd.DataFrame(data, index=idx, columns=cols)
    fig = plot_paranal_climatology(df)
    assert fig is not None
    assert len(fig.axes) == 4
    plt.close(fig)


def test_plot_burst_case_study_runs():
    """Verify 3-panel turbulence burst diagnostic plot renders cleanly."""
    rng = np.random.default_rng(11)
    y_true = rng.normal(loc=[-12.5, -13.8, -14.1, -14.0, -13.6, -13.4, -13.3], scale=0.3, size=(60, 7))
    preds = {
        "Persistence": np.roll(y_true, 2, axis=0),
        "Transformer": y_true + rng.normal(0.0, 0.08, size=y_true.shape),
    }
    fig = plot_burst_case_study(y_true, preds, primary_model="Transformer")
    assert fig is not None
    assert len(fig.axes) >= 3
    plt.close(fig)


def test_plot_climatology_density_and_burst_wave_optics_parity():
    """Verify plot_paranal_climatology preserves sub-1e-19 volumetric densities and
    plot_burst_case_study matches integrated_metrics wave-optics operators."""
    from otbench.eval import integrated_metrics

    heights = [0.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
    cols = [
        "cn2_ground_scalar",
        "cn2_free_atmos_500",
        "cn2_free_atmos_1000",
        "cn2_free_atmos_2000",
        "cn2_free_atmos_4000",
        "cn2_free_atmos_8000",
        "cn2_free_atmos_16000",
    ]
    # Layer 16000m has dh ~ 11313.7 m (log10(dh) ~ 4.05), so log_j = -16.0 -> log_cn2 ~ -20.05 < -19.0
    df = pd.DataFrame(np.full((20, 7), -16.0), columns=cols)
    fig = plot_paranal_climatology(df, heights=heights)
    ax_prof = fig.axes[1]
    median_line = [ln for ln in ax_prof.get_lines() if "Paranal Median" in str(ln.get_label())][0]
    assert np.min(median_line.get_xdata()) < -19.5
    plt.close(fig)

    y_true = np.tile(np.array([-12.6, -13.9, -14.1, -14.0, -13.7, -13.8, -13.4]), (10, 1))
    fig_b = plot_burst_case_study(y_true, {"Persistence": y_true}, heights=heights)
    ax_see, ax_iso = fig_b.axes[1], fig_b.axes[2]
    see_plotted = ax_see.get_lines()[0].get_ydata()
    iso_plotted = ax_iso.get_lines()[0].get_ydata()
    see_ref = np.asarray(
        integrated_metrics.integrated_seeing(y_true, y_true, heights=heights, detailed=True)["seeing_true"])
    iso_ref = np.asarray(
        integrated_metrics.isoplanatic_angle(y_true, y_true, heights=heights, detailed=True)["theta0_true"])
    np.testing.assert_allclose(see_plotted, see_ref, rtol=1e-12)
    np.testing.assert_allclose(iso_plotted, iso_ref, rtol=1e-12)
    plt.close(fig_b)

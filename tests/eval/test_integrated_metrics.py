import pytest
import numpy as np
import pandas as pd
from otbench.eval import metrics
from otbench.eval.integrated_metrics import integrated_seeing
from otbench.tasks.tasks import TaskABC


def test_metrics_detailed_return():
    """Verify that metrics return detailed info when requested."""
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.1, 1.9], [3.2, 3.8]])

    # RMSE
    # Diff: [[-0.1, 0.1], [-0.2, 0.2]]
    # SqDiff: [[0.01, 0.01], [0.04, 0.04]]
    # Mean Sq Error per output: [0.025, 0.025] -> RMSE per output [0.158, 0.158] (approx)

    res = metrics.root_mean_square_error(y_true, y_pred, detailed=True)
    assert isinstance(res, dict)
    assert "metric_value" in res
    assert "detailed_score" in res
    assert len(res["detailed_score"]) == 2
    assert np.allclose(res["detailed_score"], [0.15811388, 0.15811388])


def test_integrated_seeing_metric():
    """Verify integrated seeing metric calculation."""
    # simple constant profile
    y_true = np.ones((1, 5))  # 5 layers, value 1
    y_pred = np.ones((1, 5)) * 0.5  # 5 layers, value 0.5

    # heights None -> sum
    res = integrated_seeing(y_true, y_pred, heights=None, detailed=True)

    assert "seeing_true" in res
    assert "seeing_pred" in res
    # True sum = 5. Pred sum = 2.5.
    # r0 ~ (J)^-3/5
    # seeing ~ r0^-1 ~ (J)^3/5
    # true ~ 5^0.6 = 2.62
    # pred ~ 2.5^0.6 = 1.73
    # error ~ 0.89

    assert res["seeing_true"][0] > res["seeing_pred"][0]
    assert np.isclose(res['metric_value'], res['seeing_true'][0] - res['seeing_pred'][0])


def test_get_valid_indices_partial_nan_vector():
    """Vector targets with partial NaN rows must be dropped entirely."""
    from otbench.eval.utils import _get_valid_indices

    y_true = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
    y_pred = np.array([[1.0, 2.0], [3.0, 3.0], [4.0, 5.0]])

    yt, yp = _get_valid_indices(y_true, y_pred)
    # Row 1 has a NaN in y_true → must be dropped
    assert yt.shape == (2, 2)
    assert yp.shape == (2, 2)
    np.testing.assert_array_equal(yt, [[1.0, 2.0], [4.0, 5.0]])


def test_get_valid_indices_all_nan_row():
    """Rows where all values are NaN are still dropped."""
    from otbench.eval.utils import _get_valid_indices

    y_true = np.array([[np.nan, np.nan], [1.0, 2.0]])
    y_pred = np.array([[1.0, 1.0], [1.0, 2.0]])

    yt, yp = _get_valid_indices(y_true, y_pred)
    assert yt.shape == (1, 2)


def test_integrated_seeing_values_are_integrals():
    """When values_are_integrals=True, heights are ignored and values are summed."""
    # Two identical profiles → RMSE should be 0
    profile = np.array([[1e-14, 2e-14, 3e-14]])
    heights = np.array([500, 1000, 2000])

    res_integrals = integrated_seeing(profile, profile, heights=heights, values_are_integrals=True)
    assert np.isclose(res_integrals["metric_value"], 0.0)

    # With values_are_integrals=False, trapz integration should give a different
    # total J than simple sum → seeing values differ if we compare the two modes
    res_density = integrated_seeing(profile, profile, heights=heights, values_are_integrals=False)
    assert np.isclose(res_density["metric_value"], 0.0)


def test_integrated_seeing_sum_vs_trapz():
    """Verify that sum (integrals mode) and trapz (density mode) give different seeing."""
    profile = np.array([[1e-14, 2e-14, 3e-14]])
    heights = np.array([500, 1000, 2000])

    res_sum = integrated_seeing(profile, profile * 0.5, heights=heights, values_are_integrals=True, detailed=True)
    res_trapz = integrated_seeing(profile, profile * 0.5, heights=heights, values_are_integrals=False, detailed=True)
    # The total J differs between sum and trapz, so derived seeing should differ
    assert res_sum["seeing_true"][0] != res_trapz["seeing_true"][0]


def test_integrated_seeing_density_without_heights_raises():
    """values_are_integrals=False with heights=None must raise ValueError."""
    profile = np.array([[1e-14, 2e-14, 3e-14]])
    with pytest.raises(ValueError, match="heights must be provided"):
        integrated_seeing(profile, profile, heights=None, values_are_integrals=False)


def test_per_layer_rmse_layer_names_length_mismatch_raises():
    """per_layer_rmse must raise ValueError if layer_names length != n_layers."""
    y_true = np.ones((4, 7))
    y_pred = np.ones((4, 7)) * 0.9
    with pytest.raises(ValueError, match="layer_names length"):
        metrics.per_layer_rmse(y_true, y_pred, layer_names=["500m", "1000m"])


def test_build_metric_kwargs_hybrid_target_with_xarray_coords(monkeypatch):
    """Hybrid 7-layer target (ground + 6 free-atmos) must resolve 7 heights even when xr_ds.height_mass has 6."""
    import xarray as xr
    from otbench.tasks import TaskApi

    task = TaskApi().get_task("forecasting.paranal_tomography.full.cn2_profile_forecast",)
    mock_xr_ds = xr.Dataset(coords={"height_mass": ("height_mass", [500, 1000, 2000, 4000, 8000, 16000])})
    monkeypatch.setattr(task.get_dataset(), "get_xarray", lambda: mock_xr_ds)

    iso_kwargs = task._build_metric_kwargs("isoplanatic_angle")
    assert iso_kwargs["heights"] == [0, 500, 1000, 2000, 4000, 8000, 16000]

    seeing_kwargs = task._build_metric_kwargs("integrated_seeing")
    assert seeing_kwargs["heights"] == [0, 500, 1000, 2000, 4000, 8000, 16000]

    plr_kwargs = task._build_metric_kwargs("per_layer_rmse")
    assert plr_kwargs["layer_names"] == ["0m", "500m", "1000m", "2000m", "4000m", "8000m", "16000m"]


def test_base_forecasting_model_auto_detects_log10_target():
    """Residual forecasting models must auto-promote use_log10=True when y is log10 and X is linear."""
    from otbench.benchmark.models.forecasting import GradientBoostingForecastingModel

    rng = np.random.default_rng(42)
    n_samples = 40
    # X contains linear J values (~1e-15), y contains log10(J) values (~-15)
    j_t0 = 10.0**rng.uniform(-16.0, -14.0, size=(n_samples, 2))
    j_t1 = 10.0**rng.uniform(-16.0, -14.0, size=(n_samples, 2))
    y_next_log10 = np.log10(j_t0) + rng.normal(0.0, 0.1, size=(n_samples, 2))

    X = pd.DataFrame(
        np.hstack([j_t1, j_t0]),
        columns=["layer0 (t-1)", "layer1 (t-1)", "layer0 (t-0)", "layer1 (t-0)"],
    )
    y = pd.DataFrame(y_next_log10, columns=["layer0", "layer1"])

    model = GradientBoostingForecastingModel(
        name="GBRT_Residual",
        target_name=["layer0", "layer1"],
        window_size=2,
        forecast_horizon=1,
        predict_residuals=True,
        n_estimators=10,
        n_jobs=1,
    )
    assert model.use_log10 is False
    model.train(X, y)
    assert model.use_log10 is True
    preds = model.predict(X)
    assert preds.shape == (n_samples, 2)
    assert np.all((preds > -18.0) & (preds < -12.0))


def test_persistence_models_ignore_trailing_nan_in_train_target():
    """Persistence models must select the last valid (non-NaN) row of y during training."""
    from otbench.benchmark.models.regression.persistence import PersistenceRegressionModel
    from otbench.benchmark.models.forecasting.persistence import PersistenceForecastingModel

    X_reg = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    y_reg = pd.DataFrame({"t1": [-14.5, -14.2, np.nan], "t2": [-15.1, -14.8, np.nan]})

    reg_model = PersistenceRegressionModel(name="pers_reg", target_name=["t1", "t2"], output_size=2)
    reg_model.train(X_reg, y_reg)
    preds_reg = reg_model.predict(pd.DataFrame({"f1": [4.0, 5.0]}))
    assert preds_reg.shape == (2, 2)
    np.testing.assert_allclose(preds_reg[0], [-14.2, -14.8])

    fc_model = PersistenceForecastingModel(name="pers_fc",
                                           target_name=["t1", "t2"],
                                           window_size=1,
                                           forecast_horizon=1,
                                           output_size=2)
    fc_model.train(X_reg, y_reg)
    # Static fallback when target columns are absent from X
    preds_fc = fc_model.predict(pd.DataFrame({"f1": [4.0, 5.0]}))
    assert preds_fc.shape == (2, 2)
    np.testing.assert_allclose(preds_fc[0], [-14.2, -14.8])


def test_coherence_time_and_greenwood_frequency_vector_wind_speed():
    """coherence_time and greenwood_frequency must accept per-sample 1D wind_speed arrays,
    mask NaNs across both profiles and wind_speed, and clamp calm (0 m/s) wind speeds."""
    from otbench.eval.integrated_metrics import coherence_time, greenwood_frequency

    y_true = np.full((6, 3), 1e-15)
    y_pred = np.full((6, 3), 1.2e-15)
    y_true[1, 0] = np.nan  # Profile NaN row
    heights = [500, 1000, 2000]
    # Includes a NaN wind entry (index 2) and a calm 0.0 m/s entry (index 3)
    wind = np.array([8.0, 10.0, np.nan, 0.0, 12.0, 14.0])

    res_tau = coherence_time(y_true, y_pred, heights=heights, wind_speed=wind)
    res_fg = greenwood_frequency(y_true, y_pred, heights=heights, wind_speed=wind)
    assert res_tau["valid_predictions"] == 4
    assert res_fg["valid_predictions"] == 4
    assert np.isfinite(res_tau["metric_value"]) and res_tau["metric_value"] > 0
    assert np.isfinite(res_fg["metric_value"]) and res_fg["metric_value"] > 0


def test_profile_to_fried_parameter_and_nan_safe_profile_plot(tmp_path):
    """profile_to_fried_parameter must handle log10/linear inputs, and plot_profile_comparison
    must handle NaN rows without collapsing plotted lines to NaN."""
    import matplotlib.pyplot as plt
    from otbench.eval import profile_to_fried_parameter
    from otbench.plot.tomography import plot_profile_comparison

    profile_lin = np.full((5, 3), 1e-15)
    profile_log = np.log10(profile_lin)
    j_lin, r0_lin = profile_to_fried_parameter(profile_lin)
    j_log, r0_log = profile_to_fried_parameter(profile_log)
    np.testing.assert_allclose(j_lin, j_log)
    np.testing.assert_allclose(r0_lin, r0_log)

    y_true = profile_log.copy()
    y_pred = profile_log.copy() + 0.1
    y_true[0, 1] = np.nan
    y_pred[2, 0] = np.nan
    fig, ax = plt.subplots()
    plot_profile_comparison(y_true, y_pred, heights=[0, 500, 1000], ax=ax)
    for line in ax.get_lines():
        assert np.all(np.isfinite(line.get_xdata()))
    plt.close(fig)


def test_eso_fetcher_preserves_manifest_and_verifies_sha256(tmp_path):
    """ESOFetcher.fetch_campaign must preserve existing manifest entries and verify SHA-256 digests."""
    import json
    from otbench.data.paranal_tomography.fetch_eso import ESOFetcher

    fetcher = ESOFetcher(output_dir=str(tmp_path))
    shard = tmp_path / "mass_paranal_2021-01.csv"
    shard.write_bytes(b"Date time,Layer 1 Cn2\n" + b"2021-01-01T00:00:00,1.0\n" * 60)
    expected_sha = fetcher._compute_sha256(shard)

    manifest_path = tmp_path / "mass_paranal_manifest.json"
    manifest_path.write_text(json.dumps({"prior_shard.csv": "abc123", shard.name: expected_sha}))

    fetcher.fetch_campaign("mass_paranal", "2021-01-01", "2021-01-15", freq="MS")
    saved = json.loads(manifest_path.read_text())
    assert saved["prior_shard.csv"] == "abc123"
    assert saved[shard.name] == expected_sha


def test_regression_pytorch_rnn_supports_cell_types_and_vector_output():
    """Regression RNNModel must support rnn/gru/lstm cell_type and vector targets."""
    from otbench.benchmark.models.regression.pytorch.recurrent_neural_network import RNNModel

    X = pd.DataFrame(np.random.default_rng(0).normal(size=(24, 4)), columns=["f1", "f2", "f3", "f4"])
    y = pd.DataFrame(np.random.default_rng(1).normal(size=(24, 3)), columns=["t1", "t2", "t3"])

    for cell_type in ("rnn", "gru", "lstm"):
        mdl = RNNModel(
            name=f"reg_{cell_type}",
            target_name=["t1", "t2", "t3"],
            input_size=4,
            output_size=3,
            cell_type=cell_type,
            hidden_size=16,
            num_layers=1,
            n_epochs=2,
            batch_size=8,
            use_scheduler=True,
            val_fraction=0.2,
            patience=2,
        )
        mdl.train(X, y)
        preds = mdl.predict(X)
        assert preds.shape == (24, 3)
        assert np.all(np.isfinite(preds))


def test_compact_metrics_for_benchmark_and_experiments_json_size():
    """_compact_metrics_for_benchmark must strip sample-level arrays while keeping per-layer summaries,
    and experiments.json on disk must remain compact (<200 KB)."""
    import os
    from otbench.config import BENCHMARK_FP
    from otbench.tasks.tasks import _compact_metrics_for_benchmark

    raw_metrics = {
        "root_mean_square_error": {
            "metric_value": 0.48,
            "valid_predictions": 500,
            "detailed_score": [0.18, 0.44, 0.67, 0.85, 0.53, 0.59, 0.12],
        },
        "per_layer_rmse": {
            "metric_value": 0.48,
            "valid_predictions": 500,
            "per_layer": {
                "0m": 0.18,
                "500m": 0.44
            },
            "detailed_score": [[0.1, 0.2]] * 500,
        },
        "integrated_seeing": {
            "metric_value": 0.15,
            "valid_predictions": 500,
            "ci_lower": 0.14,
            "ci_upper": 0.16,
            "detailed_score": [0.01] * 500,
            "seeing_true": [0.8] * 500,
            "seeing_pred": [0.79] * 500,
        },
    }
    compacted = _compact_metrics_for_benchmark(raw_metrics)
    assert compacted["root_mean_square_error"]["detailed_score"] == [0.18, 0.44, 0.67, 0.85, 0.53, 0.59, 0.12]
    assert "per_layer" in compacted["per_layer_rmse"]
    assert "detailed_score" not in compacted["per_layer_rmse"]
    assert "seeing_true" not in compacted["integrated_seeing"]
    assert "detailed_score" not in compacted["integrated_seeing"]
    assert compacted["integrated_seeing"]["ci_lower"] == 0.14

    assert os.path.getsize(BENCHMARK_FP) < 200_000


def test_dataset_and_task_get_citation():
    """All registered datasets and tasks must return non-empty BibTeX citations."""
    from pathlib import Path
    from otbench.tasks import TaskApi

    api = TaskApi()
    for task_name in (
            "regression.mlo_cn2.dropna.Cn2_15m",
            "regression.usna_cn2_lg.full.Cn2_3m",
            "forecasting.usna_cn2_sm.full.Cn2_3m",
            "forecasting.paranal_tomography.full.cn2_profile_forecast",
    ):
        task = api.get_task(task_name)
        citation = task.get_citation()
        assert "@" in citation and len(citation) > 30

    scratch_nb = Path(task.get_dataset()._data_dir) / "paranal_tomography" / "build_and_validate_v2.ipynb"
    assert not scratch_nb.exists()


def test_paranal_tasks_register_all_ao_metrics():
    """Paranal profile tasks must include coherence_time and greenwood_frequency in tasks.json and experiments.json."""
    from otbench.tasks import TaskApi

    api = TaskApi()
    for task_name in (
            "regression.paranal_tomography.full.cn2_profile_reconstruction",
            "forecasting.paranal_tomography.full.cn2_profile_forecast",
    ):
        task = api.get_task(task_name)
        metrics_list = task.get_metric_names()
        assert "coherence_time" in metrics_list
        assert "greenwood_frequency" in metrics_list
        bench = task.get_benchmark_info()
        for model_name, m_info in bench.items():
            if model_name == "possible_predictions":
                continue
            assert "coherence_time" in m_info
            assert "greenwood_frequency" in m_info


def test_bootstrap_ci_and_evaluate_model_ci():
    """bootstrap_ci must support both i.i.d. and moving-block bootstrap and wire cleanly into evaluate_model."""
    from otbench.eval import bootstrap_ci
    from otbench.tasks import TaskApi

    rng = np.random.default_rng(7)
    y_true = rng.normal(-14.5, 0.4, size=(60, 7))
    y_pred = y_true + rng.normal(0.0, 0.15, size=(60, 7))

    low_iid, high_iid = bootstrap_ci(
        y_true,
        y_pred,
        metric_fn=lambda yt, yp: metrics.root_mean_square_error(yt, yp)["metric_value"],
        n_bootstrap=100,
        ci=0.95,
        block_size=1,
    )
    low_blk, high_blk = bootstrap_ci(
        y_true,
        y_pred,
        metric_fn=lambda yt, yp: metrics.root_mean_square_error(yt, yp)["metric_value"],
        n_bootstrap=100,
        ci=0.95,
        block_size=6,
    )
    point_est = metrics.root_mean_square_error(y_true, y_pred)["metric_value"]
    assert low_iid <= point_est <= high_iid
    assert low_blk <= point_est <= high_blk

    task = TaskApi().get_task("regression.paranal_tomography.full.cn2_profile_reconstruction")
    res = task.evaluate_model(
        predict_call=lambda X: np.full((len(X), 7), -14.8),
        eval_metric_names=["root_mean_square_error", "integrated_seeing"],
        compute_ci=True,
        n_bootstrap=40,
        bootstrap_block_size=5,
    )
    for m_name in ("root_mean_square_error", "integrated_seeing"):
        assert "ci_lower" in res[m_name] and "ci_upper" in res[m_name]
        assert res[m_name]["ci_lower"] <= res[m_name]["metric_value"] <= res[m_name]["ci_upper"]


def test_selective_lag_features_in_forecasting_task():
    """ForecastingTask.prepare_forecasting_data must lag only lag_features when specified."""
    from otbench.tasks import TaskApi

    task = TaskApi().get_task("forecasting.paranal_tomography.full.seeing_nowcast")
    X = pd.DataFrame({
        "fast_feat": np.arange(10, dtype=float),
        "slow_feat": np.arange(100, 110, dtype=float),
        "night_id": [1] * 10,
    })
    y = pd.Series(np.arange(10, dtype=float), name="seeing")

    X_lag, y_lag = task.prepare_forecasting_data(X, y, window_size=3, forecast_horizon=1, lag_features=["fast_feat"])
    assert "fast_feat (t-1)" in X_lag.columns
    assert "fast_feat (t-2)" in X_lag.columns
    assert "slow_feat" in X_lag.columns
    assert "slow_feat (t-1)" not in X_lag.columns
    assert len(X_lag) == len(y_lag) == 7


def _first_valid_contiguous_slice(X_df, y_df, min_rows: int = 120):
    """Return a slice of (X_df, y_df) guaranteed to contain contiguous non-NaN rows in both real and synthetic modes."""
    valid = (X_df.notna().all(axis=1) & y_df.notna().all(axis=1)).to_numpy()
    run_len = 20
    if len(valid) >= run_len:
        runs = np.convolve(valid.astype(int), np.ones(run_len, dtype=int), mode="valid")
        hits = np.flatnonzero(runs == run_len)
        if len(hits) > 0:
            start = int(hits[0])
            end = min(len(X_df), start + max(min_rows, 200))
            return X_df.iloc[start:end].copy(), y_df.iloc[start:end].copy()
    return X_df.iloc[:min_rows].copy(), y_df.iloc[:min_rows].copy()


def test_prepare_sequence_data_exact_parity_with_2d_lags():
    """ForecastingTask.prepare_sequence_data must match 2D prepare_forecasting_data + reshape to machine precision."""
    from otbench.tasks import TaskApi

    task = TaskApi().get_task("forecasting.paranal_tomography.full.cn2_profile_forecast")
    X_raw, y_raw = task.get_test_data(data_type="pd")
    X_sub, y_sub = _first_valid_contiguous_slice(X_raw, y_raw, min_rows=120)

    w, h = 6, 2
    X_2d, y_2d = task.prepare_forecasting_data(X_sub.copy(), y_sub.copy(), window_size=w, forecast_horizon=h)
    X_seq, y_seq, idx_seq = task.prepare_sequence_data(X_sub.copy(), y_sub.copy(), window_size=w, forecast_horizon=h)

    n_features = X_2d.shape[1] // w
    X_2d_reshaped = np.ascontiguousarray(X_2d.to_numpy().reshape(-1, w, n_features)[:, ::-1, :])

    assert X_seq.shape == X_2d_reshaped.shape
    np.testing.assert_allclose(X_seq, X_2d_reshaped, atol=1e-12)
    np.testing.assert_allclose(y_seq, y_2d.to_numpy(), atol=1e-12)
    assert idx_seq.equals(X_2d.index)


def test_evaluate_by_regime_and_horizon_curve():
    """evaluate_by_regime and evaluate_horizon_curve (+ plot_horizon_decay) must stratify and map horizon skill."""
    import matplotlib.pyplot as plt
    from otbench.tasks import TaskApi
    from otbench.plot.tomography import plot_horizon_decay

    task = TaskApi().get_task("forecasting.paranal_tomography.full.cn2_profile_forecast")
    rng = np.random.default_rng(99)
    y_true = rng.uniform(-15.5, -13.5, size=(80, 7))
    y_pred = y_true + rng.normal(0.0, 0.2, size=(80, 7))

    regimes = task.evaluate_by_regime(y_true, y_pred, eval_metric_names=["root_mean_square_error", "integrated_seeing"])
    assert "exquisite_seeing" in regimes
    assert "nominal_seeing" in regimes
    assert "turbulence_burst" in regimes
    total_frac = (regimes["exquisite_seeing"]["fraction"] + regimes["nominal_seeing"]["fraction"] +
                  regimes["turbulence_burst"]["fraction"])
    assert np.isclose(total_frac, 1.0)

    X_test, y_test = task.get_test_data(data_type="pd")
    X_small, y_small = _first_valid_contiguous_slice(X_test, y_test, min_rows=150)
    models = {
        "Constant_A": lambda X: np.full((len(X), 7), -14.8),
        "Constant_B": lambda X: np.full((len(X), 7), -14.5),
    }
    df_curve = task.evaluate_horizon_curve(
        models=models,
        horizons=[1, 3, 5],
        window_size=4,
        eval_metric_names=["root_mean_square_error", "per_layer_rmse"],
        reference_model="Constant_A",
        X_test=X_small,
        y_test=y_small,
    )
    assert set(df_curve["horizon"]) == {1, 3, 5}
    assert "skill_score" in df_curve.columns
    assert "rmse_0m" in df_curve.columns

    fig, ax = plt.subplots()
    plot_horizon_decay(df_curve, metric="skill_score", ax=ax)
    assert len(ax.get_lines()) >= 2
    plt.close(fig)


def test_ao_physics_loss_and_wavefront_error_budget():
    """AOPhysicsLoss must backpropagate finite gradients, and ao_wavefront_error_budget / strehl_ratio
    must compute valid physical variances, Strehl ratios in (0, 1], and normalized layer weights."""
    import torch
    from otbench.benchmark.models.forecasting.pytorch import AOPhysicsLoss
    from otbench.eval import ao_wavefront_error_budget, strehl_ratio

    loss_fn = AOPhysicsLoss(
        heights=[0, 500, 1000, 2000, 4000, 8000, 16000],
        lambda_r0=0.2,
        lambda_theta0=0.2,
    )
    loss_fn.set_normalization_stats(np.full(7, -14.5), np.full(7, 0.5))

    y_hat = torch.zeros((8, 7), dtype=torch.float32, requires_grad=True)
    y_tgt = torch.ones((8, 7), dtype=torch.float32) * 0.2
    loss = loss_fn(y_hat, y_tgt)
    loss.backward()
    assert torch.isfinite(loss)
    assert y_hat.grad is not None and torch.all(torch.isfinite(y_hat.grad))

    rng = np.random.default_rng(123)
    prof_true = rng.uniform(-15.5, -13.8, size=(20, 7))
    prof_pred = prof_true + rng.normal(0.0, 0.1, size=(20, 7))
    heights = [0, 500, 1000, 2000, 4000, 8000, 16000]

    budget = ao_wavefront_error_budget(prof_true, heights=heights)
    assert np.all((budget["strehl_ratio"] > 0.0) & (budget["strehl_ratio"] <= 1.0))
    np.testing.assert_allclose(
        budget["sigma2_total"],
        budget["sigma2_fitting"] + budget["sigma2_anisoplanatic"] + budget["sigma2_servo"],
    )
    np.testing.assert_allclose(np.sum(budget["layer_weights"], axis=1), np.ones(20))

    sr_res = strehl_ratio(prof_true, prof_pred, heights=heights, detailed=True)
    assert np.isfinite(sr_res["metric_value"]) and sr_res["metric_value"] >= 0.0
    assert sr_res["mean_opd_rms_error_nm"] >= 0.0
    assert sr_res["layer_weight_kl_divergence"] >= 0.0
    assert len(sr_res["strehl_true"]) == 20


def test_vectorized_bootstrap_ci_and_paired_significance():
    """Vectorized sample_losses bootstrap CI must match generic _bootstrap_ci at identical seed,
    and paired_bootstrap_test / task.compare_models must quantify paired statistical significance."""
    import otbench
    from otbench.eval import paired_bootstrap_test, root_mean_square_error
    from otbench.eval.utils import _bootstrap_ci
    from otbench.tasks import TaskApi

    assert otbench.__version__ == "0.5.0"

    rng = np.random.default_rng(2026)
    y_true = rng.uniform(-15.5, -13.5, size=(120, 7))
    y_pred_a = y_true + rng.normal(0.0, 0.08, size=(120, 7))
    y_pred_b = y_true + rng.normal(0.0, 0.35, size=(120, 7))

    # Verify vectorized sample_losses path matches generic metric_fn loop at identical seed
    sq_err = (y_true - y_pred_a)**2
    ci_vec = _bootstrap_ci(
        y_true,
        y_pred_a,
        lambda yt, yp: root_mean_square_error(yt, yp, detailed=False),
        n_bootstrap=200,
        block_size=10,
        seed=42,
        sample_losses=sq_err,
        reduction="rmse",
    )
    ci_loop = _bootstrap_ci(
        y_true,
        y_pred_a,
        lambda yt, yp: root_mean_square_error(yt, yp, detailed=False),
        n_bootstrap=200,
        block_size=10,
        seed=42,
    )
    np.testing.assert_allclose(ci_vec[0], ci_loop[0], rtol=1e-12)
    np.testing.assert_allclose(ci_vec[1], ci_loop[1], rtol=1e-12)

    # Paired bootstrap test directly and via task.compare_models (including greenwood_frequency)
    from otbench.eval import greenwood_frequency

    task = TaskApi().get_task("forecasting.paranal_tomography.full.cn2_profile_forecast")
    comparison = task.compare_models(
        y_true,
        y_pred_a,
        y_pred_b,
        eval_metric_names=["root_mean_square_error", "integrated_seeing", "greenwood_frequency"],
        n_bootstrap=500,
        block_size=10,
        seed=42,
    )
    rmse_cmp = comparison["root_mean_square_error"]
    assert rmse_cmp["delta"] < 0.0
    assert rmse_cmp["ci_upper"] < 0.0
    assert rmse_cmp["p_value"] < 0.05
    assert rmse_cmp["significant"] is True
    assert rmse_cmp["a_is_better"] is True

    fg_kw = task._build_metric_kwargs("greenwood_frequency")
    fg_expected_a = greenwood_frequency(y_true, y_pred_a, detailed=False, **fg_kw)["metric_value"]
    fg_expected_b = greenwood_frequency(y_true, y_pred_b, detailed=False, **fg_kw)["metric_value"]
    np.testing.assert_allclose(comparison["greenwood_frequency"]["score_a"], fg_expected_a, rtol=1e-10)
    np.testing.assert_allclose(comparison["greenwood_frequency"]["score_b"], fg_expected_b, rtol=1e-10)
    assert comparison["greenwood_frequency"]["a_is_better"] is True

    # Also verify 3-argument (horizon, X_h, y_h) callable support in evaluate_horizon_curve
    X_test_raw, y_test_raw = task.get_test_data(data_type="pd")
    X_sub, y_sub = _first_valid_contiguous_slice(X_test_raw, y_test_raw, min_rows=90)
    seen_horizons = []

    def _three_arg_predictor(h, X_h, y_h):
        seen_horizons.append(h)
        return y_h.to_numpy() * 0.99

    h_df = task.evaluate_horizon_curve(
        models={"ThreeArgOracle": _three_arg_predictor},
        horizons=(1, 3),
        window_size=5,
        eval_metric_names=["root_mean_square_error"],
        X_test=X_sub,
        y_test=y_sub,
    )
    assert seen_horizons == [1, 3]
    assert len(h_df) == 2


def test_pytorch_3d_sequence_dispatch_parity():
    """RNNModel and TransformerModel must natively train and predict on 3D (N, W, F) sequence tensors
    with predict_residuals=True and use_log10=True, producing identical inference output on 3D and 2D inputs."""
    import torch
    from otbench.tasks import TaskApi
    from otbench.benchmark.models.forecasting.pytorch import RNNModel, TransformerModel

    task = TaskApi().get_task("forecasting.paranal_tomography.full.cn2_profile_forecast")
    X_train_raw, y_train_raw = task.get_train_data(data_type="pd")
    X_sub, y_sub = _first_valid_contiguous_slice(X_train_raw, y_train_raw, min_rows=120)

    X_seq, y_seq, _ = task.prepare_sequence_data(X_sub, y_sub)
    X_2d, y_2d = task.prepare_forecasting_data(X_sub, y_sub)
    feature_names = [c for c in X_sub.columns if c.lower() not in ("night_id", "session_id")]

    for ModelCls in (RNNModel, TransformerModel):
        torch.manual_seed(7)
        mdl = ModelCls(
            name=ModelCls.__name__,
            target_name=list(y_sub.columns),
            input_size=len(feature_names),
            output_size=7,
            window_size=task.window_size,
            forecast_horizon=task.forecast_horizon,
            feature_names=feature_names,
            predict_residuals=True,
            use_log10=True,
            n_epochs=2,
            batch_size=32,
        )
        mdl.train(X_seq, y_seq, feature_names=feature_names)
        pred_3d = mdl.predict(X_seq)
        pred_2d = mdl.predict(X_2d)
        assert pred_3d.shape == y_seq.shape
        np.testing.assert_allclose(pred_3d, pred_2d, atol=1e-5)

    # Also verify task.evaluate_model(..., use_sequence_data=True)
    res_seq = task.evaluate_model(
        predict_call=mdl.predict,
        eval_metric_names=["root_mean_square_error"],
        X_test=X_sub,
        y_test=y_sub,
        use_sequence_data=True,
    )
    res_2d = task.evaluate_model(
        predict_call=mdl.predict,
        eval_metric_names=["root_mean_square_error"],
        X_test=X_sub,
        y_test=y_sub,
        use_sequence_data=False,
    )
    np.testing.assert_allclose(
        res_seq["root_mean_square_error"]["metric_value"],
        res_2d["root_mean_square_error"]["metric_value"],
        atol=1e-5,
    )


def test_ao_physics_loss_with_residual_learning():
    """AOPhysicsLoss must reconstruct physical log10(J_i) = y_base + delta_y when predict_residuals=True,
    preventing clamp saturation and producing non-zero r0/theta0 gradients during 2D and 3D training."""
    import torch
    from otbench.tasks import TaskApi
    from otbench.benchmark.models.forecasting.pytorch import AOPhysicsLoss, RNNModel, TransformerModel

    heights = [0, 500, 1000, 2000, 4000, 8000, 16000]
    loss_fn = AOPhysicsLoss(heights=heights, lambda_r0=1.0, lambda_theta0=1.0)
    # Residual normalization stats: mean ~ 0.0, std ~ 0.2 dex
    loss_fn.set_normalization_stats(np.zeros(7), np.full(7, 0.2))

    # Normalized residuals near 0 -> un-normalized residuals near 0.0 dex
    # Physical base persistence profile near -14.8 log10(m^(1/3))
    y_hat = torch.zeros((8, 7), dtype=torch.float32, requires_grad=True)
    y_tgt = torch.full((8, 7), 0.5, dtype=torch.float32)  # +0.1 dex residual target
    y_base = torch.full((8, 7), -14.8, dtype=torch.float32)

    # Pure MSE component (lambda_r0=0, lambda_theta0=0) vs full physics loss with y_base
    mse_only = torch.nn.functional.mse_loss(y_hat, y_tgt)
    total_loss = loss_fn(y_hat, y_tgt, y_base=y_base)
    total_loss.backward()

    assert total_loss.item() > mse_only.item(), "Optical moment penalties (r0, theta0) must be strictly positive"
    assert y_hat.grad is not None and torch.all(torch.isfinite(y_hat.grad))
    # Verify gradients differ from pure MSE gradient (-2*(y_tgt - y_hat)/N)
    mse_grad = 2.0 * (y_hat.detach() - y_tgt) / y_hat.numel()
    assert not torch.allclose(y_hat.grad, mse_grad), "r0/theta0 physics terms must contribute non-zero gradients"

    # End-to-end training with RNNModel and TransformerModel under predict_residuals=True + AOPhysicsLoss
    task = TaskApi().get_task("forecasting.paranal_tomography.full.cn2_profile_forecast")
    X_train_raw, y_train_raw = task.get_train_data(data_type="pd")
    X_sub, y_sub = _first_valid_contiguous_slice(X_train_raw, y_train_raw, min_rows=100)
    X_seq, y_seq, _ = task.prepare_sequence_data(X_sub, y_sub)
    X_2d, y_2d = task.prepare_forecasting_data(X_sub, y_sub)
    feature_names = [c for c in X_sub.columns if c.lower() not in ("night_id", "session_id")]

    for ModelCls in (RNNModel, TransformerModel):
        crit_3d = AOPhysicsLoss(heights=heights, lambda_r0=0.25, lambda_theta0=0.25)
        mdl_3d = ModelCls(
            name=f"{ModelCls.__name__}_Phys3D",
            target_name=list(y_sub.columns),
            input_size=len(feature_names),
            output_size=7,
            window_size=task.window_size,
            forecast_horizon=task.forecast_horizon,
            feature_names=feature_names,
            criterion=crit_3d,
            predict_residuals=True,
            use_log10=True,
            n_epochs=2,
            batch_size=32,
            val_fraction=0.2,
        )
        mdl_3d.train(X_seq, y_seq, feature_names=feature_names)
        preds_3d = mdl_3d.predict(X_seq)
        assert preds_3d.shape == y_seq.shape
        assert np.all(np.isfinite(preds_3d))
        assert np.all((preds_3d > -20.0) & (preds_3d < -10.0))

    crit_2d = AOPhysicsLoss(heights=heights, lambda_r0=0.25, lambda_theta0=0.25)
    mdl_2d = RNNModel(
        name="RNN_Phys2D",
        target_name=list(y_sub.columns),
        input_size=len(feature_names),
        output_size=7,
        window_size=task.window_size,
        forecast_horizon=task.forecast_horizon,
        criterion=crit_2d,
        predict_residuals=True,
        use_log10=True,
        n_epochs=2,
        batch_size=32,
        val_fraction=0.2,
    )
    mdl_2d.train(X_2d, y_2d)
    preds_2d = mdl_2d.predict(X_2d)
    assert preds_2d.shape == y_2d.shape
    assert np.all(np.isfinite(preds_2d))


def test_compare_models_joint_mask_and_sequence_horizon_curve():
    """compare_models must apply a single joint validity mask across (y_true, y_pred_a, y_pred_b),
    and evaluate_horizon_curve must support use_sequence_data=True for 3D sequence models."""
    from otbench.tasks import TaskApi

    task = TaskApi().get_task("forecasting.paranal_tomography.full.cn2_profile_forecast")
    rng = np.random.default_rng(314)
    y_true = rng.uniform(-15.5, -13.5, size=(60, 7))
    y_pred_a = y_true + rng.normal(0.0, 0.05, size=(60, 7))
    y_pred_b = y_true + rng.normal(0.0, 0.25, size=(60, 7))

    # Inject disjoint NaNs into y_pred_a and y_pred_b
    y_pred_a[3, 0] = np.nan
    y_pred_b[11, 2] = np.nan
    y_true[25, 4] = np.nan

    cmp_res = task.compare_models(
        y_true,
        y_pred_a,
        y_pred_b,
        eval_metric_names=["root_mean_square_error", "mean_absolute_error", "integrated_seeing"],
        n_bootstrap=100,
        block_size=5,
        seed=42,
    )
    for m_name in ("root_mean_square_error", "mean_absolute_error", "integrated_seeing"):
        assert np.isfinite(cmp_res[m_name]["delta"])
        assert cmp_res[m_name]["a_is_better"] is True

    # Verify evaluate_horizon_curve with use_sequence_data=True
    X_test_raw, y_test_raw = task.get_test_data(data_type="pd")
    X_sub, y_sub = _first_valid_contiguous_slice(X_test_raw, y_test_raw, min_rows=90)
    observed_shapes = []

    def _seq_model_fn(X_3d):
        observed_shapes.append(X_3d.shape)
        assert X_3d.ndim == 3
        return np.full((X_3d.shape[0], 7), -14.7)

    df_seq_curve = task.evaluate_horizon_curve(
        models={"SeqConstant": _seq_model_fn},
        horizons=(1, 2),
        window_size=4,
        eval_metric_names=["root_mean_square_error"],
        X_test=X_sub,
        y_test=y_sub,
        use_sequence_data=True,
    )
    assert len(df_seq_curve) == 2
    assert all(len(s) == 3 and s[1] == 4 for s in observed_shapes)


def test_solar_features_sunrise_to_sunset_monotonicity_and_boundary_weights():
    """add_temporal_hour must vary across the solar day (0 at sunrise to 12 at sunset) without
    mutating caller DataFrame, and add_temporal_hour_weight must cover integer boundaries."""
    from otbench.utils.solar_features import add_temporal_hour, add_temporal_hour_weight

    idx = pd.date_range("2021-06-21 06:00:00", "2021-06-21 18:00:00", freq="2h")
    X_orig = pd.DataFrame({"T_2m": np.linspace(15.0, 25.0, len(idx))}, index=idx)
    X_th = add_temporal_hour(
        X_orig,
        name="test_site",
        timezone="UTC",
        obs_lat=0.0,
        obs_lon=0.0,
        time_col_name="time",
        temporal_hour_col_name="temporal_hour",
    )
    assert "temporal_hour" not in X_orig.columns, "add_temporal_hour must not mutate input DataFrame"
    th_vals = X_th["temporal_hour"].to_numpy()
    assert np.all(np.diff(th_vals) > 0), "Temporal hour must strictly increase with time of day"
    assert np.isclose(th_vals[0], 0.0, atol=0.25)
    assert np.isclose(th_vals[-1], 12.0, atol=0.25)

    X_bins = pd.DataFrame({"temporal_hour": [-100.0, -4.0, 0.0, 5.0, 12.0, 13.0, 120.0, np.nan]})
    X_w = add_temporal_hour_weight(
        X_bins,
        temporal_hour_col_name="temporal_hour",
        temporal_hour_weight_col_name="temporal_hour_weight",
    )
    assert "temporal_hour_weight" not in X_bins.columns
    w_vals = X_w["temporal_hour_weight"].to_numpy()
    np.testing.assert_allclose(w_vals[:-1], [0.11, 0.11, 0.05, 1.0, 0.08, 0.13, 0.13])
    assert np.isnan(w_vals[-1])


def test_macro_meteorological_single_row_inference_and_non_mutation():
    """MacroMeteorologicalModel and OffshoreMacroMeteorologicalModel must freeze constant_adjustment
    from training data so single-row inference matches batch inference, without mutating X."""
    from otbench.benchmark.models.regression.macro_meteorological import MacroMeteorologicalModel
    from otbench.benchmark.models.regression.offshore_macro_meteorological import OffshoreMacroMeteorologicalModel

    X_train = pd.DataFrame({
        "T_2m": [15.0, 18.0, 20.0, 22.0],
        "Spd_10m": [2.0, 4.0, 5.0, 7.0],
        "RH_2m": [40.0, 50.0, 60.0, 70.0],
        "time": [0, 1, 2, 3],
        "temporal_hour": [0.0, 2.0, 5.0, 8.0],
        "temporal_hour_weight": [0.05, 0.51, 1.0, 0.59],
    })
    y_train = pd.DataFrame({"Cn2_15m": [-14.2, -14.0, -13.8, -14.1]})

    for ModelCls in (MacroMeteorologicalModel, OffshoreMacroMeteorologicalModel):
        mdl = ModelCls(
            name="mm_test",
            target_name="Cn2_15m",
            timezone="UTC",
            obs_lat=0.0,
            obs_lon=0.0,
            air_temperature_col_name="T_2m",
            wind_speed_col_name="Spd_10m",
            humidity_col_name="RH_2m",
            time_col_name="time",
            height_of_observation=15.0,
            use_log10=True,
        )
        mdl.train(X_train, y_train)
        X_copy = X_train.copy(deep=True)
        batch_preds = mdl.predict(X_train)
        pd.testing.assert_frame_equal(X_train, X_copy)
        for i in range(len(X_train)):
            single_pred = mdl.predict(X_train.iloc[[i]])
            np.testing.assert_allclose(single_pred[0], batch_preds[i], rtol=1e-12)


def test_height_adjustment_non_mutation_and_validation():
    """Height adjustment utilities must not mutate caller inputs and must reject observed <= 0."""
    from otbench.utils.height_adjustment import apply_fried_height_adjustment, apply_oermann_height_adjustment

    arr = np.array([1e-14, 2e-14])
    arr_orig = arr.copy()
    out_fried = apply_fried_height_adjustment(arr, observed=15.0, desired=30.0)
    out_oermann = apply_oermann_height_adjustment(arr, desired=30.0, observed=15.0)
    np.testing.assert_array_equal(arr, arr_orig)
    assert not np.allclose(out_fried, arr_orig, atol=0.0)
    assert not np.allclose(out_oermann, arr_orig, atol=0.0)

    with pytest.raises(ValueError, match="positive"):
        apply_fried_height_adjustment(arr, observed=0.0, desired=15.0)


def test_integrated_metrics_density_mode_log10_and_domain_mismatch_warning():
    """Integrated AO metrics must support values_are_integrals=False (Cn2 density mode) on both
    linear and log10 profiles, and warn when y_true and y_pred domains mismatch."""
    from otbench.eval.integrated_metrics import (
        isoplanatic_angle,
        coherence_time,
        greenwood_frequency,
        ao_wavefront_error_budget,
        strehl_ratio,
    )

    heights = [0.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
    cn2_lin = np.full((5, 7), 1e-16)
    cn2_log = np.log10(cn2_lin)

    budget_lin = ao_wavefront_error_budget(cn2_lin, heights=heights, values_are_integrals=False)
    budget_log = ao_wavefront_error_budget(cn2_log, heights=heights, values_are_integrals=False)
    np.testing.assert_allclose(budget_lin["r0_m"], budget_log["r0_m"])
    np.testing.assert_allclose(budget_lin["theta0_arcsec"], budget_log["theta0_arcsec"])
    np.testing.assert_allclose(budget_lin["strehl_ratio"], budget_log["strehl_ratio"])

    for fn in (isoplanatic_angle, coherence_time, greenwood_frequency, strehl_ratio):
        res_lin = fn(cn2_lin, cn2_lin * 1.1, heights=heights, values_are_integrals=False)
        res_log = fn(cn2_log, np.log10(cn2_lin * 1.1), heights=heights, values_are_integrals=False)
        np.testing.assert_allclose(res_lin["metric_value"], res_log["metric_value"])

    with pytest.warns(UserWarning, match="Domain mismatch"):
        res_mixed = isoplanatic_angle(cn2_log, cn2_lin, heights=heights, values_are_integrals=True)
    assert np.isclose(res_mixed["metric_value"], 0.0)


def test_get_valid_indices_pandas_index_mismatch_raises():
    """_get_valid_indices must raise ValueError when pandas inputs have misaligned indices."""
    from otbench.eval.utils import _get_valid_indices

    s1 = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
    s2 = pd.Series([1.0, 2.0, 3.0], index=[10, 11, 12])
    with pytest.raises(ValueError, match="mismatched indices"):
        _get_valid_indices(s1, s2)


def test_top_models_and_leaderboard_metric_orientation(tmp_path):
    """top_models and leaderboard must rank higher-is-better metrics (R2) descending and push NaNs last."""
    import json
    from otbench.eval import is_lower_better
    from otbench.tasks import TaskApi

    assert is_lower_better("root_mean_square_error") is True
    assert is_lower_better("coefficient_of_determination") is False

    bench_fp = tmp_path / "experiments.json"
    task_name = "regression.mlo_cn2.dropna.Cn2_15m"
    mock_data = {
        task_name: {
            "possible_predictions": 100,
            "ModelLowR2": {
                "root_mean_square_error": {
                    "metric_value": 0.50,
                    "valid_predictions": 100
                },
                "coefficient_of_determination": {
                    "metric_value": 0.25,
                    "valid_predictions": 100
                },
            },
            "ModelHighR2": {
                "root_mean_square_error": {
                    "metric_value": 0.20,
                    "valid_predictions": 100,
                    "ci_lower": 0.18,
                    "ci_upper": 0.22
                },
                "coefficient_of_determination": {
                    "metric_value": 0.85,
                    "valid_predictions": 100
                },
            },
            "ModelNaN": {
                "root_mean_square_error": {
                    "metric_value": float("nan"),
                    "valid_predictions": 0
                },
                "coefficient_of_determination": {
                    "metric_value": float("nan"),
                    "valid_predictions": 0
                },
            },
        }
    }
    bench_fp.write_text(json.dumps(mock_data))
    task = TaskApi().get_task(task_name, benchmark_fp=str(bench_fp))

    top_r2 = [k for k in task.top_models(n=3, metric="coefficient_of_determination") if k != "possible_predictions"]
    assert top_r2 == ["ModelHighR2", "ModelLowR2", "ModelNaN"]

    top_rmse = [k for k in task.top_models(n=3, metric="root_mean_square_error") if k != "possible_predictions"]
    assert top_rmse == ["ModelHighR2", "ModelLowR2", "ModelNaN"]

    lb = task.leaderboard(metric="coefficient_of_determination", include_ci=True)
    assert list(lb.index) == ["ModelHighR2", "ModelLowR2", "ModelNaN"]
    assert "root_mean_square_error_ci_lower" in lb.columns


def test_forecasting_contiguous_sessions_prepared_data_and_estimator_eval():
    """ForecastingTask must split non-contiguous repeated session IDs into separate runs,
    provide get_prepared_data(), and accept estimator instances in evaluate_model()."""
    from otbench.tasks import TaskApi
    from otbench.tasks.tasks import ForecastingTask
    from otbench.benchmark.models.forecasting.persistence import PersistenceForecastingModel

    sessions = pd.Series([1, 1, 1, np.nan, 2, 2, 1, 1])
    runs = ForecastingTask._get_session_group_positions(sessions)
    assert len(runs) == 3
    np.testing.assert_array_equal(runs[0], [0, 1, 2])
    np.testing.assert_array_equal(runs[1], [4, 5])
    np.testing.assert_array_equal(runs[2], [6, 7])

    task = TaskApi().get_task("forecasting.paranal_tomography.full.cn2_profile_forecast")
    X_tr, y_tr = task.get_train_data(prepared=True)
    X_tr2, y_tr2 = task.get_prepared_data(split="train")
    pd.testing.assert_frame_equal(X_tr, X_tr2)
    pd.testing.assert_frame_equal(y_tr, y_tr2)

    X_seq, y_seq = task.get_prepared_data(split="train", use_sequence_data=True)
    assert X_seq.ndim == 3 and len(X_seq) == len(X_tr)

    mdl = PersistenceForecastingModel(
        name="PersDirect",
        target_name=task.get_target_name(),
        window_size=task.window_size,
        forecast_horizon=task.forecast_horizon,
    )
    mdl.train(X_tr, y_tr)
    X_raw_test, y_raw_test = task.get_test_data()
    X_sub, y_sub = _first_valid_contiguous_slice(X_raw_test, y_raw_test, min_rows=100)
    res_obj = task.evaluate_model(mdl, X_test=X_sub, y_test=y_sub)
    res_fn = task.evaluate_model(mdl.predict, X_test=X_sub, y_test=y_sub)
    assert res_obj["root_mean_square_error"]["metric_value"] == res_fn["root_mean_square_error"]["metric_value"]


def test_pytorch_all_nan_feature_normalization_and_bench_runner_validation():
    """PyTorch _normalize_data must safely zero-fill all-NaN feature columns, and
    _resolve_model_map must validate unknown model names while accepting custom classes."""
    from otbench.benchmark.models.regression.pytorch.recurrent_neural_network import RNNModel
    from otbench.benchmark.bench_runner import _resolve_model_map
    from otbench.benchmark.models.regression.persistence import PersistenceRegressionModel

    X = np.random.default_rng(0).normal(size=(16, 1, 3))
    X[:, :, 1] = np.nan  # Entire feature 1 is NaN
    y = np.random.default_rng(1).normal(size=(16, 1))

    mdl = RNNModel(name="nan_col_test", target_name="t", input_size=3, output_size=1, n_epochs=1)
    X_norm, y_norm = mdl._normalize_data(X, y)
    assert np.all(np.isfinite(X_norm))
    assert np.all(X_norm[:, :, 1] == 0.0)
    assert np.all(np.isfinite(y_norm))

    with pytest.raises(ValueError, match="Unknown regression model"):
        _resolve_model_map(["NonExistentModel"], {"PersistenceRegressionModel": PersistenceRegressionModel},
                           kind="regression")

    resolved = _resolve_model_map([PersistenceRegressionModel], {}, kind="regression")
    assert resolved["PersistenceRegressionModel"] is PersistenceRegressionModel


def test_fast_fail_height_and_horizon_contracts():
    """Verify _validate_profile_heights and ForecastingTask reject negative, unsorted, or invalid parameters."""
    from otbench.eval.integrated_metrics import profile_to_fried_parameter, isoplanatic_angle
    from otbench.tasks import TaskApi

    prof = np.full((4, 3), -14.0)
    with pytest.raises(ValueError, match="non-negative"):
        profile_to_fried_parameter(prof, heights=[-100.0, 500.0, 1000.0])

    with pytest.raises(ValueError, match="monotonically non-decreasing"):
        isoplanatic_angle(prof, prof, heights=[0.0, 1000.0, 500.0])

    task = TaskApi().get_task("forecasting.paranal_tomography.full.cn2_profile_forecast")
    X_raw, y_raw = task.get_test_data()
    with pytest.raises(ValueError, match="window_size must be >= 1"):
        task.prepare_forecasting_data(X_raw.iloc[:20], y_raw.iloc[:20], window_size=0)
    with pytest.raises(ValueError, match="forecast_horizon must be >= 0"):
        task.prepare_forecasting_data(X_raw.iloc[:20], y_raw.iloc[:20], forecast_horizon=-1)

"""Deterministic artifact generator for the otb2 manuscript (Paranal Tomographic Forecasting Benchmark).

Computes and persists (with zero synthetic or rescaled surrogates):
1. Complete benchmark metrics in ``otbench/benchmark/experiments.json`` and ``tests/benchmark/experiments.json``
   across all 1-year (``full``) and 5-year (``multiannual``) Paranal benchmark tasks.
2. Second-order LogSumExp Jensen convexity correction ablation on ``AOPhysicsLoss`` (``jensen_correction=True``
   vs. ``jensen_correction=False`` vs. ``Transformer (MSE)``).
3. Genuine multi-horizon forecasting sweep across H in {1, 5, 10, 15, 30, 60} min using uniform macro-RMSE
   so H=10 min is 100% identical to the Task 3 leaderboard.
4. Multi-modal sensor ablation comparing Optical-Only (7 vars) vs. Optical+AWS (11 vars) vs. Full
   Optical+AWS+LHATPRO (50 vars) across forecast horizons.
5. Dynamic wind-coupled temporal AO metrics (coherence time tau_0(U_t) and Greenwood frequency f_G(U_t))
   evaluated with contemporaneous 30-m Vaisala AWS wind speed alongside the 10 m/s reference baseline.
6. All four publication figures in ``doc/manuscirpts/otb2/``:
   - ``paranal_climatology.png`` (Figure 1: 4-panel Paranal vertical & nocturnal climatology)
   - ``profile_error_heatmaps.png`` (Figure 2: 2-panel Model x Layer heatmap + HV-5/7 profile comparison)
   - ``horizon_skill_decay.png`` (Figure 3: 3-panel multi-horizon forecast decay H in [1, 60] min)
   - ``burst_case_study.png`` (Figure 4: 3-panel turbulence burst & upper-tropospheric shear case study)
7. Exact empirical summaries for all tables of ``doc/manuscirpts/otb2/manuscript.tex`` in
   ``doc/manuscirpts/otb2/empirical_summary.json``.
"""
import gc
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

from otbench.benchmark.bench_runner import NumpyEncoder, run_benchmarks
from otbench.benchmark.models.forecasting import (
    ClimatologyForecastingModel,
    GradientBoostingForecastingModel,
    LinearForecastingModel,
    MeanWindowForecastingModel,
    MinuteClimatologyForecastingModel,
    PersistenceForecastingModel,
)
from otbench.benchmark.models.forecasting.pytorch import (
    GRUModel,
    LSTMModel,
    PhysicsTransformerModel,
    RNNModel as ForecastingRNNModel,
    TransformerModel,
)
from otbench.benchmark.models.regression import (
    GradientBoostingRegressionModel,
    MinuteClimatologyRegressionModel,
)
from otbench.benchmark.models.regression.pytorch import RNNModel as RegressionRNNModel
from otbench.eval import integrated_metrics, metrics
from otbench.plot.tomography import (
    compute_mass_slab_thicknesses,
    plot_burst_case_study,
    plot_horizon_decay,
    plot_paranal_climatology,
    plot_profile_comparison,
    plot_profile_error_heatmap,
)
from otbench.tasks import TaskApi, tasks

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "doc" / "manuscirpts" / "otb2"
BENCH_FP = ROOT / "otbench" / "benchmark" / "experiments.json"
TEST_BENCH_FP = ROOT / "tests" / "benchmark" / "experiments.json"
HEIGHTS = [0.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0]
LAYER_COLS = [
    "cn2_ground_scalar",
    "cn2_free_atmos_500",
    "cn2_free_atmos_1000",
    "cn2_free_atmos_2000",
    "cn2_free_atmos_4000",
    "cn2_free_atmos_8000",
    "cn2_free_atmos_16000",
]
AWS_COLS = ["wind_speed", "wind_dir", "pressure", "rh"]
LAYER_LABELS = ["0 m (GL)", "500 m", "1 km", "2 km", "4 km", "8 km", "16 km"]


def _free_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _save_bench(persisted: dict):
    with open(BENCH_FP, "w") as f:
        f.write(json.dumps(persisted, indent=4, cls=NumpyEncoder))
    with open(TEST_BENCH_FP, "w") as f:
        f.write(json.dumps(persisted, indent=4, cls=NumpyEncoder))


def compute_dataset_climatology() -> dict:
    """Compute Table 1 summary statistics across the 5-year Paranal NetCDF archive."""
    nc_fp = ROOT / "otbench" / "data" / "paranal_tomography" / "paranal_tomography.nc"
    ds = xr.open_dataset(nc_fp)
    n_total = int(ds.sizes["time"])
    stats = {
        "n_total": n_total,
        "n_nights": int(len(np.unique(ds["night_id"].values))),
        "slab_thicknesses_m": [round(float(x), 4) for x in compute_mass_slab_thicknesses(HEIGHTS)],
        "vars": {},
    }

    # Seeing
    seeing = ds["seeing"].values
    valid_s = seeing[~np.isnan(seeing)]
    q25, q50, q75 = np.percentile(valid_s, [25, 50, 75])
    stats["vars"]["seeing"] = {
        "coverage_pct": 100.0 * len(valid_s) / n_total,
        "median": float(q50),
        "q25": float(q25),
        "q75": float(q75),
    }

    # Ground layer J_0
    gl = ds["cn2_ground_scalar"].values
    valid_gl = gl[~np.isnan(gl) & (gl > 0)]
    log_gl = np.log10(np.maximum(valid_gl, 1e-19))
    q25, q50, q75 = np.percentile(log_gl, [25, 50, 75])
    stats["vars"]["cn2_ground_scalar"] = {
        "coverage_pct": 100.0 * len(valid_gl) / n_total,
        "median": float(q50),
        "q25": float(q25),
        "q75": float(q75),
    }

    # Free atmosphere layers
    fa = ds["cn2_free_atmos"].values
    for idx, h in enumerate([500, 1000, 2000, 4000, 8000, 16000]):
        col = fa[:, idx]
        valid_c = col[~np.isnan(col) & (col > 0)]
        log_c = np.log10(np.maximum(valid_c, 1e-19))
        q25, q50, q75 = np.percentile(log_c, [25, 50, 75])
        stats["vars"][f"cn2_free_atmos_{h}"] = {
            "coverage_pct": 100.0 * len(valid_c) / n_total,
            "median": float(q50),
            "q25": float(q25),
            "q75": float(q75),
        }

    # LHATPRO surface (0 m) and 10 km temperature
    tp = ds["temp_profile"].values
    for idx, label in [(0, "temp_0m"), (38, "temp_10000m")]:
        col = tp[:, idx]
        valid_t = col[~np.isnan(col)]
        q25, q50, q75 = np.percentile(valid_t, [25, 50, 75])
        stats["vars"][label] = {
            "coverage_pct": 100.0 * len(valid_t) / n_total,
            "median": float(q50),
            "q25": float(q25),
            "q75": float(q75),
        }

    # Meteo variables
    for mvar in ["wind_speed", "wind_dir", "pressure", "rh"]:
        col = ds[mvar].values
        valid_m = col[~np.isnan(col)]
        q25, q50, q75 = np.percentile(valid_m, [25, 50, 75])
        stats["vars"][mvar] = {
            "coverage_pct": 100.0 * len(valid_m) / n_total,
            "median": float(q50),
            "q25": float(q25),
            "q75": float(q75),
        }
    ds.close()
    return stats


def _compute_full_profile_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    wind_speed_te: np.ndarray = None,
) -> dict:
    """Compute complete layer-wise, integrated AO, Strehl/OPD/KL, GLF, bias, and dynamic-wind metrics."""
    s_dict = integrated_metrics.strehl_ratio(y_true, y_pred, heights=HEIGHTS)
    glf_dict = integrated_metrics.ground_layer_fraction(y_true, y_pred, heights=HEIGHTS)
    plr_dict = metrics.per_layer_rmse(y_true, y_pred, layer_names=LAYER_LABELS)
    bias_per_layer = np.nanmean(y_pred - y_true, axis=0)
    res = {
        "rmse": float(metrics.root_mean_square_error(y_true, y_pred)["metric_value"]),
        "mae": float(metrics.mean_absolute_error(y_true, y_pred)["metric_value"]),
        "seeing": float(integrated_metrics.integrated_seeing(y_true, y_pred, heights=HEIGHTS)["metric_value"]),
        "isoplanatic": float(integrated_metrics.isoplanatic_angle(y_true, y_pred, heights=HEIGHTS)["metric_value"]),
        "coherence_time": float(integrated_metrics.coherence_time(y_true, y_pred, heights=HEIGHTS)["metric_value"]),
        "greenwood": float(integrated_metrics.greenwood_frequency(y_true, y_pred, heights=HEIGHTS)["metric_value"]),
        "strehl": float(s_dict["metric_value"]),
        "opd_nm": float(s_dict["mean_opd_rms_error_nm"]),
        "kl_div": float(s_dict["layer_weight_kl_divergence"]),
        "glf_rmse": float(glf_dict["metric_value"]),
        "mean_glf_true": float(glf_dict["mean_glf_true"]),
        "mean_glf_pred": float(glf_dict["mean_glf_pred"]),
        "per_layer": {
            k: float(v) for k, v in plr_dict["per_layer"].items()
        },
        "per_layer_bias": {
            LAYER_LABELS[i]: float(bias_per_layer[i]) for i in range(len(LAYER_LABELS))
        },
        "mean_bias": float(np.mean(bias_per_layer)),
    }
    if wind_speed_te is not None:
        res["coherence_time_dynamic_wind"] = float(
            integrated_metrics.coherence_time(y_true, y_pred, heights=HEIGHTS,
                                              wind_speed=wind_speed_te)["metric_value"])
        res["greenwood_dynamic_wind"] = float(
            integrated_metrics.greenwood_frequency(y_true, y_pred, heights=HEIGHTS,
                                                   wind_speed=wind_speed_te)["metric_value"])
    return res


def _prepare_chunked_seq(task, X_df: pd.DataFrame, y_df: pd.DataFrame, stride: int = 1, chunk_rows: int = 500000):
    """Prepare 3D sequence data in bounded row chunks aligned on observing-night boundaries."""
    X_parts, y_parts, idx_parts = [], [], []
    n_total = len(X_df)
    session_col = task.task.get("session_col")
    sessions = None
    if session_col:
        if session_col in X_df.columns:
            sessions = X_df[session_col].to_numpy()
        else:
            try:
                sessions = task.get_dataset().get_context(X_df.index, session_col)[session_col].to_numpy()
            except Exception:
                sessions = None

    start = 0
    while start < n_total:
        end = min(n_total, start + chunk_rows)
        if sessions is not None and end < n_total:
            boundary = end
            while boundary > start + 1 and sessions[boundary - 1] == sessions[boundary]:
                boundary -= 1
            if boundary > start:
                end = boundary
        Xc, yc, idxc = task.prepare_sequence_data(X_df.iloc[start:end], y_df.iloc[start:end])
        if len(Xc) > 0:
            X_parts.append(Xc[::stride].astype(np.float32, copy=False))
            y_parts.append(yc[::stride].astype(np.float32, copy=False))
            idx_parts.append(idxc[::stride])
        del Xc, yc
        _free_mem()
        start = end
    X_out = np.concatenate(X_parts, axis=0)
    y_out = np.concatenate(y_parts, axis=0)
    idx_out = idx_parts[0].append(idx_parts[1:]) if len(idx_parts) > 1 else idx_parts[0]
    del X_parts, y_parts, idx_parts
    _free_mem()
    return X_out, y_out, idx_out


def update_experiments_and_build_artifacts() -> dict:
    """Train and evaluate all genuine models, update experiments.json, and generate Figures 1-4."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    api = TaskApi()
    t_start = time.time()

    with open(BENCH_FP, "r") as f:
        persisted = json.load(f)

    # ── 1. Fast Statistical Baselines across all 6 Paranal Tasks (full + multiannual) ──
    print(f"[{time.time()-t_start:6.1f}s] [1/7] Running statistical baselines on all 6 Paranal tasks...", flush=True)
    fast_res = run_benchmarks(
        benchmark_tasks=[
            "regression.paranal_tomography.full.cn2_profile_reconstruction",
            "forecasting.paranal_tomography.full.seeing_nowcast",
            "forecasting.paranal_tomography.full.cn2_profile_forecast",
            "regression.paranal_tomography.multiannual.cn2_profile_reconstruction",
            "forecasting.paranal_tomography.multiannual.seeing_nowcast",
            "forecasting.paranal_tomography.multiannual.cn2_profile_forecast",
        ],
        benchmark_regression_models=[
            "PersistenceRegressionModel",
            "ClimatologyRegressionModel",
            "MinuteClimatologyRegressionModel",
        ],
        benchmark_forecasting_models=[
            "PersistenceForecastingModel",
            "ClimatologyForecastingModel",
            "MinuteClimatologyForecastingModel",
            "MeanWindowForecastingModel",
            "LinearForecastingModel",
        ],
        include_pytorch_models=False,
        write_metrics=False,
        verbose=False,
    )
    for tname, mdict in fast_res.items():
        if tname not in persisted:
            persisted[tname] = {}
        for mname, mvals in mdict.items():
            persisted[tname][mname] = mvals
    _save_bench(persisted)
    _free_mem()

    # ── 2. Genuine Task 1 (full.cn2_profile_reconstruction) GBRT + RNNModel ───────────
    print(f"[{time.time()-t_start:6.1f}s] [2/7] Training genuine GBRT & RNN on Task 1...", flush=True)
    t1_name = "regression.paranal_tomography.full.cn2_profile_reconstruction"
    task_rec = api.get_task(t1_name)
    X_rec_tr, y_rec_tr = task_rec.get_train_data()
    X_rec_va, y_rec_va = task_rec.get_validation_data()
    X_rec_comb = pd.concat([X_rec_tr, X_rec_va])
    y_rec_comb = pd.concat([y_rec_tr, y_rec_va])
    del X_rec_tr, y_rec_tr, X_rec_va, y_rec_va
    valid_rec_tr = y_rec_comb.notna().all(axis=1)
    X_rec_comb = X_rec_comb[valid_rec_tr]
    y_rec_comb = y_rec_comb[valid_rec_tr]
    X_rec_te, y_rec_te = task_rec.get_test_data()
    valid_rec_te = y_rec_te.notna().all(axis=1)
    y_rec_true = y_rec_te[valid_rec_te].to_numpy(dtype=float)
    wind_rec_te = X_rec_te.loc[valid_rec_te, "wind_speed"].to_numpy(dtype=float)

    gbrt_rec = GradientBoostingRegressionModel(
        name="GradientBoostingRegressionModel",
        target_name=LAYER_COLS,
        output_size=7,
        n_estimators=80,
        eta=0.05,
        num_leaves=31,
        colsample_bytree=0.5,
        subsample=0.8,
        subsample_freq=1,
        max_bin=63,
        n_jobs=4,
        force_col_wise=True,
        random_state=2020,
    )
    gbrt_rec.train(X_rec_comb, y_rec_comb)
    eval_gbrt_rec = task_rec.evaluate_model(
        predict_call=gbrt_rec.predict,
        detailed_metrics=True,
        X_test=X_rec_te,
        y_test=y_rec_te,
    )
    persisted[t1_name]["GradientBoostingRegressionModel"] = tasks._compact_metrics_for_benchmark(eval_gbrt_rec)

    rnn_rec = RegressionRNNModel(
        name="RNNModel",
        target_name=LAYER_COLS,
        input_size=len(X_rec_comb.columns),
        output_size=7,
        hidden_size=128,
        num_layers=2,
        batch_size=512,
        n_epochs=15,
        learning_rate=0.001,
        optimizer=torch.optim.Adam,
        weight_decay=1e-2,
        use_scheduler=True,
        val_fraction=0.15,
        patience=2,
        random_state=2020,
        verbose=False,
    )
    rnn_rec.train(X_rec_comb, y_rec_comb)
    eval_rnn_rec = task_rec.evaluate_model(
        predict_call=rnn_rec.predict,
        detailed_metrics=True,
        X_test=X_rec_te,
        y_test=y_rec_te,
    )
    persisted[t1_name]["RNNModel"] = tasks._compact_metrics_for_benchmark(eval_rnn_rec)

    m_mc_rec = MinuteClimatologyRegressionModel(name="MinuteClimatology", target_name=LAYER_COLS)
    m_mc_rec.train(X_rec_comb, y_rec_comb)

    task1_details = {}
    for m_label, pred_fn in [
        ("Persistence", lambda X: np.tile(y_rec_comb.iloc[-1].to_numpy(dtype=float), (len(X), 1))),
        ("Climatology", lambda X: np.tile(np.nanmean(y_rec_comb.to_numpy(dtype=float), axis=0), (len(X), 1))),
        ("MinuteClimatology", m_mc_rec.predict),
        ("GBRT", gbrt_rec.predict),
        ("RNN", rnn_rec.predict),
    ]:
        yp_r = np.asarray(pred_fn(X_rec_te[valid_rec_te]), dtype=float)
        task1_details[m_label] = _compute_full_profile_summary(y_rec_true, yp_r, wind_speed_te=wind_rec_te)

    del X_rec_comb, y_rec_comb, X_rec_te, y_rec_te, gbrt_rec, rnn_rec
    _save_bench(persisted)
    _free_mem()

    # ── 3. Genuine Task 2 (full.seeing_nowcast, W=20, H=5, D_in=44) ───────────────────
    print(f"[{time.time()-t_start:6.1f}s] [3/7] Training genuine ML & PyTorch models on Task 2...", flush=True)
    t2_name = "forecasting.paranal_tomography.full.seeing_nowcast"
    task_sn = api.get_task(t2_name)
    X_sn_tr, y_sn_tr = task_sn.get_train_data()
    X_sn_va, y_sn_va = task_sn.get_validation_data()
    X_sn_comb = pd.concat([X_sn_tr, X_sn_va])
    y_sn_comb = pd.concat([y_sn_tr, y_sn_va])
    del X_sn_tr, y_sn_tr, X_sn_va, y_sn_va
    X_sn_te, y_sn_te = task_sn.get_test_data()
    sn_feat_names = [c for c in X_sn_comb.columns if c != "night_id"]
    X_sn_seq_tr, y_sn_seq_tr, _ = task_sn.prepare_sequence_data(X_sn_comb, y_sn_comb)
    X_sn_seq_tr = X_sn_seq_tr.astype(np.float32, copy=False)
    X_sn_seq_te, y_sn_seq_te, idx_sn_te = task_sn.prepare_sequence_data(X_sn_te, y_sn_te)
    X_sn_seq_te = X_sn_seq_te.astype(np.float32, copy=False)
    del X_sn_comb, y_sn_comb, X_sn_te, y_sn_te
    _free_mem()

    y_sn_te_df = pd.DataFrame(
        y_sn_seq_te[:, np.newaxis] if y_sn_seq_te.ndim == 1 else y_sn_seq_te,
        index=idx_sn_te,
        columns=["seeing"],
    )

    gbrt_sn = GradientBoostingForecastingModel(
        name="GradientBoostingForecastingModel",
        target_name="seeing",
        window_size=20,
        forecast_horizon=5,
        output_size=1,
        use_log10=False,
        predict_residuals=True,
        feature_names=sn_feat_names,
        n_estimators=80,
        eta=0.05,
        num_leaves=31,
        colsample_bytree=0.35,
        subsample=0.8,
        subsample_freq=1,
        max_bin=63,
        n_jobs=4,
        force_col_wise=True,
        random_state=2020,
    )
    gbrt_sn.train(X_sn_seq_tr, y_sn_seq_tr, feature_names=sn_feat_names)
    yp_gbrt_sn = gbrt_sn.predict(X_sn_seq_te, feature_names=sn_feat_names)
    eval_gbrt_sn = task_sn.evaluate_model(
        predict_call=lambda _X, _p=yp_gbrt_sn: _p,
        detailed_metrics=True,
        use_sequence_data=True,
        X_test=X_sn_seq_te,
        y_test=y_sn_te_df,
    )
    persisted[t2_name]["GradientBoostingForecastingModel"] = tasks._compact_metrics_for_benchmark(eval_gbrt_sn)
    del gbrt_sn

    for json_key, Cls, kw in [
        (
            "RNNModel",
            ForecastingRNNModel,
            dict(
                hidden_size=128,
                num_layers=2,
                batch_size=512,
                n_epochs=15,
                learning_rate=0.001,
                optimizer=torch.optim.Adam,
            ),
        ),
        (
            "GRUModel",
            GRUModel,
            dict(hidden_size=128, num_layers=2, batch_size=512, n_epochs=15, learning_rate=0.001),
        ),
        (
            "LSTMModel",
            LSTMModel,
            dict(hidden_size=128, num_layers=2, batch_size=512, n_epochs=15, learning_rate=0.001),
        ),
        (
            "TransformerModel",
            TransformerModel,
            dict(d_model=128, nhead=4, num_layers=2, dropout=0.1, batch_size=512, n_epochs=15, learning_rate=0.001),
        ),
    ]:
        mdl_sn = Cls(
            name=json_key,
            window_size=20,
            forecast_horizon=5,
            target_name="seeing",
            input_size=len(sn_feat_names),
            output_size=1,
            use_log10=False,
            predict_residuals=True,
            use_scheduler=True,
            val_fraction=0.1,
            patience=4,
            random_state=2020,
            feature_names=sn_feat_names,
            verbose=False,
            **kw,
        )
        mdl_sn.train(X_sn_seq_tr, y_sn_seq_tr, feature_names=sn_feat_names)
        yp_sn = mdl_sn.predict(X_sn_seq_te, feature_names=sn_feat_names)
        eval_sn = task_sn.evaluate_model(
            predict_call=lambda _X, _p=yp_sn: _p,
            detailed_metrics=True,
            use_sequence_data=True,
            X_test=X_sn_seq_te,
            y_test=y_sn_te_df,
        )
        persisted[t2_name][json_key] = tasks._compact_metrics_for_benchmark(eval_sn)
        del mdl_sn
        _free_mem()

    del X_sn_seq_tr, y_sn_seq_tr, X_sn_seq_te, y_sn_seq_te, y_sn_te_df
    _save_bench(persisted)
    _free_mem()

    # ── 4. Genuine Task 3 (full.cn2_profile_forecast, W=60, H=10, D_in=50) ────────────
    print(f"[{time.time()-t_start:6.1f}s] [4/7] Training all genuine models on Task 3...", flush=True)
    t3_name = "forecasting.paranal_tomography.full.cn2_profile_forecast"
    task_fc = api.get_task(t3_name)
    X_tr_raw, y_tr_raw = task_fc.get_train_data()
    X_va_raw, y_va_raw = task_fc.get_validation_data()
    X_comb_raw = pd.concat([X_tr_raw, X_va_raw])
    y_comb_raw = pd.concat([y_tr_raw, y_va_raw])
    del X_tr_raw, y_tr_raw, X_va_raw, y_va_raw
    X_te_raw, y_te_raw = task_fc.get_test_data()

    # Generate Figure 1 (Paranal climatology) now while y_comb_raw and y_te_raw are in memory
    df_clim = pd.concat([y_comb_raw, y_te_raw], axis=0)
    fig1 = plot_paranal_climatology(df_clim, heights=HEIGHTS)
    fig1.savefig(OUT_DIR / "paranal_climatology.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)
    del df_clim

    feat_names = [c for c in X_comb_raw.columns if c != "night_id"]
    X_seq_tr, y_seq_tr, idx_tr = task_fc.prepare_sequence_data(X_comb_raw, y_comb_raw)
    X_seq_tr = X_seq_tr.astype(np.float32, copy=False)
    X_seq_te, y_seq_te, idx_te = task_fc.prepare_sequence_data(X_te_raw, y_te_raw)
    X_seq_te = X_seq_te.astype(np.float32, copy=False)
    y_te_df = pd.DataFrame(y_seq_te, index=idx_te, columns=LAYER_COLS)

    wind_te_t3 = X_te_raw.reindex(idx_te)["wind_speed"].ffill().bfill().fillna(10.0).to_numpy(dtype=float)

    preds_fc = {}
    for label, json_key, cls in [
        ("Persistence", "PersistenceForecastingModel", PersistenceForecastingModel),
        ("Climatology", "ClimatologyForecastingModel", ClimatologyForecastingModel),
        ("MeanWindow", "MeanWindowForecastingModel", MeanWindowForecastingModel),
        ("Linear", "LinearForecastingModel", LinearForecastingModel),
    ]:
        m = cls(
            name=json_key,
            target_name=LAYER_COLS,
            window_size=60,
            forecast_horizon=10,
            use_log10=True,
            feature_names=feat_names,
        )
        m.train(X_seq_tr, y_seq_tr, feature_names=feat_names)
        yp = m.predict(X_seq_te, feature_names=feat_names)
        preds_fc[label] = yp
        preds_fc[json_key] = yp

    m_min_clim = MinuteClimatologyForecastingModel(
        name="MinuteClimatologyForecastingModel",
        target_name=LAYER_COLS,
        window_size=60,
        forecast_horizon=10,
        use_log10=True,
    )
    X_min_tr = pd.DataFrame(X_seq_tr[:, -1, :1], index=idx_tr, columns=feat_names[:1])
    X_min_te = pd.DataFrame(X_seq_te[:, -1, :1], index=idx_te, columns=feat_names[:1])
    y_tr_df = pd.DataFrame(y_seq_tr, index=idx_tr, columns=LAYER_COLS)
    m_min_clim.train(X_min_tr, y_tr_df)
    preds_fc["MinuteClimatology"] = m_min_clim.predict(X_min_te)
    preds_fc["MinuteClimatologyForecastingModel"] = preds_fc["MinuteClimatology"]
    del X_min_tr, X_min_te, y_tr_df, m_min_clim

    gbrt_fc = GradientBoostingForecastingModel(
        name="GradientBoostingForecastingModel",
        target_name=LAYER_COLS,
        window_size=60,
        forecast_horizon=10,
        output_size=7,
        use_log10=True,
        predict_residuals=True,
        feature_names=feat_names,
        n_estimators=70,
        eta=0.05,
        num_leaves=31,
        colsample_bytree=0.35,
        subsample=0.75,
        subsample_freq=1,
        max_bin=63,
        n_jobs=4,
        force_col_wise=True,
        random_state=2020,
    )
    gbrt_fc.train(X_seq_tr, y_seq_tr, feature_names=feat_names)
    yp_gbrt_fc = gbrt_fc.predict(X_seq_te, feature_names=feat_names)
    preds_fc["GradientBoostingForecastingModel"] = yp_gbrt_fc
    preds_fc["GBRT (Residual)"] = yp_gbrt_fc
    eval_gbrt_fc = task_fc.evaluate_model(
        predict_call=lambda _X, _p=yp_gbrt_fc: _p,
        detailed_metrics=True,
        use_sequence_data=True,
        X_test=X_seq_te,
        y_test=y_te_df,
    )
    persisted[t3_name]["GradientBoostingForecastingModel"] = tasks._compact_metrics_for_benchmark(eval_gbrt_fc)
    del gbrt_fc

    pt_configs = [
        (
            "RNNModel",
            "RNN (Residual)",
            ForecastingRNNModel,
            dict(
                hidden_size=128,
                num_layers=2,
                batch_size=512,
                n_epochs=15,
                learning_rate=0.001,
                optimizer=torch.optim.Adam,
                predict_residuals=True,
                use_scheduler=True,
                val_fraction=0.1,
                patience=4,
                random_state=2020,
            ),
            True,
        ),
        (
            "GRUModel",
            "GRU (Residual)",
            GRUModel,
            dict(
                hidden_size=128,
                num_layers=2,
                batch_size=512,
                n_epochs=15,
                learning_rate=0.001,
                predict_residuals=True,
                use_scheduler=True,
                val_fraction=0.1,
                patience=4,
                random_state=2020,
            ),
            True,
        ),
        (
            "LSTMModel",
            "LSTM (Residual)",
            LSTMModel,
            dict(
                hidden_size=128,
                num_layers=2,
                batch_size=512,
                n_epochs=15,
                learning_rate=0.001,
                predict_residuals=True,
                use_scheduler=True,
                val_fraction=0.1,
                patience=4,
                random_state=2020,
            ),
            True,
        ),
        (
            "TransformerModel",
            "Transformer (MSE)",
            TransformerModel,
            dict(
                d_model=128,
                nhead=4,
                num_layers=2,
                dropout=0.1,
                batch_size=512,
                n_epochs=15,
                learning_rate=0.001,
                predict_residuals=True,
                use_scheduler=True,
                val_fraction=0.1,
                patience=4,
                use_physics_loss=False,
                random_state=2020,
            ),
            True,
        ),
        (
            "PhysicsTransformerUncorrected",
            "Transformer (AOPhysicsLoss, Uncorrected)",
            PhysicsTransformerModel,
            dict(
                d_model=128,
                nhead=4,
                num_layers=2,
                dropout=0.1,
                batch_size=512,
                n_epochs=15,
                learning_rate=0.001,
                predict_residuals=True,
                use_scheduler=True,
                val_fraction=0.1,
                patience=4,
                use_physics_loss=True,
                lambda_r0=0.15,
                lambda_theta0=0.15,
                jensen_correction=False,
                random_state=2020,
            ),
            False,
        ),
        (
            "PhysicsTransformerModel",
            "Transformer (AOPhysicsLoss)",
            PhysicsTransformerModel,
            dict(
                d_model=128,
                nhead=4,
                num_layers=2,
                dropout=0.1,
                batch_size=512,
                n_epochs=15,
                learning_rate=0.001,
                predict_residuals=True,
                use_scheduler=True,
                val_fraction=0.1,
                patience=4,
                use_physics_loss=True,
                lambda_r0=0.15,
                lambda_theta0=0.15,
                jensen_correction=True,
                random_state=2020,
            ),
            True,
        ),
    ]
    for json_key, display_label, Cls, kw, persist_in_exp in pt_configs:
        mdl = Cls(
            name=json_key,
            window_size=60,
            forecast_horizon=10,
            target_name=LAYER_COLS,
            input_size=len(feat_names),
            output_size=7,
            use_log10=True,
            feature_names=feat_names,
            verbose=False,
            **kw,
        )
        mdl.train(X_seq_tr, y_seq_tr, feature_names=feat_names)
        y_pred_pt = mdl.predict(X_seq_te, feature_names=feat_names)
        preds_fc[display_label] = y_pred_pt
        preds_fc[json_key] = y_pred_pt
        if persist_in_exp:
            eval_res = task_fc.evaluate_model(
                predict_call=lambda _X, _p=y_pred_pt: _p,
                detailed_metrics=True,
                use_sequence_data=True,
                X_test=X_seq_te,
                y_test=y_te_df,
            )
            persisted[t3_name][json_key] = tasks._compact_metrics_for_benchmark(eval_res)
        del mdl
        _free_mem()

    _save_bench(persisted)

    # Generate Figure 2 and Figure 4 + Task 3 summaries/regimes/bootstrap before freeing X_seq_tr
    fig2, (ax_heat, ax_prof) = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1.35, 1.0]})
    pers_layer_rmse = np.asarray(
        persisted[t3_name]["PersistenceForecastingModel"]["root_mean_square_error"]["detailed_score"], dtype=float)
    rmse_dict = {}
    model_display_map = [
        ("Persistence", pers_layer_rmse),
        (
            "Linear",
            np.asarray(persisted[t3_name]["LinearForecastingModel"]["root_mean_square_error"]["detailed_score"],
                       dtype=float),
        ),
        (
            "MeanWindow",
            np.asarray(
                persisted[t3_name]["MeanWindowForecastingModel"]["root_mean_square_error"]["detailed_score"],
                dtype=float,
            ),
        ),
        (
            "GBRT (Residual)",
            np.asarray(
                persisted[t3_name]["GradientBoostingForecastingModel"]["root_mean_square_error"]["detailed_score"],
                dtype=float,
            ),
        ),
        (
            "RNN (Residual)",
            np.asarray(persisted[t3_name]["RNNModel"]["root_mean_square_error"]["detailed_score"], dtype=float),
        ),
        (
            "GRU (Residual)",
            np.asarray(persisted[t3_name]["GRUModel"]["root_mean_square_error"]["detailed_score"], dtype=float),
        ),
        (
            "LSTM (Residual)",
            np.asarray(persisted[t3_name]["LSTMModel"]["root_mean_square_error"]["detailed_score"], dtype=float),
        ),
        (
            "Transformer (MSE)",
            np.asarray(persisted[t3_name]["TransformerModel"]["root_mean_square_error"]["detailed_score"], dtype=float),
        ),
        (
            "Transformer (AOPhysicsLoss)",
            np.asarray(persisted[t3_name]["PhysicsTransformerModel"]["root_mean_square_error"]["detailed_score"],
                       dtype=float),
        ),
    ]
    for label, r_arr in model_display_map:
        rmse_dict[label] = r_arr

    plot_profile_error_heatmap(
        rmse_dict,
        layer_labels=LAYER_LABELS,
        title="(a) Layer-Wise Forecast RMSE by Architecture and Altitude",
        cbar_label=r"Per-Layer RMSE ($\log_{10}\ \mathrm{m}^{1/3}$)",
        cmap="YlOrRd",
        fmt=".3f",
        ax=ax_heat,
    )
    plot_profile_comparison(
        y_seq_te,
        preds_fc["Transformer (AOPhysicsLoss)"],
        heights=HEIGHTS,
        title=r"(b) Vertical $C_n^2(h)$ Profile vs. Hufnagel-Valley 5/7",
        ax=ax_prof,
        add_hv_reference=True,
        convert_to_density=True,
    )
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "profile_error_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    j_tot_te = np.sum(10.0**y_seq_te, axis=1)
    win_len = 160
    best_start = 0
    best_ptp = -1.0
    for start in range(0, len(y_seq_te) - win_len, 25):
        t_seg = pd.to_datetime(idx_te[start:start + win_len])
        if (t_seg[-1] - t_seg[0]) <= pd.Timedelta(minutes=win_len + 30):
            ptp = float(
                np.percentile(j_tot_te[start:start + win_len], 95) - np.percentile(j_tot_te[start:start + win_len], 10))
            if ptp > best_ptp:
                best_ptp = ptp
                best_start = start

    sl = slice(best_start, best_start + win_len)
    fig4 = plot_burst_case_study(
        y_seq_te[sl],
        predictions={
            "Persistence (H=10m)": preds_fc["Persistence"][sl],
            "GBRT (Residual)": preds_fc["GradientBoostingForecastingModel"][sl],
            "Transformer (AOPhysicsLoss)": preds_fc["Transformer (AOPhysicsLoss)"][sl],
        },
        heights=HEIGHTS,
        primary_model="Transformer (AOPhysicsLoss)",
    )
    fig4.savefig(OUT_DIR / "burst_case_study.png", dpi=300, bbox_inches="tight")
    plt.close(fig4)

    regimes_pers = task_fc.evaluate_by_regime(y_seq_te, preds_fc["Persistence"])
    regimes_gbrt = task_fc.evaluate_by_regime(y_seq_te, preds_fc["GradientBoostingForecastingModel"])
    regimes_gru = task_fc.evaluate_by_regime(y_seq_te, preds_fc["GRUModel"])
    regimes_tr = task_fc.evaluate_by_regime(y_seq_te, preds_fc["TransformerModel"])
    regimes_phys = task_fc.evaluate_by_regime(y_seq_te, preds_fc["Transformer (AOPhysicsLoss)"])

    task3_full_summary = {}
    for m_label, yp in [
        ("Persistence", preds_fc["Persistence"]),
        ("Climatology", preds_fc["Climatology"]),
        ("MinuteClimatology", preds_fc["MinuteClimatology"]),
        ("MeanWindow", preds_fc["MeanWindow"]),
        ("Linear", preds_fc["Linear"]),
        ("GBRT", preds_fc["GradientBoostingForecastingModel"]),
        ("RNN", preds_fc["RNNModel"]),
        ("GRU", preds_fc["GRUModel"]),
        ("LSTM", preds_fc["LSTMModel"]),
        ("Transformer", preds_fc["TransformerModel"]),
        ("PhysicsTransformerUncorrected", preds_fc["PhysicsTransformerUncorrected"]),
        ("PhysicsTransformer", preds_fc["PhysicsTransformerModel"]),
    ]:
        task3_full_summary[m_label] = _compute_full_profile_summary(y_seq_te, yp, wind_speed_te=wind_te_t3)

    paired_tests = {
        "PhysicsTransformer_vs_Persistence":
            task_fc.compare_models(
                y_seq_te,
                preds_fc["PhysicsTransformerModel"],
                preds_fc["Persistence"],
                eval_metric_names=["root_mean_square_error", "integrated_seeing", "isoplanatic_angle", "strehl_ratio"],
                n_bootstrap=500,
                block_size=60,
                seed=42,
            ),
        "PhysicsTransformer_vs_TransformerMSE":
            task_fc.compare_models(
                y_seq_te,
                preds_fc["PhysicsTransformerModel"],
                preds_fc["TransformerModel"],
                eval_metric_names=["root_mean_square_error", "integrated_seeing", "isoplanatic_angle", "strehl_ratio"],
                n_bootstrap=500,
                block_size=60,
                seed=42,
            ),
        "PhysicsTransformer_vs_UncorrectedAOPhysicsLoss":
            task_fc.compare_models(
                y_seq_te,
                preds_fc["PhysicsTransformerModel"],
                preds_fc["PhysicsTransformerUncorrected"],
                eval_metric_names=["root_mean_square_error", "integrated_seeing", "isoplanatic_angle", "strehl_ratio"],
                n_bootstrap=500,
                block_size=60,
                seed=42,
            ),
    }

    # ── 5. Genuine Multi-Horizon Sweep (H in {1, 5, 10, 15, 30, 60} min) & Sensor Ablation ──
    print(f"[{time.time()-t_start:6.1f}s] [5/7] Running genuine Multi-Horizon Sweep & Sensor Ablation...", flush=True)
    horizons = [1, 5, 10, 15, 30, 60]
    horizon_rows = []
    modality_ablation_rows = []

    opt_idx = [feat_names.index(c) for c in LAYER_COLS]
    opt_aws_cols = LAYER_COLS + AWS_COLS
    opt_aws_idx = [feat_names.index(c) for c in opt_aws_cols]

    for h_step in horizons:
        print(f"  -> Horizon H={h_step} min...", flush=True)
        if h_step == 10:
            X_h_tr, y_h_tr = X_seq_tr[::2], y_seq_tr[::2]
            X_h_te, y_h_te = X_seq_te, y_seq_te
            y_p_h = preds_fc["Persistence"]
            y_mw_h = preds_fc["MeanWindow"]
            y_gbrt_h = preds_fc["GradientBoostingForecastingModel"]
            y_lstm_h = preds_fc["LSTMModel"]
            y_tr_phys_h = preds_fc["PhysicsTransformerModel"]
        else:
            X_h_tr_full, y_h_tr_full, _ = task_fc.prepare_sequence_data(X_comb_raw,
                                                                        y_comb_raw,
                                                                        window_size=60,
                                                                        forecast_horizon=h_step)
            X_h_tr = X_h_tr_full[::2].astype(np.float32, copy=False)
            y_h_tr = y_h_tr_full[::2].astype(np.float32, copy=False)
            del X_h_tr_full, y_h_tr_full
            X_h_te, y_h_te, _ = task_fc.prepare_sequence_data(X_te_raw,
                                                              y_te_raw,
                                                              window_size=60,
                                                              forecast_horizon=h_step)
            X_h_te = X_h_te.astype(np.float32, copy=False)

            m_p_h = PersistenceForecastingModel(
                name="Persistence",
                target_name=LAYER_COLS,
                window_size=60,
                forecast_horizon=h_step,
                use_log10=True,
                feature_names=feat_names,
            )
            m_p_h.train(X_h_tr, y_h_tr, feature_names=feat_names)
            y_p_h = m_p_h.predict(X_h_te, feature_names=feat_names)

            m_mw_h = MeanWindowForecastingModel(
                name="MeanWindow",
                target_name=LAYER_COLS,
                window_size=60,
                forecast_horizon=h_step,
                use_log10=True,
                feature_names=feat_names,
            )
            m_mw_h.train(X_h_tr, y_h_tr, feature_names=feat_names)
            y_mw_h = m_mw_h.predict(X_h_te, feature_names=feat_names)

            m_gbrt_h = GradientBoostingForecastingModel(
                name="GBRT",
                target_name=LAYER_COLS,
                window_size=60,
                forecast_horizon=h_step,
                output_size=7,
                use_log10=True,
                predict_residuals=True,
                feature_names=feat_names,
                n_estimators=65,
                eta=0.05,
                num_leaves=31,
                colsample_bytree=0.35,
                subsample=0.75,
                subsample_freq=1,
                max_bin=63,
                n_jobs=4,
                force_col_wise=True,
                random_state=2020,
            )
            m_gbrt_h.train(X_h_tr, y_h_tr, feature_names=feat_names)
            y_gbrt_h = m_gbrt_h.predict(X_h_te, feature_names=feat_names)
            del m_gbrt_h

            m_lstm_h = LSTMModel(
                name="LSTM",
                window_size=60,
                forecast_horizon=h_step,
                target_name=LAYER_COLS,
                input_size=len(feat_names),
                output_size=7,
                use_log10=True,
                feature_names=feat_names,
                hidden_size=128,
                num_layers=2,
                batch_size=512,
                n_epochs=15,
                learning_rate=0.001,
                predict_residuals=True,
                use_scheduler=True,
                val_fraction=0.1,
                patience=4,
                random_state=2020,
                verbose=False,
            )
            m_lstm_h.train(X_h_tr, y_h_tr, feature_names=feat_names)
            y_lstm_h = m_lstm_h.predict(X_h_te, feature_names=feat_names)
            del m_lstm_h

            m_phys_h = PhysicsTransformerModel(
                name="PhysicsTransformer",
                window_size=60,
                forecast_horizon=h_step,
                target_name=LAYER_COLS,
                input_size=len(feat_names),
                output_size=7,
                use_log10=True,
                feature_names=feat_names,
                d_model=128,
                nhead=4,
                num_layers=2,
                dropout=0.1,
                batch_size=512,
                n_epochs=15,
                learning_rate=0.001,
                predict_residuals=True,
                use_scheduler=True,
                val_fraction=0.1,
                patience=4,
                use_physics_loss=True,
                lambda_r0=0.15,
                lambda_theta0=0.15,
                jensen_correction=True,
                random_state=2020,
                verbose=False,
            )
            m_phys_h.train(X_h_tr, y_h_tr, feature_names=feat_names)
            y_tr_phys_h = m_phys_h.predict(X_h_te, feature_names=feat_names)
            del m_phys_h
            _free_mem()

        pers_rmse = float(metrics.root_mean_square_error(y_h_te, y_p_h)["metric_value"])
        for m_label, yp_curr in [
            ("Persistence", y_p_h),
            ("MeanWindow", y_mw_h),
            ("GBRT (Residual)", y_gbrt_h),
            ("LSTM (Residual)", y_lstm_h),
            ("Transformer (AOPhysicsLoss)", y_tr_phys_h),
        ]:
            rmse_val = float(metrics.root_mean_square_error(y_h_te, yp_curr)["metric_value"])
            mae_val = float(metrics.mean_absolute_error(y_h_te, yp_curr)["metric_value"])
            ss_val = 1.0 - ((rmse_val**2) / max(pers_rmse**2, 1e-12))
            see_val = float(integrated_metrics.integrated_seeing(y_h_te, yp_curr, heights=HEIGHTS)["metric_value"])
            iso_val = float(integrated_metrics.isoplanatic_angle(y_h_te, yp_curr, heights=HEIGHTS)["metric_value"])
            strehl_val = float(integrated_metrics.strehl_ratio(y_h_te, yp_curr, heights=HEIGHTS)["metric_value"])
            horizon_rows.append({
                "horizon": h_step,
                "model": m_label,
                "root_mean_square_error": rmse_val,
                "mean_absolute_error": mae_val,
                "skill_score": ss_val,
                "integrated_seeing": see_val,
                "isoplanatic_angle": iso_val,
                "strehl_ratio": strehl_val,
            })

        # Multi-Modal Sensor Ablation at horizon h_step: Optical-Only (7) vs Optical+AWS (11) vs Full (50)
        for mod_label, sub_idx, sub_names in [
            ("Optical-Only (7 vars)", opt_idx, LAYER_COLS),
            ("Optical + AWS (11 vars)", opt_aws_idx, opt_aws_cols),
        ]:
            X_sub_tr = X_h_tr[:, :, sub_idx]
            X_sub_te = X_h_te[:, :, sub_idx]
            m_mod = PhysicsTransformerModel(
                name=f"PhysicsTransformer_{mod_label}",
                window_size=60,
                forecast_horizon=h_step,
                target_name=LAYER_COLS,
                input_size=len(sub_names),
                output_size=7,
                use_log10=True,
                feature_names=sub_names,
                d_model=128,
                nhead=4,
                num_layers=2,
                dropout=0.1,
                batch_size=512,
                n_epochs=15,
                learning_rate=0.001,
                predict_residuals=True,
                use_scheduler=True,
                val_fraction=0.1,
                patience=4,
                use_physics_loss=True,
                lambda_r0=0.15,
                lambda_theta0=0.15,
                jensen_correction=True,
                random_state=2020,
                verbose=False,
            )
            m_mod.train(X_sub_tr, y_h_tr, feature_names=sub_names)
            yp_mod = m_mod.predict(X_sub_te, feature_names=sub_names)
            plr_mod = metrics.per_layer_rmse(y_h_te, yp_mod, layer_names=LAYER_LABELS)["per_layer"]
            modality_ablation_rows.append({
                "horizon":
                    h_step,
                "modality":
                    mod_label,
                "n_features":
                    len(sub_names),
                "rmse":
                    float(metrics.root_mean_square_error(y_h_te, yp_mod)["metric_value"]),
                "mae":
                    float(metrics.mean_absolute_error(y_h_te, yp_mod)["metric_value"]),
                "seeing":
                    float(integrated_metrics.integrated_seeing(y_h_te, yp_mod, heights=HEIGHTS)["metric_value"]),
                "isoplanatic":
                    float(integrated_metrics.isoplanatic_angle(y_h_te, yp_mod, heights=HEIGHTS)["metric_value"]),
                "strehl":
                    float(integrated_metrics.strehl_ratio(y_h_te, yp_mod, heights=HEIGHTS)["metric_value"]),
                "gl_rmse":
                    float(plr_mod["0 m (GL)"]),
                "fa_mean_rmse":
                    float(np.mean([plr_mod[k] for k in LAYER_LABELS[1:]])),
            })
            del m_mod, X_sub_tr, X_sub_te
            _free_mem()

        plr_full = metrics.per_layer_rmse(y_h_te, y_tr_phys_h, layer_names=LAYER_LABELS)["per_layer"]
        modality_ablation_rows.append({
            "horizon":
                h_step,
            "modality":
                "Full Multi-Modal (50 vars: Optical + AWS + LHATPRO)",
            "n_features":
                len(feat_names),
            "rmse":
                float(metrics.root_mean_square_error(y_h_te, y_tr_phys_h)["metric_value"]),
            "mae":
                float(metrics.mean_absolute_error(y_h_te, y_tr_phys_h)["metric_value"]),
            "seeing":
                float(integrated_metrics.integrated_seeing(y_h_te, y_tr_phys_h, heights=HEIGHTS)["metric_value"]),
            "isoplanatic":
                float(integrated_metrics.isoplanatic_angle(y_h_te, y_tr_phys_h, heights=HEIGHTS)["metric_value"]),
            "strehl":
                float(integrated_metrics.strehl_ratio(y_h_te, y_tr_phys_h, heights=HEIGHTS)["metric_value"]),
            "gl_rmse":
                float(plr_full["0 m (GL)"]),
            "fa_mean_rmse":
                float(np.mean([plr_full[k] for k in LAYER_LABELS[1:]])),
        })
        if h_step != 10:
            del X_h_tr, y_h_tr, X_h_te, y_h_te
            _free_mem()

    # Generate Figure 3 now and free Task 3 raw/sequence arrays before multiannual stage
    horizon_df = pd.DataFrame(horizon_rows)
    fig3, (ax_h1, ax_h2, ax_h3) = plt.subplots(1, 3, figsize=(15.0, 4.6))
    plot_horizon_decay(
        horizon_df,
        metric="root_mean_square_error",
        ax=ax_h1,
        title=r"(a) Profile RMSE vs. Lead Time $H$",
        ylabel=r"Macro RMSE ($\log_{10}\ \mathrm{m}^{1/3}$)",
        xlabel="Forecast Horizon H (minutes)",
    )
    plot_horizon_decay(
        horizon_df,
        metric="skill_score",
        ax=ax_h2,
        title=r"(b) Persistence Skill Score $\mathrm{SS}(H)$",
        ylabel="Skill Score vs. Persistence",
        xlabel="Forecast Horizon H (minutes)",
    )
    plot_horizon_decay(
        horizon_df,
        metric="integrated_seeing",
        ax=ax_h3,
        title=r"(c) Derived AO Seeing Error $\mathrm{RMSE}_{\varepsilon_0}(H)$",
        ylabel=r"Seeing RMSE $\varepsilon_0$ (arcsec)",
        xlabel="Forecast Horizon H (minutes)",
    )
    fig3.tight_layout()
    fig3.savefig(OUT_DIR / "horizon_skill_decay.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)

    del X_comb_raw, y_comb_raw, X_te_raw, y_te_raw, X_seq_tr, y_seq_tr, X_seq_te, y_seq_te, y_te_df, preds_fc
    _free_mem()

    # ── 6. Complete 5-Year Multiannual Leaderboard (Tasks 1, 2, 3) ───────────────────
    print(f"[{time.time()-t_start:6.1f}s] [6/7] Training ML & PyTorch models on 5-year multiannual tasks...",
          flush=True)
    # 6a. Multiannual Task 1 (cn2_profile_reconstruction)
    ma_t1_name = "regression.paranal_tomography.multiannual.cn2_profile_reconstruction"
    ma_t1 = api.get_task(ma_t1_name)
    X_mt1_tr, y_mt1_tr = ma_t1.get_train_data()
    X_mt1_va, y_mt1_va = ma_t1.get_validation_data()
    X_mt1_comb = pd.concat([X_mt1_tr, X_mt1_va])
    y_mt1_comb = pd.concat([y_mt1_tr, y_mt1_va])
    del X_mt1_tr, y_mt1_tr, X_mt1_va, y_mt1_va
    valid_mt1 = y_mt1_comb.notna().all(axis=1)
    X_mt1_comb = X_mt1_comb[valid_mt1].iloc[::2]
    y_mt1_comb = y_mt1_comb[valid_mt1].iloc[::2]
    X_mt1_te, y_mt1_te = ma_t1.get_test_data()

    gbrt_mt1 = GradientBoostingRegressionModel(
        name="GradientBoostingRegressionModel",
        target_name=LAYER_COLS,
        output_size=7,
        n_estimators=80,
        eta=0.05,
        num_leaves=31,
        colsample_bytree=0.5,
        subsample=0.7,
        subsample_freq=1,
        max_bin=63,
        n_jobs=4,
        force_col_wise=True,
        random_state=2020,
    )
    gbrt_mt1.train(X_mt1_comb, y_mt1_comb)
    persisted[ma_t1_name]["GradientBoostingRegressionModel"] = tasks._compact_metrics_for_benchmark(
        ma_t1.evaluate_model(predict_call=gbrt_mt1.predict, detailed_metrics=True, X_test=X_mt1_te, y_test=y_mt1_te))
    del gbrt_mt1

    rnn_mt1 = RegressionRNNModel(
        name="RNNModel",
        target_name=LAYER_COLS,
        input_size=len(X_mt1_comb.columns),
        output_size=7,
        hidden_size=128,
        num_layers=2,
        batch_size=1024,
        n_epochs=12,
        learning_rate=0.001,
        optimizer=torch.optim.Adam,
        weight_decay=1e-2,
        use_scheduler=True,
        val_fraction=0.15,
        patience=2,
        random_state=2020,
        verbose=False,
    )
    rnn_mt1.train(X_mt1_comb, y_mt1_comb)
    persisted[ma_t1_name]["RNNModel"] = tasks._compact_metrics_for_benchmark(
        ma_t1.evaluate_model(predict_call=rnn_mt1.predict, detailed_metrics=True, X_test=X_mt1_te, y_test=y_mt1_te))
    del X_mt1_comb, y_mt1_comb, X_mt1_te, y_mt1_te, rnn_mt1
    _save_bench(persisted)
    _free_mem()

    # 6b. Multiannual Task 2 (seeing_nowcast) - chunked sequence extraction
    print(f"[{time.time()-t_start:6.1f}s]   -> Multiannual Task 2 (seeing_nowcast)...", flush=True)
    ma_t2_name = "forecasting.paranal_tomography.multiannual.seeing_nowcast"
    ma_t2 = api.get_task(ma_t2_name)
    X_mt2_tr, y_mt2_tr = ma_t2.get_train_data()
    X_mt2_va, y_mt2_va = ma_t2.get_validation_data()
    X_mt2_comb = pd.concat([X_mt2_tr, X_mt2_va])
    y_mt2_comb = pd.concat([y_mt2_tr, y_mt2_va])
    del X_mt2_tr, y_mt2_tr, X_mt2_va, y_mt2_va
    mt2_feats = [c for c in X_mt2_comb.columns if c != "night_id"]
    X_mt2_seq_tr, y_mt2_seq_tr, _ = _prepare_chunked_seq(ma_t2, X_mt2_comb, y_mt2_comb, stride=4, chunk_rows=450000)
    del X_mt2_comb, y_mt2_comb
    _free_mem()

    X_mt2_te, y_mt2_te = ma_t2.get_test_data()
    X_mt2_seq_te, y_mt2_seq_te, idx_mt2_te = _prepare_chunked_seq(ma_t2,
                                                                  X_mt2_te,
                                                                  y_mt2_te,
                                                                  stride=1,
                                                                  chunk_rows=450000)
    del X_mt2_te, y_mt2_te
    _free_mem()

    y_mt2_te_df = pd.DataFrame(
        y_mt2_seq_te[:, np.newaxis] if y_mt2_seq_te.ndim == 1 else y_mt2_seq_te,
        index=idx_mt2_te,
        columns=["seeing"],
    )

    gbrt_mt2 = GradientBoostingForecastingModel(
        name="GradientBoostingForecastingModel",
        target_name="seeing",
        window_size=20,
        forecast_horizon=5,
        output_size=1,
        use_log10=False,
        predict_residuals=True,
        feature_names=mt2_feats,
        n_estimators=70,
        eta=0.05,
        num_leaves=31,
        colsample_bytree=0.35,
        subsample=0.7,
        subsample_freq=1,
        max_bin=63,
        n_jobs=4,
        force_col_wise=True,
        random_state=2020,
    )
    gbrt_mt2.train(X_mt2_seq_tr, y_mt2_seq_tr, feature_names=mt2_feats)
    yp_gbrt_mt2 = gbrt_mt2.predict(X_mt2_seq_te, feature_names=mt2_feats)
    persisted[ma_t2_name]["GradientBoostingForecastingModel"] = tasks._compact_metrics_for_benchmark(
        ma_t2.evaluate_model(
            predict_call=lambda _X, _p=yp_gbrt_mt2: _p,
            detailed_metrics=True,
            use_sequence_data=True,
            X_test=X_mt2_seq_te,
            y_test=y_mt2_te_df,
        ))
    del gbrt_mt2

    for json_key, Cls, kw in [
        (
            "RNNModel",
            ForecastingRNNModel,
            dict(
                hidden_size=128,
                num_layers=2,
                batch_size=1024,
                n_epochs=12,
                learning_rate=0.001,
                optimizer=torch.optim.Adam,
            ),
        ),
        (
            "GRUModel",
            GRUModel,
            dict(hidden_size=128, num_layers=2, batch_size=1024, n_epochs=12, learning_rate=0.001),
        ),
        (
            "LSTMModel",
            LSTMModel,
            dict(hidden_size=128, num_layers=2, batch_size=1024, n_epochs=12, learning_rate=0.001),
        ),
        (
            "TransformerModel",
            TransformerModel,
            dict(d_model=128, nhead=4, num_layers=2, dropout=0.1, batch_size=1024, n_epochs=12, learning_rate=0.001),
        ),
    ]:
        mdl_mt2 = Cls(
            name=json_key,
            window_size=20,
            forecast_horizon=5,
            target_name="seeing",
            input_size=len(mt2_feats),
            output_size=1,
            use_log10=False,
            predict_residuals=True,
            use_scheduler=True,
            val_fraction=0.1,
            patience=4,
            random_state=2020,
            feature_names=mt2_feats,
            verbose=False,
            **kw,
        )
        mdl_mt2.train(X_mt2_seq_tr, y_mt2_seq_tr, feature_names=mt2_feats)
        yp_mt2 = mdl_mt2.predict(X_mt2_seq_te, feature_names=mt2_feats)
        persisted[ma_t2_name][json_key] = tasks._compact_metrics_for_benchmark(
            ma_t2.evaluate_model(
                predict_call=lambda _X, _p=yp_mt2: _p,
                detailed_metrics=True,
                use_sequence_data=True,
                X_test=X_mt2_seq_te,
                y_test=y_mt2_te_df,
            ))
        del mdl_mt2
        _free_mem()

    del X_mt2_seq_tr, y_mt2_seq_tr, X_mt2_seq_te, y_mt2_seq_te, y_mt2_te_df
    _save_bench(persisted)
    _free_mem()

    # 6c. Multiannual Task 3 (cn2_profile_forecast) - chunked sequence extraction
    print(f"[{time.time()-t_start:6.1f}s]   -> Multiannual Task 3 (cn2_profile_forecast)...", flush=True)
    ma_t3_name = "forecasting.paranal_tomography.multiannual.cn2_profile_forecast"
    ma_t3 = api.get_task(ma_t3_name)
    X_mt3_tr, y_mt3_tr = ma_t3.get_train_data()
    X_mt3_va, y_mt3_va = ma_t3.get_validation_data()
    X_mt3_comb = pd.concat([X_mt3_tr, X_mt3_va])
    y_mt3_comb = pd.concat([y_mt3_tr, y_mt3_va])
    del X_mt3_tr, y_mt3_tr, X_mt3_va, y_mt3_va
    mt3_feats = [c for c in X_mt3_comb.columns if c != "night_id"]
    X_mt3_seq_tr, y_mt3_seq_tr, _ = _prepare_chunked_seq(ma_t3, X_mt3_comb, y_mt3_comb, stride=4, chunk_rows=450000)
    del X_mt3_comb, y_mt3_comb
    _free_mem()

    X_mt3_te, y_mt3_te = ma_t3.get_test_data()
    X_mt3_seq_te, y_mt3_seq_te, idx_mt3_te = _prepare_chunked_seq(ma_t3,
                                                                  X_mt3_te,
                                                                  y_mt3_te,
                                                                  stride=1,
                                                                  chunk_rows=450000)
    del X_mt3_te, y_mt3_te
    _free_mem()

    y_mt3_te_df = pd.DataFrame(y_mt3_seq_te, index=idx_mt3_te, columns=LAYER_COLS)

    gbrt_mt3 = GradientBoostingForecastingModel(
        name="GradientBoostingForecastingModel",
        target_name=LAYER_COLS,
        window_size=60,
        forecast_horizon=10,
        output_size=7,
        use_log10=True,
        predict_residuals=True,
        feature_names=mt3_feats,
        n_estimators=65,
        eta=0.05,
        num_leaves=31,
        colsample_bytree=0.35,
        subsample=0.7,
        subsample_freq=1,
        max_bin=63,
        n_jobs=4,
        force_col_wise=True,
        random_state=2020,
    )
    gbrt_mt3.train(X_mt3_seq_tr, y_mt3_seq_tr, feature_names=mt3_feats)
    yp_gbrt_mt3 = gbrt_mt3.predict(X_mt3_seq_te, feature_names=mt3_feats)
    persisted[ma_t3_name]["GradientBoostingForecastingModel"] = tasks._compact_metrics_for_benchmark(
        ma_t3.evaluate_model(
            predict_call=lambda _X, _p=yp_gbrt_mt3: _p,
            detailed_metrics=True,
            use_sequence_data=True,
            X_test=X_mt3_seq_te,
            y_test=y_mt3_te_df,
        ))
    del gbrt_mt3

    for json_key, Cls, kw in [
        (
            "RNNModel",
            ForecastingRNNModel,
            dict(
                hidden_size=128,
                num_layers=2,
                batch_size=1024,
                n_epochs=12,
                learning_rate=0.001,
                optimizer=torch.optim.Adam,
            ),
        ),
        (
            "GRUModel",
            GRUModel,
            dict(hidden_size=128, num_layers=2, batch_size=1024, n_epochs=12, learning_rate=0.001),
        ),
        (
            "LSTMModel",
            LSTMModel,
            dict(hidden_size=128, num_layers=2, batch_size=1024, n_epochs=12, learning_rate=0.001),
        ),
        (
            "TransformerModel",
            TransformerModel,
            dict(
                d_model=128,
                nhead=4,
                num_layers=2,
                dropout=0.1,
                batch_size=1024,
                n_epochs=12,
                learning_rate=0.001,
                use_physics_loss=False,
            ),
        ),
        (
            "PhysicsTransformerModel",
            PhysicsTransformerModel,
            dict(
                d_model=128,
                nhead=4,
                num_layers=2,
                dropout=0.1,
                batch_size=1024,
                n_epochs=12,
                learning_rate=0.001,
                use_physics_loss=True,
                lambda_r0=0.15,
                lambda_theta0=0.15,
                jensen_correction=True,
            ),
        ),
    ]:
        mdl_mt3 = Cls(
            name=json_key,
            window_size=60,
            forecast_horizon=10,
            target_name=LAYER_COLS,
            input_size=len(mt3_feats),
            output_size=7,
            use_log10=True,
            predict_residuals=True,
            use_scheduler=True,
            val_fraction=0.1,
            patience=4,
            random_state=2020,
            feature_names=mt3_feats,
            verbose=False,
            **kw,
        )
        mdl_mt3.train(X_mt3_seq_tr, y_mt3_seq_tr, feature_names=mt3_feats)
        yp_mt3 = mdl_mt3.predict(X_mt3_seq_te, feature_names=mt3_feats)
        persisted[ma_t3_name][json_key] = tasks._compact_metrics_for_benchmark(
            ma_t3.evaluate_model(
                predict_call=lambda _X, _p=yp_mt3: _p,
                detailed_metrics=True,
                use_sequence_data=True,
                X_test=X_mt3_seq_te,
                y_test=y_mt3_te_df,
            ))
        del mdl_mt3
        _free_mem()

    del X_mt3_seq_tr, y_mt3_seq_tr, X_mt3_seq_te, y_mt3_seq_te, y_mt3_te_df
    _save_bench(persisted)
    _free_mem()

    print(f"[{time.time()-t_start:6.1f}s] [7/7] Finalizing empirical_summary.json...", flush=True)
    return {
        "climatology": compute_dataset_climatology(),
        "task1_details": task1_details,
        "task2_summary": persisted[t2_name],
        "task3_full_summary": task3_full_summary,
        "paired_tests": paired_tests,
        "horizon_rows": horizon_rows,
        "modality_ablation_rows": modality_ablation_rows,
        "regimes": {
            "Persistence": regimes_pers,
            "GBRT": regimes_gbrt,
            "GRU": regimes_gru,
            "Transformer": regimes_tr,
            "Transformer_AOPhysicsLoss": regimes_phys,
        },
        "multiannual": {
            ma_t1_name: persisted[ma_t1_name],
            ma_t2_name: persisted[ma_t2_name],
            ma_t3_name: persisted[ma_t3_name],
        },
    }


if __name__ == "__main__":
    summary = update_experiments_and_build_artifacts()
    out_summary_fp = OUT_DIR / "empirical_summary.json"
    with open(out_summary_fp, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    print(f"Successfully generated all otb2 figures and {out_summary_fp}", flush=True)

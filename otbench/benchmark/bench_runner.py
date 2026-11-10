import os
import json
import pprint
from typing import List, Union

import pandas as pd
import numpy as np

from otbench.tasks import TaskApi, tasks
from otbench.config import BENCHMARK_FP
import otbench.benchmark.models.regression as regression_models
import otbench.benchmark.models.forecasting as forecasting_models

PPRINTER = pprint.PrettyPrinter(indent=4, width=120, compact=True)


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.complexfloating):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.void):
            return None
        return json.JSONEncoder.default(self, obj)


def _resolve_model_map(
    requested: Union[List, str, dict, None],
    available_registry: dict,
    kind: str,
    extended_registry: Union[dict, None] = None,
) -> dict:
    """Resolve user-supplied model names, classes, or mapping against available_registry."""
    if requested is None:
        return dict(available_registry)
    if isinstance(requested, dict):
        return dict(requested)
    if isinstance(requested, str):
        requested = [requested]

    lookup = dict(available_registry)
    if extended_registry:
        lookup.update(extended_registry)

    resolved = {}
    for item in requested:
        if isinstance(item, str):
            if item not in lookup:
                raise ValueError(f"Unknown {kind} model '{item}'. Available {kind} models: {sorted(lookup.keys())}")
            resolved[item] = lookup[item]
        elif isinstance(item, tuple) and len(item) >= 2 and callable(item[1]):
            resolved[str(item[0])] = item[1]
        elif callable(item):
            name = getattr(item, "__name__", str(item))
            resolved[name] = item
        else:
            raise ValueError(f"Unsupported model specification {item!r} for {kind} benchmark.")
    return resolved


def run_benchmarks(benchmark_tasks: Union[List[str], str, None] = None,
                   benchmark_regression_models: Union[List[str], str, dict, None] = None,
                   benchmark_forecasting_models: Union[List[str], str, dict, None] = None,
                   verbose: bool = True,
                   include_pytorch_models: bool = True,
                   write_metrics: bool = True,
                   metrics_fp: Union[os.PathLike, str, None] = None,
                   n_epochs_override: Union[int, None] = None,
                   seed: Union[int, None] = None,
                   include_multiannual: bool = False) -> dict:
    """Run benchmarks for all tasks and models."""
    available_reg = {n: getattr(regression_models, n) for n in regression_models.__all__}
    available_fcn = {n: getattr(forecasting_models, n) for n in forecasting_models.__all__}
    extended_fcn = dict(available_fcn)

    if include_pytorch_models:
        try:
            import otbench.benchmark.models.regression.pytorch as pt_regression_models
            import otbench.benchmark.models.forecasting.pytorch as pt_forecasting_models

            available_reg.update({n: getattr(pt_regression_models, n) for n in pt_regression_models.__all__})
            available_fcn.update({n: getattr(pt_forecasting_models, n) for n in pt_forecasting_models.__all__})
            ext_names = getattr(pt_forecasting_models, "__all_extended__", pt_forecasting_models.__all__)
            extended_fcn.update({n: getattr(pt_forecasting_models, n) for n in ext_names})
        except ImportError as e:
            print(f"failed to import dependency with error {e}.\n skipping PyTorch models.")

    reg_models = _resolve_model_map(benchmark_regression_models, available_reg, kind="regression")
    fcn_models = _resolve_model_map(benchmark_forecasting_models,
                                    available_fcn,
                                    kind="forecasting",
                                    extended_registry=extended_fcn)

    if metrics_fp is None:
        metrics_fp = BENCHMARK_FP

    task_api = TaskApi()
    if benchmark_tasks is None:
        benchmark_tasks = sorted(t for t in task_api.list_tasks() if include_multiannual or ".multiannual." not in t)
    elif isinstance(benchmark_tasks, str):
        benchmark_tasks = [benchmark_tasks]

    benchmark_results = {}

    # Task Execution Loop
    for task_name in benchmark_tasks:
        if verbose:
            print(f"Running benchmark for {task_name}...")

        task = task_api.get_task(task_name, benchmark_fp=metrics_fp)
        target_name = task.get_target_name()
        task_info = task.get_info()

        if verbose:
            PPRINTER.pprint(task_info)

        # Task Metadata Extraction
        obs_timezone = task_info["obs_tz"]
        obs_lat = task_info["obs_lat"]
        obs_lon = task_info["obs_lon"]
        use_log10 = task_info["log_transform"]

        # Data Loading
        # Train + Val combined gives models the most pre-test data for benchmarking.
        X_train, y_train = task.get_train_data(data_type="pd")
        X_val, y_val = task.get_validation_data(data_type="pd")
        X_combined = pd.concat([X_train, X_val])
        y_combined = pd.concat([y_train, y_val])

        X_test, y_test = task.get_test_data(data_type="pd")

        # Select model class and prepare training data
        if isinstance(task, tasks.RegressionTask):
            models = reg_models
            # Exclude rows where any target is NaN while preserving feature-NaN rows for tree models
            valid_rows = y_combined.notna().all(axis=1)
            X_bench = X_combined[valid_rows]
            y_bench = y_combined[valid_rows]

        elif isinstance(task, tasks.ForecastingTask):
            models = fcn_models
            session_col = task_info.get("session_col")
            seq_feature_names = [c for c in X_combined.columns if c != session_col]
            X_seq_bench, y_seq_bench, idx_seq_bench = task.prepare_sequence_data(X_combined, y_combined)
            y_cols = (list(y_combined.columns) if isinstance(y_combined, pd.DataFrame) else [y_combined.name])
            y_2d = y_seq_bench[:, np.newaxis] if y_seq_bench.ndim == 1 else y_seq_bench
            y_bench_df = pd.DataFrame(y_2d, index=idx_seq_bench, columns=y_cols)

            seq_native_models = {
                "RNNModel",
                "GRUModel",
                "LSTMModel",
                "TransformerModel",
                "PhysicsTransformerModel",
                "PersistenceForecastingModel",
                "ClimatologyForecastingModel",
                "MeanWindowForecastingModel",
                "LinearForecastingModel",
                "GradientBoostingForecastingModel",
            }
            # Lazily materialize full 2D lagged DataFrame only when a model requires all lagged columns as DataFrame
            needs_2d_bench = any(
                m not in seq_native_models and m != "MinuteClimatologyForecastingModel" for m in models.keys())
            lag_col_names = list(seq_feature_names) + [
                f"{col} (t-{lag})" for lag in range(1, task.window_size) for col in seq_feature_names
            ]
            if needs_2d_bench and len(X_seq_bench) > 0:
                X_flat = X_seq_bench[:, ::-1, :].reshape(len(X_seq_bench), task.window_size * len(seq_feature_names))
                X_bench = pd.DataFrame(X_flat, index=idx_seq_bench, columns=lag_col_names)
                y_bench = y_bench_df
            else:
                X_bench = None
                y_bench = y_seq_bench

        else:
            raise ValueError(f"unknown task type {type(task)}.")

        if isinstance(task, tasks.ForecastingTask):
            X_seq_eval, y_eval, idx_eval = task.prepare_sequence_data(X_test, y_test)
            possible_preds = int(len(y_eval))
            y_eval_2d = y_eval[:, np.newaxis] if y_eval.ndim == 1 else y_eval
            y_eval_df = pd.DataFrame(y_eval_2d, index=idx_eval, columns=y_cols)
            X_eval_2d = None
            if needs_2d_bench and len(X_seq_eval) > 0:
                X_eval_flat = X_seq_eval[:, ::-1, :].reshape(len(X_seq_eval), task.window_size * len(seq_feature_names))
                X_eval_2d = pd.DataFrame(X_eval_flat, index=idx_eval, columns=lag_col_names)
                X_eval_2d.attrs["_already_prepared"] = True
        else:
            possible_preds = int(y_test.notna().all(axis=1).sum())

        benchmark_results[task_name] = {}
        benchmark_results[task_name]["possible_predictions"] = possible_preds

        # Resolve dataset feature mapping
        ds_name = task_info["ds_name"]
        datasets_config = task.get_dataset()._supported_datasets()
        feature_map = datasets_config.get(ds_name, {}).get("feature_map")
        if not feature_map:
            raise ValueError(f"No feature_map configured for dataset '{ds_name}' in datasets.json. "
                             f"Cannot run benchmarks for task '{task_name}'.")

        height_of_observation = feature_map["height_of_observation"]
        air_temperature_col_name = feature_map["air_temperature_col_name"]
        water_temperature_col_name = feature_map.get("water_temperature_col_name")
        humidity_col_name = feature_map["humidity_col_name"]
        wind_speed_col_name = feature_map["wind_speed_col_name"]
        time_col_name = feature_map["time_col_name"]

        output_dim = 1
        if hasattr(y_bench, "shape") and len(y_bench.shape) > 1:
            output_dim = y_bench.shape[1]

        is_vector_task = output_dim > 1

        scalar_only_models = [
            "MacroMeteorologicalModel",
            "OffshoreMacroMeteorologicalModel",
            "AWTModel",
            "HybridAWTRegressionModel",
        ]

        # Model Training Loop
        for model_name, model in models.items():
            # Skip models requiring water temp if missing
            if water_temperature_col_name is None and ("AirWaterTemperature" in model_name or "AWT" in model_name):
                if verbose:
                    print(f"Skipping {model_name} (needs Water Temp).")
                continue

            # Skip scalar models for vector tasks
            if is_vector_task and model_name in scalar_only_models:
                if verbose:
                    print(f"Skipping {model_name} for vector task '{task_name}' (incompatible).")
                continue

            if verbose:
                print(f"Running benchmark for {model_name}...")

            base_input_size = (len(seq_feature_names)
                               if isinstance(task, tasks.ForecastingTask) else len(X_bench.columns))
            model_kwargs = dict(
                name=model_name,
                target_name=target_name,
                timezone=obs_timezone,
                obs_lat=obs_lat,
                obs_lon=obs_lon,
                air_temperature_col_name=air_temperature_col_name,
                water_temperature_col_name=water_temperature_col_name,
                humidity_col_name=humidity_col_name,
                wind_speed_col_name=wind_speed_col_name,
                time_col_name=time_col_name,
                height_of_observation=height_of_observation,
                enforce_dynamic_range=True,
                constant_adjustment=True,
                use_log10=use_log10,
                verbose=verbose,
                input_size=base_input_size,
                output_size=output_dim,
            )

            # Forecasting Specific Configs
            if "forecasting" in task_name:
                model_kwargs["forecast_horizon"] = task.forecast_horizon
                model_kwargs["window_size"] = task.window_size
                model_kwargs["feature_names"] = seq_feature_names
                if X_bench is not None:
                    n_cols = len(X_bench.columns)
                    assert n_cols % task.window_size == 0, (f"X_bench has {n_cols} columns which is not divisible by "
                                                            f"window_size={task.window_size}. Check that _add_lags was "
                                                            f"called with uniform lag depth across all features.")
                model_kwargs["input_size"] = len(seq_feature_names)

            # Model Specific Configs
            if any(k in model_name for k in ("RNNModel", "GRUModel", "LSTMModel")) and "forecasting" in task_name:
                model_kwargs.update({
                    "batch_size": 256,
                    "n_epochs": 20,
                    "predict_residuals": True,
                    "use_scheduler": True,
                    "val_fraction": 0.1,
                    "patience": 5,
                })
                if "paranal_tomography" in task_name:
                    model_kwargs["hidden_size"] = 128
                    if "RNNModel" in model_name:
                        import torch
                        model_kwargs["optimizer"] = torch.optim.Adam

            if "TransformerModel" in model_name:
                model_kwargs.update({
                    "d_model": 128,
                    "nhead": 4,
                    "num_layers": 2,
                    "dropout": 0.1,
                    # Transformer needs larger batch, residual target framing, and scheduled epochs
                    "batch_size": 256,
                    "n_epochs": 20,
                    "predict_residuals": True,
                    "use_scheduler": True,
                    "val_fraction": 0.1,
                    "patience": 5,
                })

            # Override epochs
            if n_epochs_override is not None:
                model_kwargs["n_epochs"] = n_epochs_override

            if seed is not None:
                model_kwargs["random_state"] = int(seed)
                np.random.seed(int(seed))
                try:
                    import torch
                    torch.manual_seed(int(seed))
                except ImportError:
                    pass

            # Instantiate & Train
            try:
                mdl = model(**model_kwargs)
                if isinstance(task, tasks.ForecastingTask):
                    if model_name in seq_native_models:
                        mdl.train(
                            X_seq_bench,
                            y_seq_bench,
                            feature_names=seq_feature_names,
                        )
                        results = task.evaluate_model(
                            predict_call=mdl.predict,
                            x_transforms=None,
                            x_transform_kwargs=None,
                            detailed_metrics=True,
                            use_sequence_data=True,
                            X_test=X_seq_eval,
                            y_test=y_eval_df,
                        )
                    elif model_name == "MinuteClimatologyForecastingModel":
                        X_min_tr = pd.DataFrame(
                            X_seq_bench[:, -1, :1] if len(X_seq_bench) > 0 else np.empty((0, 1)),
                            index=idx_seq_bench,
                            columns=seq_feature_names[:1],
                        )
                        X_min_te = pd.DataFrame(
                            X_seq_eval[:, -1, :1] if len(X_seq_eval) > 0 else np.empty((0, 1)),
                            index=idx_eval,
                            columns=seq_feature_names[:1],
                        )
                        X_min_te.attrs["_already_prepared"] = True
                        mdl.train(X_min_tr, y_bench_df)
                        results = task.evaluate_model(
                            predict_call=mdl.predict,
                            x_transforms=None,
                            x_transform_kwargs=None,
                            detailed_metrics=True,
                            X_test=X_min_te,
                            y_test=y_eval_df,
                        )
                    else:
                        mdl.train(X_bench, y_bench)
                        results = task.evaluate_model(
                            predict_call=mdl.predict,
                            x_transforms=None,
                            x_transform_kwargs=None,
                            detailed_metrics=True,
                            X_test=X_eval_2d,
                            y_test=y_eval_df,
                        )
                else:
                    mdl.train(X_bench, y_bench)
                    results = task.evaluate_model(
                        predict_call=mdl.predict,
                        x_transforms=None,
                        x_transform_kwargs=None,
                        detailed_metrics=True,
                        X_test=X_test,
                        y_test=y_test,
                    )
                benchmark_results[task_name][model_name] = tasks._compact_metrics_for_benchmark(results)

                if verbose:
                    print(f"Done running benchmark for {model_name}.")
                    # Only print scalar summary for brevity
                    summary = {k: v['metric_value'] for k, v in results.items() if isinstance(v, dict)}
                    PPRINTER.pprint(summary)
            except Exception as e:
                print(f"Failed to run {model_name} on {task_name}: {e}")
                import traceback
                traceback.print_exc()

    if write_metrics:
        persisted_results = {}
        if os.path.exists(metrics_fp):
            try:
                with open(metrics_fp, "r") as f:
                    persisted_results = json.load(f)
            except (json.JSONDecodeError, OSError):
                persisted_results = {}
        for task_name, task_res in benchmark_results.items():
            if task_name in persisted_results and isinstance(persisted_results[task_name], dict):
                persisted_results[task_name].update(task_res)
            else:
                persisted_results[task_name] = task_res
        with open(metrics_fp, "w") as f:
            f.write(json.dumps(persisted_results, indent=4, cls=NumpyEncoder))
        if verbose:
            print(f"Wrote benchmark metrics to {metrics_fp}.")

    if verbose:
        print("Done running benchmarks.")

    return benchmark_results

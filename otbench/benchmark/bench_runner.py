import os
import json
import pprint
from typing import List, Union

import pandas as pd
import numpy as np

from otbench.tasks import TaskApi, tasks
from otbench.config import BENCHMARK_FP, settings
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


def run_benchmarks(benchmark_tasks: Union[List[str], str, None] = None,
                   benchmark_regression_models: Union[List[str], str, None] = None,
                   benchmark_forecasting_models: Union[List[str], str, None] = None,
                   verbose: bool = True,
                   include_pytorch_models: bool = True,
                   write_metrics: bool = True,
                   metrics_fp: Union[os.PathLike, str, None] = None,
                   n_epochs_override: Union[int, None] = None) -> dict:
    """Run benchmarks for all tasks and models."""
    if benchmark_regression_models is None:
        reg_models = {n: getattr(regression_models, n) for n in regression_models.__all__}
    else:
        if isinstance(benchmark_regression_models, str):
            benchmark_regression_models = [benchmark_regression_models]
        reg_models = {
            n: getattr(regression_models, n) for n in benchmark_regression_models if n in regression_models.__all__
        }
    if benchmark_forecasting_models is None:
        fcn_models = {n: getattr(forecasting_models, n) for n in forecasting_models.__all__}
    else:
        if isinstance(benchmark_forecasting_models, str):
            benchmark_forecasting_models = [benchmark_forecasting_models]
        fcn_models = {
            n: getattr(forecasting_models, n) for n in benchmark_forecasting_models if n in forecasting_models.__all__
        }

    if include_pytorch_models:
        try:
            import otbench.benchmark.models.regression.pytorch as pt_regression_models
            import otbench.benchmark.models.forecasting.pytorch as pt_forecasting_models

            if benchmark_regression_models is None:
                pytorch_regression_models = {n: getattr(pt_regression_models, n) for n in pt_regression_models.__all__}
            else:
                pytorch_regression_models = {
                    n: getattr(pt_regression_models, n)
                    for n in benchmark_regression_models
                    if n in pt_regression_models.__all__
                }
            if benchmark_forecasting_models is None:
                pytorch_forecasting_models = {
                    n: getattr(pt_forecasting_models, n) for n in pt_forecasting_models.__all__
                }
            else:
                pytorch_forecasting_models = {
                    n: getattr(pt_forecasting_models, n)
                    for n in benchmark_forecasting_models
                    if n in pt_forecasting_models.__all__
                }

            reg_models = {**reg_models, **pytorch_regression_models}
            fcn_models = {**fcn_models, **pytorch_forecasting_models}

        except ImportError as e:
            print(f"failed to import dependency with error {e}.\n skipping PyTorch models.")

    if metrics_fp is None:
        metrics_fp = BENCHMARK_FP

    task_api = TaskApi()
    if benchmark_tasks is None:
        benchmark_tasks = sorted(task_api.list_tasks())
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

        # Select model class and prepare training data.
        # session_col (e.g. night_id) is already present in X because it is
        # not in the task's remove list; prepare_forecasting_data handles it
        # internally — no explicit context recovery is needed here.
        if isinstance(task, tasks.RegressionTask):
            models = reg_models
            # Exclude rows where any target is NaN.  These carry no training
            # signal and are rejected by sklearn MultiOutputRegressor; they
            # also cause NaN-gradient poisoning in PyTorch when normalize_data
            # is False.  LightGBM handles NaN *features* natively so we only
            # filter on the target mask, preserving all X-NaN rows that tree
            # models can use.
            valid_rows = y_combined.notna().all(axis=1)
            X_bench = X_combined[valid_rows]
            y_bench = y_combined[valid_rows]

        elif isinstance(task, tasks.ForecastingTask):
            models = fcn_models
            X_bench, y_bench = task.prepare_forecasting_data(X_combined, y_combined)

        else:
            raise ValueError(f"unknown task type {type(task)}.")

        # possible_predictions: ground-truth rows available in the evaluation
        # split, after applying the same windowing that evaluate_model uses.
        # .all(axis=1) correctly handles both scalar (1-col) and vector targets:
        # a row is "possible" only when every target value is non-NaN.
        if isinstance(task, tasks.ForecastingTask):
            _, y_eval = task.prepare_forecasting_data(X_test, y_test)
        else:
            y_eval = y_test

        benchmark_results[task_name] = {}
        benchmark_results[task_name]["possible_predictions"] = int(
            y_eval.notna().all(axis=1).sum()
        )

        # Feature Mapping (declarative, from datasets.json)
        ds_name = task_info["ds_name"]
        with open(settings.DATASETS_FP, 'r') as f:
            datasets_config = json.load(f)
        feature_map = datasets_config.get(ds_name, {}).get("feature_map")
        if not feature_map:
            raise ValueError(
                f"No feature_map configured for dataset '{ds_name}' in datasets.json. "
                f"Cannot run benchmarks for task '{task_name}'."
            )

        height_of_observation = feature_map["height_of_observation"]
        air_temperature_col_name = feature_map["air_temperature_col_name"]
        water_temperature_col_name = feature_map.get("water_temperature_col_name")
        humidity_col_name = feature_map["humidity_col_name"]
        wind_speed_col_name = feature_map["wind_speed_col_name"]
        time_col_name = feature_map["time_col_name"]

        # Determine Vector Status
        # Check output dimensionality to filter incompatible models
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
                input_size=len(X_bench.columns),
                output_size=output_dim,  # Critical for Vector Tasks
            )

            # Forecasting Specific Configs
            if "forecasting" in task_name:
                model_kwargs["forecast_horizon"] = task.forecast_horizon
                model_kwargs["window_size"] = task.window_size
                # Input size for RNN/Transformer is Features per Step.
                # _add_lags uniformly lags every feature, so total columns must be
                # an exact multiple of window_size.
                n_cols = len(X_bench.columns)
                assert n_cols % task.window_size == 0, (
                    f"X_bench has {n_cols} columns which is not divisible by "
                    f"window_size={task.window_size}. Check that _add_lags was "
                    f"called with uniform lag depth across all features."
                )
                model_kwargs["input_size"] = n_cols // task.window_size
                model_kwargs["in_channels"] = task.window_size

            # Override epochs
            if n_epochs_override is not None:
                model_kwargs["n_epochs"] = n_epochs_override

            # Model Specific Configs
            if "TransformerModel" in model_name:
                model_kwargs.update({
                    "d_model": 128,
                    "nhead": 4,
                    "num_layers": 2,
                    "dropout": 0.1,
                    # Transformer needs larger batch and careful epochs
                    "batch_size": 256,
                    "n_epochs": 20
                })

            # Instantiate & Train
            try:
                mdl = model(**model_kwargs)
                mdl.train(X_bench.copy(deep=True), y_bench.copy(deep=True))

                # Evaluate
                results = task.evaluate_model(predict_call=mdl.predict, x_transforms=None, x_transform_kwargs=None, detailed_metrics=True,)
                benchmark_results[task_name][model_name] = results

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
        with open(metrics_fp, "w") as f:
            f.write(json.dumps(benchmark_results, indent=4, cls=NumpyEncoder))
        if verbose:
            print(f"Wrote benchmark metrics to {metrics_fp}.")

    if verbose:
        print("Done running benchmarks.")

    return benchmark_results

import os
import json
import pprint
from typing import List, Union

import pandas as pd

from otbench.tasks import TaskApi, tasks
from otbench.config import BENCHMARK_FP
import otbench.benchmark.models.regression as regression_models
import otbench.benchmark.models.forecasting as forecasting_models

PPRINTER = pprint.PrettyPrinter(indent=4, width=120, compact=True)


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
        if type(benchmark_regression_models) == str:
            benchmark_regression_models = [benchmark_regression_models]
        reg_models = {
            n: getattr(regression_models, n) for n in benchmark_regression_models if n in regression_models.__all__
        }
    if benchmark_forecasting_models is None:
        fcn_models = {n: getattr(forecasting_models, n) for n in forecasting_models.__all__}
    else:
        if type(benchmark_forecasting_models) == str:
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
    elif type(benchmark_tasks) == str:
        benchmark_tasks = [benchmark_tasks]
    benchmark_results = {}

    for task_name in benchmark_tasks:
        if verbose:
            print(f"Running benchmark for {task_name}...")

        task = task_api.get_task(task_name, benchmark_fp=metrics_fp)
        target_name = task.get_target_name()
        task_info = task.get_info()

        if verbose:
            PPRINTER.pprint(task_info)

        obs_timezone = task_info["obs_tz"]
        obs_lat = task_info["obs_lat"]
        obs_lon = task_info["obs_lon"]
        use_log10 = task_info["log_transform"]

        _, y_test = task.get_test_data(data_type="pd")

        if type(task) == tasks.RegressionTask:
            models = reg_models
        elif type(task) == tasks.ForecastingTask:
            models = fcn_models
            _, y_test = task.prepare_forecasting_data(_, y_test)
        else:
            raise ValueError(f"unknown task type {type(task)}.")

        benchmark_results[task_name] = {}
        benchmark_results[task_name]["possible_predictions"] = int(y_test.notna().sum().values[0])

        X_train, y_train = task.get_train_data(data_type="pd")
        X_val, y_val = task.get_validation_data(data_type="pd")
        X, y = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
        if type(task) == tasks.ForecastingTask:
            X, y = task.prepare_forecasting_data(X, y)

        if "mlo_cn2" in task_name:
            height_of_observation = 15.0
            air_temperature_col_name = "T_2m"
            water_temperature_col_name = ""
            humidity_col_name = "RH_2m"
            wind_speed_col_name = "Spd_10m"
            time_col_name = "time"
        elif "usna" in task_name:
            if "sm" in task_name:
                air_temperature_col_name = "T_5m"
                wind_speed_col_name = "Spd_10m"
            else:
                air_temperature_col_name = "T_3m"
                wind_speed_col_name = "Spd_3m"
            height_of_observation = 3.0
            water_temperature_col_name = "T_0m"
            humidity_col_name = "RH_3m"
            time_col_name = "time"
        else:
            raise ValueError(f"benchmarks not configured for task {task_name}.")

        for model_name, model in models.items():
            #if model_name != "RandomForestRegressionModel":
            if "AWT" in model_name and "mlo_cn2" in task_name:
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
                input_size=len(X.columns),
            )
            # if forecast model, add forecast horizon and window size
            if "forecasting" in task_name:
                model_kwargs["forecast_horizon"] = task.forecast_horizon
                model_kwargs["window_size"] = task.window_size
                model_kwargs["input_size"] = len(X.columns) // task.window_size
                model_kwargs["in_channels"] = task.window_size

            # adjust num epochs if provided
            if n_epochs_override is not None:
                model_kwargs["n_epochs"] = n_epochs_override

            mdl = model(**model_kwargs)
            mdl.train(X.copy(deep=True), y.copy(deep=True))  # copy to avoid modifying original data

            results = task.evaluate_model(predict_call=mdl.predict, x_transforms=None, x_transform_kwargs=None)
            benchmark_results[task_name][model_name] = results
            if verbose:
                print(f"Done running benchmark for {model_name}.")
            if verbose:
                PPRINTER.pprint(results)

    if write_metrics:
        with open(metrics_fp, "w") as f:
            f.write(json.dumps(benchmark_results, indent=4))
        if verbose:
            print(f"Wrote benchmark metrics to {metrics_fp}.")
    else:
        if verbose:
            print("Skipping writing benchmark metrics.")

    if verbose:
        print("Done running benchmarks.")
    if verbose:
        PPRINTER.pprint(benchmark_results)

    return benchmark_results

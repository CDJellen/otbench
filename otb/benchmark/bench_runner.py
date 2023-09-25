import os
import json
import pprint
from typing import Union

import pandas as pd

from otb.tasks import TaskApi, tasks
from otb.config import BENCHMARK_FP
from otb.benchmark.models import (
    MeanRegressionModel,
    MeanWindowForecastingModel,
    LinearForecastingModel,
    AWTModel,
    MacroMeterologicalModel,
    OffshoreMacroMeterologicalModel
)


REGRESSION_MODELS = {
    "mean_regression": MeanRegressionModel,
    "air_water_temperature_model": AWTModel,
    "macro_meterological": MacroMeterologicalModel,
    "offshore_macro_meterological": OffshoreMacroMeterologicalModel
}
FORECASTING_MODELS = {
    "mean_window_forecasting": MeanWindowForecastingModel,
    "linear_forecasting": LinearForecastingModel,
    "macro_meterological": MacroMeterologicalModel,
    "offshore_macro_meterological": OffshoreMacroMeterologicalModel
}
PPRINTER = pprint.PrettyPrinter(indent=4, width=120, compact=True)


def run_benchmarks(
    verbose: bool = True,
    write_metrics: bool = True,
    metrics_fp: Union[os.PathLike, str, None] = None
    ) -> dict:
    
    task_api = TaskApi()
    all_tasks = task_api.list_tasks()
    benchmark_results = {}

    for task_name in all_tasks:
        if verbose: print(f"Running benchmark for {task_name}...")

        task = task_api.get_task(task_name)
        target_name = task.get_target_name()
        task_info = task.get_info()
        
        if verbose: PPRINTER.pprint(task_info)
        
        obs_timezone = task_info["obs_tz"]
        obs_lat = task_info["obs_lat"]
        obs_lon = task_info["obs_lon"]
        use_log10 = task_info["log_transform"]

        _, y_test = task.get_test_data(data_type="pd")

        if type(task) == tasks.RegressionTask:
            models = REGRESSION_MODELS
        elif type(task) == tasks.ForecastingTask:
            models = FORECASTING_MODELS
            _, y_test = task.prepare_forecasting_data(_, y_test)
        else:
            raise ValueError(f"unknown task type {type(task)}.")
        
        benchmark_results[task_name] = {}
        benchmark_results[task_name]["possible_predictions"] = int(y_test.notna().sum().values[0])

        X_train, y_train = task.get_train_data(data_type="pd")
        X_val, y_val = task.get_val_data(data_type="pd")
        X, y = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])

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
            if "water" in model_name and "mlo_cn2" in task_name: continue
            if verbose: print(f"Running benchmark for {model_name}...")
            
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
                use_log10=use_log10
            )

            mdl = model(**model_kwargs)
            mdl.train(X, y)

            results = task.evaluate_model(predict_call=mdl.predict, x_transforms=None, x_transform_kwargs=None)
            benchmark_results[task_name][model_name] = results
            if verbose: print(f"Done running benchmark for {model_name}.")
            if verbose: PPRINTER.pprint(results)

    if write_metrics:
        if metrics_fp is None: metrics_fp = BENCHMARK_FP
        with open(metrics_fp, "w") as f:
            f.write(json.dumps(benchmark_results, indent=4))
        if verbose: print(f"Wrote benchmark metrics to {metrics_fp}.")
    else:
        if verbose: print("Skipping writing benchmark metrics.")

    if verbose: print("Done running benchmarks.")
    if verbose: PPRINTER.pprint(benchmark_results)
    return benchmark_results

if __name__ == "__main__":
    run_benchmarks()
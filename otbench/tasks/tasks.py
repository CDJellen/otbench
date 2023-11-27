import os
import json
import warnings
from abc import ABC
from datetime import datetime
from enum import Enum
from typing import Any, Callable, List, Tuple, Union

import pandas as pd

import otbench.eval.metrics as eval_metrics
from otbench.dataset import Dataset
from otbench.config import BENCHMARK_FP


class TaskTypes(Enum):
    REGRESSION = "regression"
    FORECASTING = "forecasting"


class TaskABC(ABC):

    def get_info(self, keys: Union[List[str], None] = None) -> dict:
        """Returns the full task information dictionary."""
        raise NotImplementedError

    def get_description(self) -> str:
        """Return the description of the task."""
        raise NotImplementedError

    def get_long_description(self) -> str:
        """Return the description of the task."""
        raise NotImplementedError

    def get_benchmark_info(self, task_name: Union[str, None]) -> dict:
        """Returns the benchmark information dictionary."""
        raise NotImplementedError

    def top_models(self, n: int = 5, metric: str = "") -> List[str]:
        """Returns the top n models for this task."""
        raise NotImplementedError

    def get_transforms(self) -> dict:
        """Return the description of the transforms applied to the X and y data."""
        raise NotImplementedError

    def get_target_name(self) -> str:
        """Return the target feature name for this task."""
        raise NotImplementedError

    def get_unavailable_features(self) -> List[str]:
        """Return the names of features which are unavailable for training and inference in this task."""
        raise NotImplementedError

    def get_metric_names(self) -> List[str]:
        """Return the target feature name for this task."""
        raise NotImplementedError

    def get_dataset(self) -> Dataset:
        """Return the underlying dataset."""
        raise NotImplementedError

    def get_df(self) -> pd.DataFrame:
        """Return the underlying pd.DataFrame for this task's dataset."""
        raise NotImplementedError

    def get_data(self, data_type: str) -> Any:
        """Return the underlying data."""
        raise NotImplementedError

    def get_train_data(self, data_type: str) -> Any:
        """Return the underlying training data for this task."""
        raise NotImplementedError

    def get_test_data(self, data_type: str) -> Any:
        """Return the underlying test data for this task."""
        raise NotImplementedError

    def get_validation_data(self, data_type: str) -> Any:
        """Return the underlying validation data for this task."""
        raise NotImplementedError

    def evaluate_model(predict_call: Callable,
                       data_type: str,
                       x_transforms: Union[Callable, None] = None,
                       x_transform_kwargs: Union[dict, None] = None,
                       eval_metric_names: Union[List[str], None] = None,
                       return_predictions: bool = False,
                       include_as_benchmark: bool = False,
                       model_name: Union[str, None] = None,
                       overwrite: bool = False) -> Union[dict, Tuple[dict, 'np.ndarray']]:
        """Evaluate a model against this task's transformed validation set, default against all metrics."""
        raise NotImplementedError


class BaseTask(TaskABC):

    def __init__(self, task_type: str, task_name: str, task: dict, benchmark_fp: str = BENCHMARK_FP) -> None:
        super().__init__()
        self.task_type = task_type
        self.task_name = task_name
        self.task = task
        self.benchmark_fp = benchmark_fp
        self._init_dataset_for_task()

    def get_info(self, keys: Union[List[str], None] = None) -> dict:
        """Returns the task information dictionary."""
        task = self.task
        return {k: task[k] for k in keys} if keys is not None else task

    def get_benchmark_info(self, task_name: Union[str, None] = None) -> dict:
        """Returns the benchmark information dictionary."""
        if os.path.exists(self.benchmark_fp):
            with open(self.benchmark_fp, "r") as f:
                benchmark_info = json.load(f)
            if task_name is None:
                task_name = self.task_name
            if task_name == "*":
                return benchmark_info
            return benchmark_info[task_name]
        else:
            raise FileNotFoundError(
                f"benchmark file {self.benchmark_fp} not found. Please run `otb benchmark` to generate the benchmark file."
            )

    def top_models(self, n: int = 5, metric: str = "") -> List[str]:
        """Returns the top n models for this task."""
        benchmark_info = self.get_benchmark_info(task_name=self.task_name)
        if metric == "":
            metrics = self.get_metric_names()
            if len(metrics) < 1:
                raise ValueError(f"task {self.task_name} has no metrics defined.")
            metric = metrics[0]
        model_benchmark_info = {k: v for k, v in benchmark_info.items() if k != "possible_predictions" and metric in v}
        model_benchmark_info_keys = sorted(model_benchmark_info.keys(),
                                           key=lambda x: model_benchmark_info[x][metric]["metric_value"])[:n]
        model_benchmark_info = {k: model_benchmark_info[k] for k in model_benchmark_info_keys}
        model_benchmark_info["possible_predictions"] = benchmark_info["possible_predictions"]

        return model_benchmark_info

    def get_description(self) -> str:
        """Return the description of the task."""
        return self.task["description"]

    def get_long_description(self) -> str:
        """Return the description of the task."""
        return self.task["description_long"]

    def get_transforms(self) -> dict:
        """Return the description of the transforms applied to the X and y data."""
        transform_info = {
            "log_transform": {
                "description": "Whether to apply a base-10 log transform to the target.",
                "value": self.task["log_transform"]
            },
            "dropna": {
                "description":
                    "Whether to drop observations with any missing values across both the features and target.",
                "value":
                    self.task["dropna"]
            }
        }

        return transform_info

    def get_target_name(self) -> str:
        """Return the target feature name for this task."""
        return self.task["target"]

    def get_unavailable_features(self) -> List[str]:
        """Return the names of features which are unavailable for training and inference in this task."""
        return self.task["remove"]

    def get_metric_names(self) -> List[str]:
        """Return the target feature name for this task."""
        return self.task["eval_metrics"]

    def get_dataset(self) -> Dataset:
        """Return the underlying dataset."""
        return self._ds

    def get_df(self) -> pd.DataFrame:
        """Return the underlying pd.DataFrame for this task's dataset."""
        return self._ds._df

    def get_data(self, data_type: str = "pd") -> Any:
        """Return the underlying data."""
        return self._ds.get_all(data_type=data_type)

    def get_train_data(self, data_type: str = "pd") -> Any:
        """Return the underlying training data for this task."""
        return self._ds.get_train(task=self.task, data_type=data_type)

    def get_test_data(self, data_type: str = "pd") -> Any:
        """Return the underlying test data for this task."""
        return self._ds.get_test(task=self.task, data_type=data_type)

    def get_validation_data(self, data_type: str = "pd") -> Any:
        """Return the underlying validation data for this task."""
        return self._ds.get_val(task=self.task, data_type=data_type)

    def evaluate_model(self,
                       predict_call: Callable,
                       data_type: str = "pd",
                       x_transforms: Union[Callable, None] = None,
                       x_transform_kwargs: Union[dict, None] = None,
                       eval_metric_names: Union[List[str], None] = None,
                       return_predictions: bool = False,
                       include_as_benchmark: bool = False,
                       model_name: Union[str, None] = None,
                       overwrite: bool = True) -> Union[dict, Tuple[dict, 'np.ndarray']]:
        """Evaluate a model against this task's transformed test set, default against all metrics."""
        raise NotImplementedError

    def _init_dataset_for_task(self) -> None:
        """Get the underlying dataset for the given task"""
        self._ds = Dataset(name=self.task["ds_name"])

    def _add_experiment_to_benchmarks(self, model_name: str, model_metrics: dict, overwrite: bool) -> None:
        if model_name is None:
            raise ValueError(f"model_name must be provided if include_as_benchmark is True.")
        benchmark_info = self.get_benchmark_info(task_name="*")
        if not overwrite and model_name in benchmark_info:
            model_name = f"{model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        benchmark_info[self.task_name][model_name] = model_metrics
        with open(self.benchmark_fp, "w") as f:
            json.dump(benchmark_info, f, indent=4)


class RegressionTask(BaseTask):

    def __init__(self, task_type: str, task_name: str, task: dict, benchmark_fp: str = BENCHMARK_FP) -> None:
        super().__init__(task_type=task_type, task_name=task_name, task=task, benchmark_fp=benchmark_fp)

    def evaluate_model(self,
                       predict_call: Callable,
                       predict_call_kwargs: Union[dict, None] = None,
                       data_type: str = "pd",
                       x_transforms: Union[Callable, None] = None,
                       x_transform_kwargs: Union[dict, None] = None,
                       eval_metric_names: Union[List[str], None] = None,
                       return_predictions: bool = False,
                       include_as_benchmark: bool = False,
                       model_name: Union[str, None] = None,
                       overwrite: bool = True) -> Union[dict, Tuple[dict, 'np.ndarray']]:
        """Evaluate a model against this task's transformed test set, default against all metrics."""
        # obtain evaluation data
        X_test, y_test = self.get_test_data(data_type=data_type)

        # apply x_transforms if present
        if x_transforms is not None:
            if x_transform_kwargs is not None:
                X_test = x_transforms(X_test, **x_transform_kwargs)
            else:
                X_test = x_transforms(X_test)

        # get predictions
        if predict_call_kwargs is not None:
            y_test_pred = predict_call(X_test, **predict_call_kwargs)
        else:
            y_test_pred = predict_call(X_test)

        if eval_metric_names is None:
            eval_metric_names = self.task["eval_metrics"]
        model_metrics = {k: -1 for k in eval_metric_names}

        for m in eval_metric_names:
            val = getattr(eval_metrics, m)(y_test, y_test_pred)
            model_metrics[m] = val

        if include_as_benchmark:
            self._add_experiment_to_benchmarks(model_name=model_name, model_metrics=model_metrics, overwrite=overwrite)
        if return_predictions:
            return model_metrics, y_test_pred
        return model_metrics


class ForecastingTask(BaseTask):

    def __init__(self, task_type: str, task_name: str, task: dict, benchmark_fp: str = BENCHMARK_FP) -> None:
        super().__init__(task_type=task_type, task_name=task_name, task=task, benchmark_fp=benchmark_fp)
        self.forecast_horizon = self.task["forecast_horizon"]
        self.window_size = self.task["window_size"]

    def prepare_forecasting_data(self,
                                 X: pd.DataFrame,
                                 y: Union[pd.DataFrame, pd.Series],
                                 window_size: Union[int, None] = None,
                                 forecast_horizon: Union[int, None] = None):
        """Prepare data for forecasting."""
        window_size = window_size if window_size is not None else self.window_size
        forecast_horizon = forecast_horizon if forecast_horizon is not None else self.forecast_horizon

        X = self._join_target(X, y)
        if window_size > 1:
            X = self._add_lags(X, (window_size - 1))
        y = self._shift_target(y, forecast_horizon)
        X, y = self._obtain_valid_data(X, y, (window_size - 1), forecast_horizon)

        return X, y

    def evaluate_model(self,
                       predict_call: Callable,
                       predict_call_kwargs: Union[dict, None] = None,
                       window_size: Union[int, None] = None,
                       forecast_horizon: Union[int, None] = None,
                       data_type: str = "pd",
                       x_transforms: Union[Callable, None] = None,
                       x_transform_kwargs: Union[dict, None] = None,
                       forecast_transforms: Union[Callable, None] = None,
                       forecast_transform_kwargs: Union[dict, None] = None,
                       eval_metric_names: Union[List[str], None] = None,
                       return_predictions: bool = False,
                       include_as_benchmark: bool = False,
                       model_name: Union[str, None] = None,
                       overwrite: bool = True) -> Union[dict, Tuple[dict, 'np.ndarray']]:
        """Evaluate a model against this task's transformed test set, default against all metrics."""
        window_size = window_size if window_size is not None else self.window_size
        forecast_horizon = forecast_horizon if forecast_horizon is not None else self.forecast_horizon

        # obtain evaluation data
        X_test, y_test = self.get_test_data(data_type=data_type)
        assert forecast_horizon + window_size < len(
            X_test
        ), f"window_size and forecast_horizon must be less than the length of the evaluation set ({len(X_test)})."

        # apply x_transforms if present
        if x_transforms is not None:
            if x_transform_kwargs is not None:
                X_test = x_transforms(X_test, **x_transform_kwargs)
            else:
                X_test = x_transforms(X_test)

        # apply window and forecasting horizon transforms
        if forecast_transforms is not None:
            if forecast_transform_kwargs is not None:
                y_test = forecast_transforms(y_test, **forecast_transform_kwargs)
            else:
                y_test = forecast_transforms(y_test)

        # obtain forecasting data
        X_test, y_test = self.prepare_forecasting_data(X_test, y_test, window_size, forecast_horizon)

        # get predictions
        if predict_call_kwargs is not None:
            y_test_pred = predict_call(X_test, **predict_call_kwargs)
        else:
            y_test_pred = predict_call(X_test)

        if eval_metric_names is None:
            eval_metric_names = self.task["eval_metrics"]
        model_metrics = {k: -1 for k in eval_metric_names}

        for m in eval_metric_names:
            val = getattr(eval_metrics, m)(y_test, y_test_pred)
            model_metrics[m] = val

        if include_as_benchmark:
            self._add_experiment_to_benchmarks(model_name=model_name, model_metrics=model_metrics, overwrite=overwrite)
        if return_predictions:
            return model_metrics, y_test_pred
        return model_metrics

    def _join_target(self, X, y):
        """Include the target in the features."""
        X = X.join(y)

        return X

    def _add_lags(self, X, window_size):
        """Lag all feature columns from 1 to lags inclusive."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            X = X.assign(
                **{f'{col} (t-{lag})': X[col].shift(lag) for lag in range(1, window_size + 1) for col in X.columns})

        return X

    def _shift_target(self, y, forecast_horizon):
        """Shift the target by the forecast horizon."""
        if forecast_horizon > 0:
            y = y.shift(-forecast_horizon)

        return y

    def _obtain_valid_data(self, X, y, window_size, forecast_horizon):
        """Obtain data that is valid for training and evaluation."""
        if forecast_horizon > 0:
            X = X[window_size:-1 * forecast_horizon]
            y = y[window_size:-1 * forecast_horizon]

        return X, y


class TaskApi(object):
    """A factory for creating optical turbulence modeling tasks."""

    def __init__(self, root_dir: Union[str, None] = None) -> None:
        """Read the currently-supported benchmarking task for loaders and evaluators."""
        if root_dir is None:
            root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))  # obtain root path
        tasks_path = os.path.join(root_dir, 'config', 'tasks.json')
        tasks = json.load(open(tasks_path, 'rb'))

        self.tasks = tasks
        self._build_task_names()

    def get_task(self, task_name: str, benchmark_fp: str = BENCHMARK_FP) -> Union[RegressionTask, ForecastingTask]:
        """Get a task by name."""
        if self._is_supported_task(task_name=task_name):
            if task_name.split(".")[0] == TaskTypes.REGRESSION.value:
                return RegressionTask(task_type=TaskTypes.REGRESSION,
                                      task_name=task_name,
                                      task=self._get_task(key=task_name),
                                      benchmark_fp=benchmark_fp)
            elif task_name.split(".")[0] == TaskTypes.FORECASTING.value:
                return ForecastingTask(task_type=TaskTypes.REGRESSION,
                                       task_name=task_name,
                                       task=self._get_task(key=task_name),
                                       benchmark_fp=benchmark_fp)
            else:
                raise NotImplementedError(f"task type {task_name.split('.')[0]} not supported.")
        else:
            raise NotImplementedError(f"task {task_name} not supported.")

    def list_tasks(self) -> List[str]:
        """List all currently supported tasks."""
        return list(self.task_names)

    def _get_task(self, key: str) -> dict:
        """Traverse a task key, returning bottom-level data."""
        path = key.split(".")
        d = self.tasks.copy()
        for i in range(len(path)):
            d = d[path[i]]
        return d

    def _is_supported_task(self, task_name: str) -> bool:
        """Check whether a task is supported by otbench."""
        return task_name in self.task_names

    def _build_task_names(self) -> None:
        """Gather `.` separated task names for easier inclusion checking."""
        task_names = set()
        task_paths = [(t, t, v) for t, v in self.tasks.items()]
        next_paths = []

        while task_paths or next_paths:
            if not task_paths:
                task_paths = next_paths
                next_paths = []

            _, full_k, v = task_paths.pop()

            if 'description' in v:
                task_names.add(full_k)
            else:
                next_paths.extend([(nk, f'{full_k}.{nk}', nv) for nk, nv in v.items()])

        self.task_names = task_names

import os
import json
from abc import ABC
from enum import Enum
from typing import Any, Callable, List, Union, Tuple

import pandas as pd

import otb.eval.metrics as eval
from otb.dataset import Dataset



class TaskTypes(Enum):
    REGRESSION = "regression"
    FORECASTING = "forecasting"
    INTERPOLATION = "interpolation"


class TaskABC(ABC):

    def get_all_info(self) -> dict:
        """Returns the full task information dictionary."""
        pass

    def get_description(self) -> str:
        """Return the description of the task."""
        pass

    def get_long_description(self) -> str:
        """Return the description of the task."""
        pass

    def get_transforms(self) -> dict:
        """Return the description of the transforms applied to the X and y data."""
        pass

    def get_target_name(self) -> str:
        """Return the target feature name for this task."""
        pass

    def get_unavailable_features(self) -> List[str]:
        """Return the names of features which are unavailable for training and inference in this task."""
        pass
    
    def get_metric_names(self) -> List[str]:
        """Return the target feature name for this task."""
        pass

    def get_dataset(self) -> Dataset:
        """Return the underlying dataset."""
        pass

    def get_df(self) -> pd.DataFrame:
        """Return the underlying pd.DataFrame for this task's dataset."""
        pass

    def get_data(self, data_type: str) -> Any:
        """Return the underlying data."""
        pass

    def get_train_data(self, data_type: str) -> Any:
        """Return the underlying training data for this task."""
        pass

    def get_test_data(self, data_type: str) -> Any:
        """Return the underlying test data for this task."""
        pass

    def get_val_data(self, data_type: str) -> Any:
        """Return the underlying validation data for this task."""
        pass

    def evaluate_model(self, predict_call: Callable, x_transforms: Union[Callable, None] = None, eval_metric_names: Union[List[str], None] = None) -> dict:
        """Evaluate a model against this task's transformed validation set, default against all metrics."""
        pass


class BaseTask(TaskABC):
    
    def __init__(self, task_type: str, task: dict) -> None:
        super().__init__()
        self.task_type = task_type,
        self.task = task
        self._init_dataset_for_task()

    def get_all_info(self) -> dict:
        """Returns the full task information dictionary."""
        return self.task

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
                "description": "Whether to drop observations with any missing values across both the features and target.",
                "value": self.task["dropna"]       
            }
        }

        return transform_info

    def get_target_name(self) -> str:
        """Return the target feature name for this task."""
        return self.task["target_name"]
    
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

    def get_val_data(self, data_type: str = "pd") -> Any:
        """Return the underlying validation data for this task."""
        return self._ds.get_val(task=self.task, data_type=data_type)

    def evaluate_model(self, predict_call: Callable, data_type: str = "pd", x_transforms: Union[Callable, None] = None, x_transform_kwargs: Union[dict, None] = None, eval_metric_names: Union[List[str], None] = None) -> dict:
        """Evaluate a model against this task's transformed validation set, default against all metrics."""
        raise NotImplementedError

    def _init_dataset_for_task(self) -> None:
        """Get the underlying dataset for the given task"""
        self._ds = Dataset(name=self.task["ds_name"])

class RegressionTask(BaseTask):

    def __init__(self, task_type: str, task: dict) -> None:
        super().__init__(task_type=task_type, task=task)

    def evaluate_model(self, predict_call: Callable, data_type: str = "pd", x_transforms: Union[Callable, None] = None, x_transform_kwargs: Union[dict, None] = None, eval_metric_names: Union[List[str], None] = None) -> dict:
        """Evaluate a model against this task's transformed validation set, default against all metrics."""
        # obtain evaluation data
        X_val, y_val = self.get_val_data(data_type=data_type)
        
        # apply x_transforms if present
        if x_transforms is not None:
            if x_transform_kwargs is not None:
                X_val = x_transforms(X_val, **x_transform_kwargs)
            else:
                X_val = x_transforms(X_val)
        
        # get predictions
        y_val_pred = predict_call(X_val)

        if eval_metric_names is None: eval_metric_names = self.task["eval_metrics"]
        model_metrics = {k: -1 for k in eval_metric_names}

        for m in eval_metric_names:
            val = getattr(eval, m)(y_val, y_val_pred)
            model_metrics[m] = val

        return model_metrics

class ForecastingTask(BaseTask):

    def __init__(self) -> None:
        super().__init__()


class InterpolationTask(BaseTask):

    def __init__(self) -> None:
        super().__init__()


class TaskFactory(object):
    """A factory for creating optical turbulence modeling tasks."""

    def __init__(self, root_dir: Union[str, None] = None) -> None:
        """Read the currently-supported benchmarking task for loaders and evaluators."""
        if root_dir is None:
            root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))  # obtain root path
        tasks_path = os.path.join(root_dir, 'config', 'tasks.json')
        tasks = json.load(open(tasks_path, 'rb'))

        self.tasks = tasks
        self._build_task_names()
    
    def get_task(self, task_name: str) -> Union[RegressionTask, ForecastingTask, InterpolationTask]:
        """Get a task by name."""
        if self._is_supported_task(task_name=task_name):
            # @TODO update factory pattern
            return RegressionTask(task_type=TaskTypes.REGRESSION, task=self._get_task(key=task_name))
        else: raise NotImplementedError

    def list_tasks(self) -> List[str]:
        """List all currently supported tasks."""
        return self.task_names

    def _get_task(self, key: str) -> dict:
        """Traverse a task key, returning bottom-level data."""
        path = key.split(".")
        d = self.tasks.copy()
        for i in range(len(path)):
            d = d[path[i]]
        return d

    def _is_supported_task(self, task_name: str) -> bool:
        """Check whether a task is supported by ot-benchmark."""
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

import pytest
import pandas as pd

from otb.tasks.tasks import TaskABC, BaseTask, RegressionTask, ForecastingTask
from tests import TESTS_BENCHMARK_FP


@pytest.mark.slow
def test_tasks(task_api):
    """Test the TaskApi."""
    # get a regression task
    task = task_api.get_task("regression.mlo_cn2.dropna.Cn2_15m", benchmark_fp=TESTS_BENCHMARK_FP)
    # check the task name
    assert task.task_name == "regression.mlo_cn2.dropna.Cn2_15m"
    assert isinstance(task, RegressionTask)
    assert isinstance(task.task, dict)

    # test task get info
    task_info = task.get_info()
    assert isinstance(task_info, dict)

    # test task get target name
    target_name = task.get_target_name()
    assert isinstance(target_name, str)
    assert len(target_name) > 0

    # test task get long description
    long_description = task.get_long_description()
    assert isinstance(long_description, str)
    assert len(long_description) > 0

    # test task get transforms
    transforms = task.get_transforms()
    assert isinstance(transforms, dict)
    assert len(transforms) > 0
    assert "log_transform" in transforms
    assert "dropna" in transforms

    # test task get metric names
    metric_names = task.get_metric_names()
    assert isinstance(metric_names, list)
    assert len(metric_names) > 0

    # test task get unavailable features
    unavailable_features = task.get_unavailable_features()
    assert isinstance(unavailable_features, list)
    assert len(unavailable_features) > 0

    # test task get dataset
    dataset = task.get_dataset()
    assert dataset is not None

    # test task get df
    df = task.get_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    # test task get train data
    X_train, y_train = task.get_train_data()
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.DataFrame)
    assert len(X_train) > 0
    assert len(y_train) > 0
    assert len(X_train) == len(y_train)
    assert len(X_train) < len(df)

    # test task get test data
    X_test, y_test = task.get_test_data()
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.DataFrame)
    assert len(X_test) > 0
    assert len(y_test) > 0
    assert len(X_test) == len(y_test)
    assert len(X_test) < len(df)

    # test task get val data
    X_val, y_val = task.get_val_data()
    assert isinstance(X_val, pd.DataFrame)
    assert isinstance(y_val, pd.DataFrame)
    assert len(X_val) > 0
    assert len(y_val) > 0
    assert len(X_val) == len(y_val)
    assert len(X_val) < len(df)

    # get task benchmark data
    benchmark_data = task.get_benchmark_info()
    assert isinstance(benchmark_data, dict)

    # get a forecasting task
    task = task_api.get_task("forecasting.mlo_cn2.dropna.Cn2_15m", benchmark_fp=TESTS_BENCHMARK_FP)
    # check the task name
    assert task.task_name == "forecasting.mlo_cn2.dropna.Cn2_15m"
    assert isinstance(task, ForecastingTask)
    assert isinstance(task.task, dict)

    # test task get info
    task_info = task.get_info()
    assert isinstance(task_info, dict)

    # test task get target name
    target_name = task.get_target_name()
    assert isinstance(target_name, str)
    assert len(target_name) > 0

    # test task get long description
    long_description = task.get_long_description()
    assert isinstance(long_description, str)
    assert len(long_description) > 0

    # test task get transforms
    transforms = task.get_transforms()
    assert isinstance(transforms, dict)
    assert len(transforms) > 0
    assert "log_transform" in transforms
    assert "dropna" in transforms

    # test task get metric names
    metric_names = task.get_metric_names()
    assert isinstance(metric_names, list)
    assert len(metric_names) > 0

    # test task get unavailable features
    unavailable_features = task.get_unavailable_features()
    assert isinstance(unavailable_features, list)
    assert len(unavailable_features) > 0

    # test task get dataset
    dataset = task.get_dataset()
    assert dataset is not None

    # test task get df
    df = task.get_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    # test task get train data
    X_train, y_train = task.get_train_data()
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.DataFrame)
    assert len(X_train) > 0
    assert len(y_train) > 0
    assert len(X_train) == len(y_train)
    assert len(X_train) < len(df)

    # test task get test data
    X_test, y_test = task.get_test_data()
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, pd.DataFrame)
    assert len(X_test) > 0
    assert len(y_test) > 0
    assert len(X_test) == len(y_test)
    assert len(X_test) < len(df)

    # test task get val data
    X_val, y_val = task.get_val_data()
    assert isinstance(X_val, pd.DataFrame)
    assert isinstance(y_val, pd.DataFrame)
    assert len(X_val) > 0
    assert len(y_val) > 0
    assert len(X_val) == len(y_val)
    assert len(X_val) < len(df)

    # get task benchmark data
    benchmark_data = task.get_benchmark_info()
    assert isinstance(benchmark_data, dict)

    # get a task that doesn't exist
    with pytest.raises(NotImplementedError):
        task = task_api.get_task("not_a_task")


def test_list_tasks(task_api):
    """Test listing all tasks."""
    assert isinstance(task_api.list_tasks(), list)


@pytest.mark.private
def test_is_supported_task(task_api):
    """Test the is_supported_task function."""
    assert task_api._is_supported_task("regression.mlo_cn2.dropna.Cn2_15m")
    assert task_api._is_supported_task("forecasting.mlo_cn2.dropna.Cn2_15m")
    assert not task_api._is_supported_task("not_a_task")


def test_task_abc():
    """Test the TaskABC."""
    task = TaskABC()
    with pytest.raises(NotImplementedError):
        task.get_info()
    with pytest.raises(NotImplementedError):
        task.get_target_name()
    with pytest.raises(NotImplementedError):
        task.get_description()
    with pytest.raises(NotImplementedError):
        task.get_long_description()
    with pytest.raises(NotImplementedError):
        task.get_transforms()
    with pytest.raises(NotImplementedError):
        task.get_metric_names()
    with pytest.raises(NotImplementedError):
        task.get_unavailable_features()
    with pytest.raises(NotImplementedError):
        task.get_dataset()
    with pytest.raises(NotImplementedError):
        task.get_df()
    with pytest.raises(NotImplementedError):
        task.get_data("foo")
    with pytest.raises(NotImplementedError):
        task.get_train_data("foo")
    with pytest.raises(NotImplementedError):
        task.get_test_data("foo")
    with pytest.raises(NotImplementedError):
        task.get_val_data("foo")
    with pytest.raises(NotImplementedError):
        task.evaluate_model(lambda x: x, "foo")
    with pytest.raises(NotImplementedError):
        task.get_benchmark_info("foo")
    with pytest.raises(NotImplementedError):
        task.top_models()
    with pytest.raises(NotImplementedError):
        task.get_metric_names()

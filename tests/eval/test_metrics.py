import pytest
import pandas as pd
import numpy as np

import otbench.eval.metrics as otb_metrics


@pytest.fixture(scope='module')
def metrics():
    """Return a function that yields a pointer to the metrics module."""
    yield otb_metrics


@pytest.fixture(scope='module')
def sample_data():
    y_true = pd.Series([1, 2, 3, 4, 5])
    y_pred = pd.Series([1, 2, 3, 4, 5])
    yield y_true, y_pred


def test_r2_score(metrics, sample_data):
    """Test the r2 score metric."""
    y_true, y_pred = sample_data
    want = {"metric_value": 1.0, "valid_predictions": 5}
    assert metrics.r2_score(y_true=y_true, y_pred=y_pred) == want


def test_root_mean_square_error(metrics, sample_data):
    """Test the root mean squared error metric."""
    y_true, y_pred = sample_data
    want = {"metric_value": 0.0, "valid_predictions": 5}
    assert metrics.root_mean_square_error(y_true=y_true, y_pred=y_pred) == want


def test_mean_absolute_error(metrics, sample_data):
    """Test the mean absolute error metric."""
    y_true, y_pred = sample_data
    want = {"metric_value": 0.0, "valid_predictions": 5}
    assert metrics.mean_absolute_error(y_true=y_true, y_pred=y_pred) == want


def test_mean_absolute_percentage_error(metrics, sample_data):
    """Test the mean absolute percentage error metric."""
    y_true, y_pred = sample_data
    want = {"metric_value": 0.0, "valid_predictions": 5}
    assert metrics.mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred) == want


def test_is_implemented_metric(metrics):
    """Test the is_implemented_metric function."""
    assert metrics.is_implemented_metric(metric_name="r2_score")
    assert metrics.is_implemented_metric(metric_name="root_mean_square_error")
    assert metrics.is_implemented_metric(metric_name="mean_absolute_error")
    assert metrics.is_implemented_metric(metric_name="mean_absolute_percentage_error")
    assert not metrics.is_implemented_metric(metric_name="not_implemented_metric")


def test_format_metric(metrics):
    """Test the _format_metric function."""
    assert metrics._format_metric(metric_value=1, valid_predictions=1) == {"metric_value": 1, "valid_predictions": 1}


def test_get_valid_indices(metrics, sample_data):
    """Test the _get_valid_indices function."""
    y_true, y_pred = sample_data
    assert len(metrics._get_valid_indices(y_true=y_true, y_pred=y_pred)) == len((y_true, y_pred))
    # add nan to y_true
    y_true[0] = np.nan
    assert len(metrics._get_valid_indices(y_true=y_true, y_pred=y_pred)) == len((y_true[:-1], y_pred[:-1]))
    with pytest.raises(ValueError):
        metrics._get_valid_indices(y_true=y_true, y_pred=y_pred[:-1])

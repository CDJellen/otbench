import pytest

import pandas as pd
import numpy as np
import xarray as xr

from otb.dataset import Dataset


@pytest.fixture
def dataset():
    """Yield a Dataset instance."""
    ds = Dataset(name="mlo_cn2",)
    yield ds


@pytest.fixture
def task():
    """Yield a task."""
    task = {
        "train_idx": ["1000:1010"],
        "val_idx": ["1010:1020"],
        "test_idx": ["2020:2030"],
        "target": "Cn2_15m",
        "remove": ["Spd_10m"],
        "dropna": True,
        "log_transform": True,
    }
    yield task


def test_get_slice(dataset):
    """Test the get_slice method."""
    start_indices = [0, 1, 2]
    end_indices = [1, 2, 3]
    df = dataset.get_slice(start_indices, end_indices)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    start_indices = [0, 1, 2]
    end_indices = [1, 2]
    with pytest.raises(ValueError):
        dataset.get_slice(start_indices, end_indices)


def test_get_all(dataset):
    """Test the get_all method."""
    df = dataset.get_all()
    assert isinstance(df, pd.DataFrame)


def test_get_train(dataset, task):
    """Test the get_train method."""
    X, y = dataset.get_train(task=task)
    assert isinstance(X, pd.DataFrame)
    assert len(X) == 10
    assert len(y) == 10
    assert "Spd_10m" not in X.columns


def test_get_test(dataset, task):
    """Test the get_test method."""
    X, y = dataset.get_test(task=task)
    assert isinstance(X, pd.DataFrame)
    assert len(X) == 10
    assert len(y) == 10
    assert "Spd_10m" not in X.columns


def test_get_val(dataset, task):
    """Test the get_val method."""
    X, y = dataset.get_val(task=task)
    assert isinstance(X, pd.DataFrame)
    assert len(X) == 10
    assert len(y) == 10
    assert "Spd_10m" not in X.columns


def test_handle_return_type(dataset):
    """Test the _handle_return_type method."""
    df = pd.DataFrame({"a": [1, 2, 3], "time": [1, 2, 3]})

    got = dataset._handle_return_type(df, "pd")
    assert isinstance(got, pd.DataFrame)

    got = dataset._handle_return_type(df, "np")
    assert isinstance(got, np.ndarray)

    got = dataset._handle_return_type(df, "xr")
    assert isinstance(got, xr.Dataset)

    got = dataset._handle_return_type(df, "nc")
    assert isinstance(got, xr.Dataset)

    with pytest.raises(NotImplementedError):
        dataset._handle_return_type(df, "foo")

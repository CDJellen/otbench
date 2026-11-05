from pathlib import Path

import pytest

import otbench.config as otb_config
from otbench.config import DatasetConfig, Task, settings


@pytest.fixture(scope='module')
def config():
    """Return a function that yields a pointer to the config."""
    yield otb_config


def test_config(config):
    """Test the attributes of the config."""
    assert isinstance(config.CACHE_DIR, str)
    assert isinstance(config.DATA_DIR, str)
    assert isinstance(config.DATASETS_FP, str)
    assert isinstance(config.RETURN_TYPES, list)
    assert isinstance(config.ROOT_DIR, str)
    assert isinstance(config.TASKS_FP, str)
    assert isinstance(config.BENCHMARK_FP, str)


def test_settings_paths():
    """Test that settings paths are correctly resolved as Path objects."""
    assert isinstance(settings.ROOT_DIR, Path)
    assert isinstance(settings.DATA_DIR, Path)
    assert settings.DATA_DIR.name == "data"
    assert settings.RETURN_TYPES == ["pd", "np", "xr", "nc"]


def test_task_model():
    """Test Task model validation."""
    task_data = {
        "train_idx": ["0:100"],
        "test_idx": ["100:120"],
        "val_idx": ["120:130"],
        "target": "Cn2_15m",
        "remove": ["col1"],
        "dropna": True,
        "log_transform": False,
    }
    task = Task(**task_data)
    assert task.train_idx == ["0:100"]
    assert task.target == "Cn2_15m"
    assert task.dropna is True


def test_task_model_defaults():
    """Test default values in Task model."""
    task_data = {
        "train_idx": ["0:100"],
        "test_idx": ["100:120"],
        "val_idx": ["120:130"],
        "target": "Cn2_15m",
    }
    task = Task(**task_data)
    assert task.remove == []
    assert task.dropna is True
    assert task.log_transform is False


def test_dataset_config_model():
    """Test DatasetConfig model."""
    config_data = {
        "local_data_path": "data.nc",
        "description": "A test dataset",
    }
    config = DatasetConfig(**config_data)
    assert config.local_data_path == "data.nc"
    assert config.description == "A test dataset"

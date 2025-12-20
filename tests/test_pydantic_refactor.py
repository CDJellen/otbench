from pathlib import Path
from otbench.config import settings
from otbench.models import Task, DatasetConfig

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
        "log_transform": False
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
        "target": "Cn2_15m"
    }
    task = Task(**task_data)
    assert task.remove == []
    assert task.dropna is True
    assert task.log_transform is False

def test_dataset_config_model():
    """Test DatasetConfig model."""
    config_data = {
        "local_data_path": "data.nc",
        "description": "A test dataset"
    }
    config = DatasetConfig(**config_data)
    assert config.local_data_path == "data.nc"
    assert config.description == "A test dataset"

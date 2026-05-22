import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from otbench.benchmark.models.regression.pytorch.base_pytorch_model import BasePyTorchRegressionModel


def test_base_pytorch_model_initialization():
    """Test initialization of BasePyTorchRegressionModel."""
    model = BasePyTorchRegressionModel(name="test_model", target_name="target")
    assert model.name == "test_model"
    assert model.target_name == "target"
    assert model.device.type in ["cpu", "cuda"]


def test_base_pytorch_model_abstract_methods():
    """Test that abstract methods raise NotImplementedError."""
    model = BasePyTorchRegressionModel(name="test_model", target_name="target")
    X = pd.DataFrame({"feature": [1, 2, 3]})
    y = pd.DataFrame({"target": [1, 2, 3]})
    
    with pytest.raises(NotImplementedError):
        model.train(X, y)
        
    with pytest.raises(NotImplementedError):
        model.predict(X)


def test_base_pytorch_model_set_model():
    """Test set_model method."""
    model = BasePyTorchRegressionModel(name="test_model", target_name="target", verbose=True)
    torch_model = nn.Linear(1, 1)
    # SGD requires params, so we must set_optimizer_callable_params=True
    model.set_model(torch_model, normalize_data=True, set_optimizer_callable_params=True)
    
    assert model.model == torch_model
    assert model.normalize_data is True
    assert model.optimizer is not None


def test_base_pytorch_model_data_handling():
    """Test data handling methods (set_training_data, _set_dataloader_from_data, etc.)."""
    model = BasePyTorchRegressionModel(name="test_model", target_name="target", window_size=1, batch_size=2)
    # Simple linear model to satisfy set_model requirement for optimizer creation
    model.set_model(nn.Linear(1, 1), set_optimizer_callable_params=True)
    
    X = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0]})
    y = pd.DataFrame({"target": [1.0, 2.0, 3.0, 4.0]})
    
    # Test set_training_data with DataFrame
    model.set_training_data(X, y)
    assert model.train_dataloader is not None
    assert len(model.train_dataloader) == 2 # 4 samples / 2 batch_size

    model.set_model(nn.Linear(1, 1), normalize_data=True, set_optimizer_callable_params=True)
    model.set_training_data(X, y)
    
    # Check if means and stds are calculated
    assert hasattr(model, "X_mean")
    assert hasattr(model, "X_std")
    
    # Test set_test_data
    model.set_test_data(X, y)
    assert model.test_dataloader is not None
    
    # Test set_validation_data
    model.set_validation_data(X, y)
    assert model.val_dataloader is not None


def test_base_pytorch_model_data_validation():
    """Test error handling in data methods."""
    model = BasePyTorchRegressionModel(name="test_model", target_name="target")
    X_df = pd.DataFrame({"feature": [1]})
    
    with pytest.raises(ValueError, match="y must be supplied"):
        model.set_training_data([1, 2, 3], None)

    with pytest.raises(ValueError, match="X and y must be both be pd.DataFrame objects or np.ndarray objects"):
        model.set_training_data(X_df, np.array([1]))

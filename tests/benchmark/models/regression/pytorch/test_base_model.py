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
    assert len(model.train_dataloader) == 2  # 4 samples / 2 batch_size

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


def test_regression_rnn_numpy_array_train_and_predict():
    """RNNModel (regression) must train, normalize, and predict identically on np.ndarray and pd.DataFrame."""
    from otbench.benchmark.models.regression.pytorch import RNNModel

    rng = np.random.default_rng(42)
    X_np = rng.normal(0.0, 1.0, size=(20, 4))
    y_np = rng.normal(0.0, 1.0, size=(20, 2))
    X_df = pd.DataFrame(X_np, columns=[f"f{i}" for i in range(4)])
    y_df = pd.DataFrame(y_np, columns=["t0", "t1"])

    mdl_np = RNNModel(
        name="rnn_np",
        target_name=["t0", "t1"],
        input_size=4,
        output_size=2,
        hidden_size=16,
        num_layers=1,
        n_epochs=3,
        batch_size=8,
        random_state=123,
    )
    mdl_np.train(X_np, y_np)
    pred_np = mdl_np.predict(X_np)
    assert pred_np.shape == (20, 2)
    assert np.all(np.isfinite(pred_np))

    mdl_df = RNNModel(
        name="rnn_df",
        target_name=["t0", "t1"],
        input_size=4,
        output_size=2,
        hidden_size=16,
        num_layers=1,
        n_epochs=3,
        batch_size=8,
        random_state=123,
    )
    mdl_df.train(X_df, y_df)
    pred_df = mdl_df.predict(X_df)
    np.testing.assert_allclose(pred_np, pred_df, atol=1e-5)


import pytest
import pandas as pd
import numpy as np
import torch
from otbench.benchmark.models.regression.pytorch.recurrent_neural_network import RNNModel as RegressionRNNModel
from otbench.benchmark.models.forecasting.pytorch.recurrent_neural_network import RNNModel as ForecastingRNNModel

def test_rnn_regression_init_output_size():
    """Test that RNNModel (regression) correctly accepts output_size and maps it to num_classes."""
    kwargs = {
        "name": "RegressionRNNModel",
        "target_name": "target",
        "input_size": 10,
        "output_size": 15,
        # num_classes defaulted to 1
    }
    model = RegressionRNNModel(**kwargs)
    assert model.num_classes == 15, "RNNModel should accept output_size and set num_classes accordingly."

def test_rnn_regression_vector_predict():
    """Test that RNNModel (regression) predict method handles vector outputs correctly."""
    output_dim = 3
    model = RegressionRNNModel(
        name="RegressionRNNModel",
        target_name="target",
        input_size=5,
        output_size=output_dim,
        verbose=False
    )
    
    # Dummy data
    X = pd.DataFrame(np.random.randn(10, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.DataFrame(np.random.randn(10, output_dim), columns=[f"t{i}" for i in range(output_dim)])
    
    # Train briefly
    model.train(X, y)
    
    # Predict
    preds = model.predict(X)
    
    # Check shape
    assert preds.shape == (10, output_dim), f"Detailed predictions shape mismatch. Expected (10, {output_dim}), got {preds.shape}"

def test_rnn_forecasting_init_output_size():
    """Test that RNNModel (forecasting) correctly accepts output_size and maps it to num_classes."""
    kwargs = {
        "name": "ForecastingRNNModel",
        "target_name": "target",
        "input_size": 10,
        "window_size": 5,
        "forecast_horizon": 1,
        "output_size": 15,
    }
    model = ForecastingRNNModel(**kwargs)
    assert model.num_classes == 15, "Forecasting RNNModel should accept output_size and set num_classes accordingly."

def test_rnn_forecasting_vector_predict():
    """Test that RNNModel (forecasting) predict method handles vector outputs correctly."""
    output_dim = 3
    model = ForecastingRNNModel(
        name="ForecastingRNNModel",
        target_name="target",
        input_size=5,
        window_size=2,
        forecast_horizon=1,
        output_size=output_dim,
        verbose=False
    )
    
    # Dummy data
    # For forecasting, input logic is handled by the model/task wrapper usually, 
    # but here we test the model directly.
    # BasePyTorchForecastingModel might expect X to be prepared already or not.
    # Actually BasePyTorchForecastingModel inherits form BasePyTorchModel?
    # No, it seems `otbench.benchmark.models.forecasting.pytorch.base_pytorch_model` exists.
    # Let's assume standard behavior:
    X = pd.DataFrame(np.random.randn(10, 10), columns=[f"f{i}" for i in range(10)]) # 10 features = 5 input_features * 2 window_size?
    y = pd.DataFrame(np.random.randn(10, output_dim), columns=[f"t{i}" for i in range(output_dim)])
    
    model.train(X, y)
    preds = model.predict(X)
    
    assert preds.shape == (10, output_dim), f"Forecasting predictions shape mismatch. Expected (10, {output_dim}), got {preds.shape}"

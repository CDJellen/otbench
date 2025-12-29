
import pytest
import numpy as np
import pandas as pd
import torch
from otbench.benchmark.models.regression.pytorch.recurrent_neural_network import RNNModel
from otbench.benchmark.models.regression.pytorch.base_pytorch_model import BasePyTorchRegressionModel

def test_normalization_vector():
    """Test that normalization works correctly for vector targets (multiple columns)."""
    # Create dummy data
    N = 100
    X = pd.DataFrame(np.random.randn(N, 5), columns=[f'x_{i}' for i in range(5)])
    # y has 2 columns with distinct means
    y = pd.DataFrame(np.concatenate([
        np.random.randn(N, 1) + 10,  # Mean approx 10
        np.random.randn(N, 1) - 10   # Mean approx -10
    ], axis=1), columns=['y1', 'y2'])
    
    model = RNNModel("test", "target", 5, output_size=2)
    # Using private methods for unit testing logic
    model.X_mean = None
    model.y_mean = None
    
    X_norm, y_norm = model._normalize_data(X.copy(), y.copy())
    
    # Check if means were calculated per column
    assert model.y_mean.shape == (2,)
    assert np.abs(model.y_mean[0] - 10) < 1.0
    assert np.abs(model.y_mean[1] - (-10)) < 1.0
    
    # Check if normalized data has mean approx 0
    assert np.allclose(np.mean(y_norm, axis=0), 0, atol=0.1)

def test_rnn_vector_mismatch_error():
    """Test that RNNModel raises ValueError if y dimension doesn't match output_size."""
    N = 20
    X = pd.DataFrame(np.random.randn(N, 5))
    y = pd.DataFrame(np.random.randn(N, 3)) # 3 targets
    
    # Model expects 1 target by default or if specified
    model = RNNModel("test", "target", 5, output_size=1)
    
    with pytest.raises(ValueError, match="Model initialized with output_size=1"):
        model.train(X, y)

def test_rnn_vector_output_shape():
    """Test RNNModel produces correct output shape for vector targets."""
    N = 20
    X = pd.DataFrame(np.random.randn(N, 5), columns=[f'x_{i}' for i in range(5)])
    y = pd.DataFrame(np.random.randn(N, 3), columns=[f'y_{i}' for i in range(3)])
    
    model = RNNModel("test", "target", input_size=5, output_size=3, n_epochs=1, batch_size=5)
    model.train(X, y)
    preds = model.predict(X)
    
    assert preds.shape == (N, 3)

def test_rnn_scalar_output_shape():
    """Test RNNModel produces correct output shape for scalar target (backward compatibility)."""
    N = 20
    X = pd.DataFrame(np.random.randn(N, 5), columns=[f'x_{i}' for i in range(5)])
    y = pd.DataFrame(np.random.randn(N, 1), columns=['y1'])
    
    model = RNNModel("test", "target", input_size=5, output_size=1, n_epochs=1, batch_size=5)
    model.train(X, y)
    preds = model.predict(X)
    
    # Should be (N,) or (N, 1)? Original code seemed to produce (N,) for scalars?
    # Let's check what it does now. We want consistency. 
    # If the original code did `pred.append(y_pred[0][0])` it implies (N,).
    # New code does the same if num_classes == 1.
    assert preds.shape == (N,)

if __name__ == "__main__":
    # Manually run if executed as script
    try:
        test_normalization_vector()
        print("test_normalization_vector passed")
        test_rnn_vector_mismatch_error()
        print("test_rnn_vector_mismatch_error passed")
        test_rnn_vector_output_shape()
        print("test_rnn_vector_output_shape passed")
        test_rnn_scalar_output_shape()
        print("test_rnn_scalar_output_shape passed")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

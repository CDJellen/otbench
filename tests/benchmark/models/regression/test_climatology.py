import pytest
import pandas as pd
import numpy as np
from otbench.benchmark.models.regression.climatology import ClimatologyRegressionModel

def test_climatology_initialization():
    """Test initialization."""
    model = ClimatologyRegressionModel(name="test_clim", target_name="target")
    assert model.name == "test_clim"
    assert model.target_name == "target"
    assert np.isnan(model.global_mean)

def test_climatology_scalar_train_predict():
    """Test training and prediction with scalar inputs."""
    model = ClimatologyRegressionModel(name="test_clim", target_name="target")
    X = pd.DataFrame({"feature": range(10)})
    
    # y as Series
    y_series = pd.Series(range(10)) # mean = 4.5
    model.train(X, y_series)
    assert model.global_mean == 4.5
    
    preds = model.predict(X)
    assert len(preds) == 10
    assert np.all(preds == 4.5)
    
    # y as DataFrame
    y_df = pd.DataFrame({"target": range(10)})
    model.train(X, y_df)
    assert model.global_mean == 4.5

    # y as numpy array
    y_np = np.arange(10)
    model.train(X, y_np)
    assert model.global_mean == 4.5

def test_climatology_vector_train_predict():
    """Test training and prediction with vector inputs."""
    model = ClimatologyRegressionModel(name="test_clim", target_name="target")
    X = pd.DataFrame({"feature": range(2)})
    
    # y as 2D array (sample, features)
    # Sample 1: [1, 10]
    # Sample 2: [3, 30]
    # Mean: [2, 20]
    y_vector = np.array([[1, 10], [3, 30]])
    
    model.train(X, y_vector)
    assert np.all(model.global_mean == np.array([2.0, 20.0]))
    
    preds = model.predict(X)
    assert preds.shape == (2, 2)
    assert np.all(preds[0] == [2.0, 20.0])
    assert np.all(preds[1] == [2.0, 20.0])

def test_climatology_nan_handling():
    """Test handling of NaNs in training data."""
    model = ClimatologyRegressionModel(name="test_clim", target_name="target")
    X = pd.DataFrame({"feature": range(3)})
    y = np.array([1.0, 3.0, np.nan]) # mean of 1 and 3 is 2
    
    model.train(X, y)
    assert model.global_mean == 2.0

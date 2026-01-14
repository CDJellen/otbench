import pytest
import pandas as pd
import numpy as np

from otbench.benchmark.models.regression.minute_climatology import MinuteClimatologyRegressionModel


def test_minute_climatology_initialization():
    """Test initialization."""
    model = MinuteClimatologyRegressionModel(name="test_minute_clim", target_name="target")
    assert model.name == "test_minute_clim"
    assert model.target_name == "target"
    assert model.means == {}

def test_minute_climatology_train_predict():
    """Test basic training and prediction flow."""
    model = MinuteClimatologyRegressionModel(name="test_minute_clim", target_name="target")
    
    # Create data covering 2 days, 2 minutes each day (00:00 and 00:01)
    dates = pd.date_range(start="2024-01-01 00:00:00", periods=4, freq="min")
    X = pd.DataFrame({"feature": [1, 1, 1, 1]}, index=dates)

    dates_d1 = pd.date_range(start="2024-01-01 00:00:00", periods=2, freq="min")
    dates_d2 = pd.date_range(start="2024-01-02 00:00:00", periods=2, freq="min")
    dates = dates_d1.union(dates_d2)
    
    X = pd.DataFrame({"feature": range(4)}, index=dates)
    # 00:00 values: 10 (on D1), 12 (on D2) -> Mean 11
    # 00:01 values: 20 (on D1), 22 (on D2) -> Mean 21
    y = pd.Series([10, 20, 12, 22], index=dates, name="target")
    
    model.train(X, y)
    
    # Check internal means
    import datetime
    t0 = datetime.time(0, 0)
    t1 = datetime.time(0, 1)
    
    assert t0 in model.means
    assert t1 in model.means
    # Means stored as numpy arrays
    assert model.means[t0] == 11.0
    assert model.means[t1] == 21.0
    assert model.global_mean == 16.0
    
    # Predict on known times
    preds = model.predict(X)
    assert preds[0] == 11.0
    assert preds[1] == 21.0
    
    # Predict on unknown time (00:05) -> should fallback to global mean
    X_new = pd.DataFrame({"feature": [1]}, index=[pd.Timestamp("2024-01-03 00:05:00")])
    pred_new = model.predict(X_new)
    assert pred_new[0] == 16.0


def test_minute_climatology_input_formats():
    """Test handling of various input formats for y (Series, DataFrame, ndarray)."""
    model = MinuteClimatologyRegressionModel(name="test_minute_clim", target_name="target")
    dates = pd.date_range(start="2024-01-01", periods=2, freq="min")
    X = pd.DataFrame({"feature": [1, 1]}, index=dates)
    
    # Test with numpy array
    y_np = np.array([10, 20])
    model.train(X, y_np)
    assert model.global_mean == 15.0
    
    # Test with DataFrame
    y_df = pd.DataFrame({"target": [10, 20]}, index=dates)
    model.train(X, y_df)
    assert model.global_mean == 15.0
    
    # Test with 2D numpy array
    y_np_2d = np.array([[10], [20]])
    model.train(X, y_np_2d)
    assert model.global_mean == 15.0


def test_minute_climatology_nan_handling():
    """Test fallback when specific time mean is NaN."""
    model = MinuteClimatologyRegressionModel(name="test_minute_clim", target_name="target")
    dates = pd.date_range(start="2024-01-01", periods=2, freq="min")
    X = pd.DataFrame({"feature": [1, 1]}, index=dates)
    
    # Create y where one time slot is ALL NaNs across history (only 1 day here)
    y = pd.Series([np.nan, 20.0], index=dates, name="target")
    
    model.train(X, y)
    
    # Global mean should ignore NaNs -> 20.0
    assert model.global_mean == 20.0
    
    # Predict for the NaN time -> should return global mean
    preds = model.predict(X)
    assert preds[0] == 20.0 
    assert preds[1] == 20.0

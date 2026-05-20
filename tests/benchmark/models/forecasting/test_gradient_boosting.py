import pytest
import numpy as np
import pandas as pd

from otbench.benchmark.models.forecasting.gradient_boosting_regression_tree import (
    GradientBoostingForecastingModel,
)


def test_instantiation_with_extra_kwargs():
    """GBRT must not crash when given framework kwargs it doesn't consume."""
    model = GradientBoostingForecastingModel(
        name="gbrt",
        target_name="seeing",
        window_size=5,
        forecast_horizon=1,
        output_size=1,
        # These framework keys should be silently filtered:
        input_size=10,
        predict_residuals=True,
        use_log10=True,
        timezone="UTC",
        obs_lat=-24.6,
        obs_lon=-70.4,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        batch_size=32,
        n_epochs=10,
        learning_rate=0.001,
        hidden_size=64,
    )
    assert model.name == "gbrt"


def test_scalar_train_predict():
    """Scalar target: train and predict produce correct shapes."""
    rng = np.random.default_rng(42)
    X_train = pd.DataFrame(rng.standard_normal((50, 5)), columns=[f"f{i}" for i in range(5)])
    y_train = rng.standard_normal(50)
    X_test = pd.DataFrame(rng.standard_normal((10, 5)), columns=[f"f{i}" for i in range(5)])

    model = GradientBoostingForecastingModel(
        name="gbrt", target_name="target", window_size=1, forecast_horizon=1, output_size=1,
    )
    model._train(X_train, y_train)
    preds = model._predict(X_test)
    assert preds.shape == (10,)


def test_vector_train_predict():
    """Vector target: train and predict produce correct shapes."""
    rng = np.random.default_rng(42)
    X_train = pd.DataFrame(rng.standard_normal((50, 5)), columns=[f"f{i}" for i in range(5)])
    y_train = pd.DataFrame(rng.standard_normal((50, 3)), columns=["t0", "t1", "t2"])
    X_test = pd.DataFrame(rng.standard_normal((10, 5)), columns=[f"f{i}" for i in range(5)])

    model = GradientBoostingForecastingModel(
        name="gbrt", target_name=["t0", "t1", "t2"],
        window_size=1, forecast_horizon=1, output_size=3,
    )
    model._train(X_train, y_train)
    preds = model._predict(X_test)
    assert preds.shape == (10, 3)


def test_empty_train_predict():
    """Empty input must not crash."""
    model = GradientBoostingForecastingModel(
        name="gbrt", target_name="target", window_size=1, forecast_horizon=1, output_size=1,
    )
    X_empty = pd.DataFrame(columns=["f0", "f1"])
    model._train(X_empty, np.array([]))
    preds = model._predict(X_empty)
    assert len(preds) == 0


def test_nan_in_y_scalar_is_silently_dropped():
    """Scalar target: rows with NaN y must be dropped before fit, not crash."""
    rng = np.random.default_rng(7)
    X_train = pd.DataFrame(rng.standard_normal((50, 4)), columns=[f"f{i}" for i in range(4)])
    y_train = pd.Series(rng.standard_normal(50), name="t")
    y_train.iloc[5:10] = float("nan")

    model = GradientBoostingForecastingModel(
        name="gbrt", target_name="t", window_size=1, forecast_horizon=1, output_size=1,
    )
    model._train(X_train, y_train)  # must not raise

    X_test = pd.DataFrame(rng.standard_normal((8, 4)), columns=[f"f{i}" for i in range(4)])
    preds = model._predict(X_test)
    assert preds.shape == (8,)
    assert not np.isnan(preds).any()


def test_nan_in_y_vector_is_silently_dropped():
    """Vector target: rows where any target column is NaN must be dropped."""
    rng = np.random.default_rng(9)
    X_train = pd.DataFrame(rng.standard_normal((60, 5)), columns=[f"f{i}" for i in range(5)])
    y_train = pd.DataFrame(rng.standard_normal((60, 3)), columns=["t0", "t1", "t2"])
    y_train.iloc[10:15, 0] = float("nan")
    y_train.iloc[30:33] = float("nan")

    model = GradientBoostingForecastingModel(
        name="gbrt", target_name=["t0", "t1", "t2"],
        window_size=1, forecast_horizon=1, output_size=3,
    )
    model._train(X_train, y_train)  # must not raise

    X_test = pd.DataFrame(rng.standard_normal((10, 5)), columns=[f"f{i}" for i in range(5)])
    preds = model._predict(X_test)
    assert preds.shape == (10, 3)
    assert not np.isnan(preds).any()


def test_all_nan_y_returns_gracefully():
    """When every target row is NaN, _train must return without error."""
    X_train = pd.DataFrame({"f0": [1.0, 2.0, 3.0]})
    y_train = pd.DataFrame({"t0": [float("nan")] * 3, "t1": [float("nan")] * 3})
    model = GradientBoostingForecastingModel(
        name="gbrt", target_name=["t0", "t1"], window_size=1, forecast_horizon=1, output_size=2,
    )
    model._train(X_train, y_train)  # must not raise

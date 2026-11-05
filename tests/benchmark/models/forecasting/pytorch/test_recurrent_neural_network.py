import numpy as np
import pandas as pd
import pytest
import torch.nn as nn
import torch.optim as optim

from otbench.benchmark.models.forecasting.pytorch import TransformerModel
from otbench.benchmark.models.forecasting.pytorch.recurrent_neural_network import RNNModel
from otbench.benchmark.models.regression.pytorch.recurrent_neural_network import (
    RNNModel as RegressionRNNModel,)


@pytest.mark.slow
def test_rnn_model(task_api):
    """Test the RNNModel."""
    bs = 32
    X = pd.DataFrame({
        "T_10m": [10 for _ in range(50 * bs)],
        "T_0m": [10 for _ in range(50 * bs)],
    })
    y = pd.DataFrame({"Cn2_15m": [1.58e-16 for _ in range(50 * bs)]})

    X_train, X_test = X[0:40 * bs], X[40 * bs:]
    y_train, y_test = y[0:40 * bs], y[40 * bs:]

    task = task_api.get_task("forecasting.mlo_cn2.dropna.Cn2_15m")
    X_train, y_train = task.prepare_forecasting_data(X_train, y_train)
    X_test, y_test = task.prepare_forecasting_data(X_test, y_test)

    model = RNNModel(
        name="rnn_model",
        window_size=task.window_size,
        forecast_horizon=task.forecast_horizon,
        target_name="Cn2_15m",
        input_size=(len(X_train.columns) // task.window_size),
        obs_window_size=1,
        hidden_size=32,
        num_layers=1,
        num_classes=1,
        batch_size=bs,
        n_epochs=50,
        criterion=nn.MSELoss(),
        learning_rate=0.025,
        optimizer=optim.SGD,
        verbose=True,
    )

    assert model.name == "rnn_model"

    model.train(X_train, y_train)
    predictions = model.predict(X_test)
    assert isinstance(predictions, pd.Series) or isinstance(predictions, np.ndarray)
    assert len(predictions) == len(y_test)
    assert np.allclose(predictions, y_test.values.ravel())


def test_rnn_regression_init_output_size():
    """Test that RNNModel (regression) accepts output_size and maps it to num_classes."""
    kwargs = {
        "name": "RegressionRNNModel",
        "target_name": "target",
        "input_size": 10,
        "output_size": 15,
    }
    model = RegressionRNNModel(**kwargs)
    assert model.num_classes == 15


def test_rnn_regression_vector_predict():
    """Test that RNNModel (regression) predict handles vector outputs."""
    output_dim = 3
    model = RegressionRNNModel(
        name="RegressionRNNModel",
        target_name="target",
        input_size=5,
        output_size=output_dim,
        hidden_size=32,
        n_epochs=5,
        verbose=False,
    )

    X = pd.DataFrame(np.random.randn(10, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.DataFrame(np.random.randn(10, output_dim), columns=[f"t{i}" for i in range(output_dim)])

    model.train(X, y)
    preds = model.predict(X)
    assert preds.shape == (10, output_dim)


def test_rnn_forecasting_init_output_size():
    """Test that RNNModel (forecasting) accepts output_size and maps it to num_classes."""
    kwargs = {
        "name": "ForecastingRNNModel",
        "target_name": "target",
        "input_size": 10,
        "window_size": 5,
        "forecast_horizon": 1,
        "output_size": 15,
    }
    model = RNNModel(**kwargs)
    assert model.num_classes == 15


def test_rnn_forecasting_vector_predict():
    """Test that RNNModel (forecasting) predict handles vector outputs."""
    output_dim = 3
    model = RNNModel(
        name="ForecastingRNNModel",
        target_name="target",
        input_size=5,
        window_size=2,
        forecast_horizon=1,
        output_size=output_dim,
        hidden_size=32,
        n_epochs=5,
        verbose=False,
    )

    X = pd.DataFrame(np.random.randn(10, 10), columns=[f"f{i}" for i in range(10)])
    y = pd.DataFrame(np.random.randn(10, output_dim), columns=[f"t{i}" for i in range(output_dim)])

    model.train(X, y)
    preds = model.predict(X)
    assert preds.shape == (10, output_dim)


@pytest.mark.parametrize("cell_type", ["rnn", "gru", "lstm"])
def test_rnn_forecasting_cell_types_and_protocol(cell_type):
    """Test RNNModel with rnn/gru/lstm cells, early stopping protocol, and log10 residual learning."""
    rng = np.random.default_rng(0)
    n_samples, window_size, n_features, output_dim = 24, 3, 2, 2
    X_linear = 10.0**rng.uniform(-16.0, -14.0, size=(n_samples, window_size * n_features))
    cols = [f"cn2_layer{j} (t-{w})" for w in range(window_size - 1, -1, -1) for j in range(n_features)]
    X = pd.DataFrame(X_linear, columns=cols)
    y = pd.DataFrame(
        rng.uniform(-16.0, -14.0, size=(n_samples, output_dim)),
        columns=["cn2_layer0", "cn2_layer1"],
    )

    model = RNNModel(
        name=f"Forecasting_{cell_type.upper()}",
        target_name=["cn2_layer0", "cn2_layer1"],
        input_size=n_features,
        window_size=window_size,
        forecast_horizon=1,
        output_size=output_dim,
        cell_type=cell_type,
        hidden_size=16,
        num_layers=1,
        batch_size=8,
        n_epochs=6,
        patience=2,
        val_fraction=0.25,
        use_scheduler=True,
        weight_decay=1e-2,
        predict_residuals=True,
        use_log10=True,
        verbose=False,
    )
    model.train(X, y)
    assert np.all(model.X_mean < -10.0)
    preds = model.predict(X)
    assert preds.shape == (n_samples, output_dim)
    assert np.all(np.isfinite(preds))


def test_transformer_forecasting_vector_predict_and_log10():
    """Test TransformerModel vector forecasting with shared training protocol and log10 transform."""
    rng = np.random.default_rng(1)
    n_samples, window_size, n_features, output_dim = 24, 3, 2, 2
    X_linear = 10.0**rng.uniform(-16.0, -14.0, size=(n_samples, window_size * n_features))
    cols = [f"cn2_layer{j} (t-{w})" for w in range(window_size - 1, -1, -1) for j in range(n_features)]
    X = pd.DataFrame(X_linear, columns=cols)
    y = pd.DataFrame(
        rng.uniform(-16.0, -14.0, size=(n_samples, output_dim)),
        columns=["cn2_layer0", "cn2_layer1"],
    )

    model = TransformerModel(
        name="Transformer_Residual",
        target_name=["cn2_layer0", "cn2_layer1"],
        input_size=n_features,
        window_size=window_size,
        forecast_horizon=1,
        output_size=output_dim,
        d_model=16,
        nhead=4,
        num_layers=1,
        dropout=0.1,
        batch_size=8,
        n_epochs=6,
        patience=2,
        val_fraction=0.25,
        use_scheduler=True,
        weight_decay=1e-2,
        predict_residuals=True,
        use_log10=True,
        verbose=False,
    )
    model.train(X, y)
    assert np.all(model.X_mean < -10.0)
    preds = model.predict(X)
    assert preds.shape == (n_samples, output_dim)
    assert np.all(np.isfinite(preds))

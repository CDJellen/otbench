import pytest

import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim

from otb.benchmark.models.forecasting.pytorch.convolutional_neural_network import CNNModel


@pytest.mark.slow
def test_cnn_model(task_api):
    """Test the CNNModel."""
    # create a DataFrame with sample columns
    bs = 32
    # create a DataFrame with the required columns
    X = pd.DataFrame({
        "T_40m": [10, 20, 30, 40, 50] * bs,
        "T_30m": [10, 20, 30, 40, 50] * bs,
        "T_20m": [10, 20, 30, 40, 50] * bs,
        "T_10m": [10, 20, 30, 40, 50] * bs,
        "T_0m": [10, 20, 30, 40, 50] * bs,
    })
    y = pd.DataFrame({"Cn2_15m": [1.58e-16, 1.58e-16, 1.58e-16, 1.58e-16, 1.58e-16] * bs})

    X_train, X_test = X[0:4*bs], X[4*bs:]
    y_train, y_test = y[0:4*bs], y[4*bs:]

    # we need a task to prepare forecasting data
    task = task_api.get_task("forecasting.mlo_cn2.dropna.Cn2_15m")
    X_train, y_train = task.prepare_forecasting_data(X_train, y_train)
    X_test, y_test = task.prepare_forecasting_data(X_test, y_test)

    model = CNNModel(
        name="cnn_model",
        target_name="Cn2_15m",
        forecast_horizon=task.forecast_horizon,
        input_size=(len(X_train.columns) // task.window_size),  # single row
        window_size=task.window_size,  # single row
        in_channels=task.window_size,
        out_channels=1,
        kernel_size=3,
        stride=1,
        padding="valid",
        bias=False,
        batch_size=bs,
        n_epochs=50,
        criterion = nn.MSELoss(),
        learning_rate=0.025,
        optimizer = optim.SGD,
        verbose=True,
    )

    # check the model name
    assert model.name == "cnn_model"

    # train the model
    model.train(X_train, y_train)
    # predict
    predictions = model.predict(X_test)
    # check the predictions
    assert (isinstance(predictions, pd.Series) or isinstance(predictions, np.ndarray))
    assert len(predictions) == len(y_test)
    assert np.allclose(predictions, y_test.values.ravel())


@pytest.mark.slow
def test_cnn_model_kernel_size_1(task_api):
    """Test the CNNModel."""
    # create a DataFrame with sample columns
    bs = 32
    # create a DataFrame with the required columns
    X = pd.DataFrame({
        "T_40m": [10, 20, 30, 40, 50] * bs,
        "T_30m": [10, 20, 30, 40, 50] * bs,
        "T_20m": [10, 20, 30, 40, 50] * bs,
        "T_10m": [10, 20, 30, 40, 50] * bs,
        "T_0m": [10, 20, 30, 40, 50] * bs,
    })
    y = pd.DataFrame({"Cn2_15m": [1.58e-16, 1.58e-16, 1.58e-16, 1.58e-16, 1.58e-16] * bs})

    X_train, X_test = X[0:4*bs], X[4*bs:]
    y_train, y_test = y[0:4*bs], y[4*bs:]

    # we need a task to prepare forecasting data
    task = task_api.get_task("forecasting.mlo_cn2.dropna.Cn2_15m")
    X_train, y_train = task.prepare_forecasting_data(X_train, y_train)
    X_test, y_test = task.prepare_forecasting_data(X_test, y_test)

    model = CNNModel(
        name="cnn_model",
        target_name="Cn2_15m",
        forecast_horizon=task.forecast_horizon,
        input_size=(len(X_train.columns) // task.window_size),  # single row
        window_size=task.window_size,  # single row
        in_channels=task.window_size,
        out_channels=1,
        kernel_size=1,
        stride=1,
        padding="valid",
        bias=False,
        batch_size=bs,
        n_epochs=50,
        criterion = nn.MSELoss(),
        learning_rate=0.025,
        optimizer = optim.SGD,
        verbose=True,
    )

    # check the model name
    assert model.name == "cnn_model"

    # train the model
    model.train(X_train, y_train)
    # predict
    predictions = model.predict(X_test)
    # check the predictions
    assert (isinstance(predictions, pd.Series) or isinstance(predictions, np.ndarray))
    assert len(predictions) == len(y_test)
    assert np.allclose(predictions, y_test.values.ravel())

def test_cnn_model_fail_assertion():
    """Test the CNNModel."""
    # create a DataFrame with sample columns
    bs = 1
    X = pd.DataFrame({
        "T_10m": [10, 20, 30, 40, 50] * bs,
        "T_0m": [10, 20, 30, 40, 50] * bs,
    })

    with pytest.raises(AssertionError):
        model = CNNModel(
            name="cnn_model",
            target_name="Cn2_15m",
            forecast_horizon=1,
            input_size=len(X.columns),  # single row
            window_size=1,  # single row
            in_channels=1,
            out_channels=1,
            kernel_size=99,
            stride=1,
            padding="valid",
            bias=False,
            batch_size=bs,
            n_epochs=50,
            criterion = nn.MSELoss(),
            learning_rate=0.025,
            optimizer = optim.SGD,
            verbose=True,
        )
    with pytest.raises(AssertionError):
        model = CNNModel(
            name="cnn_model",
            target_name="Cn2_15m",
            forecast_horizon=1,
            input_size=0,  # error
            window_size=1,  # single row
            in_channels=1,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding="valid",
            bias=False,
            batch_size=bs,
            n_epochs=50,
            criterion = nn.MSELoss(),
            learning_rate=0.025,
            optimizer = optim.SGD,
            verbose=True,
        )

from otb.tasks import TaskApi
test_cnn_model(TaskApi())
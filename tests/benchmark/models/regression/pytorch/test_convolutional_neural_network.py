import pytest

import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim

from otb.benchmark.models.regression.pytorch.convolutional_neural_network import CNNModel


@pytest.mark.slow
def test_cnn_model():
    """Test the CNNModel."""
    # create a DataFrame with sample columns
    bs = 32
    X = pd.DataFrame({
        "T_40m": [10, 20, 30, 40, 50] * bs,
        "T_30m": [10, 20, 30, 40, 50] * bs,
        "T_20m": [10, 20, 30, 40, 50] * bs,
        "T_10m": [10, 20, 30, 40, 50] * bs,
        "T_0m": [10, 20, 30, 40, 50] * bs,
    })
    y = pd.DataFrame({"Cn2_15m": [1.58e-16, 1.58e-16, 1.58e-16, 1.58e-16, 1.58e-16] * bs})

    model = CNNModel(
        name="cnn_model",
        target_name="Cn2_15m",
        num_features=len(X.columns),  # single row
        window_size=0,  # single row
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        stride=1,
        padding="valid",
        bias=False,
        nonlinearity=True,
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
    model.train(X[0:4*bs], y[0:4*bs])
    # predict
    predictions = model.predict(X[4*bs:])
    # check the predictions
    assert (isinstance(predictions, pd.Series) or isinstance(predictions, np.ndarray))
    assert len(predictions) == len(y[4*bs:])
    assert np.allclose(predictions, y[4*bs:].values.ravel())


@pytest.mark.slow
def test_cnn_model_kernel_size_1():
    """Test the CNNModel."""
    # create a DataFrame with sample columns
    bs = 32
    X = pd.DataFrame({
        "T_10m": [10, 20, 30, 40, 50] * bs,
        "T_0m": [10, 20, 30, 40, 50] * bs,
    })
    y = pd.DataFrame({"Cn2_15m": [1.58e-16, 1.58e-16, 1.58e-16, 1.58e-16, 1.58e-16] * bs})

    model = CNNModel(
        name="cnn_model",
        target_name="Cn2_15m",
        num_features=len(X.columns),  # single row
        window_size=0,  # single row
        in_channels=1,
        out_channels=1,
        kernel_size=1,
        stride=1,
        padding="valid",
        bias=False,
        nonlinearity=True,
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
    model.train(X[0:4*bs], y[0:4*bs])
    # predict
    predictions = model.predict(X[4*bs:])
    # check the predictions
    assert (isinstance(predictions, pd.Series) or isinstance(predictions, np.ndarray))
    assert len(predictions) == len(y[4*bs:])
    assert np.allclose(predictions, y[4*bs:].values.ravel())


def test_cnn_model_fail_assertion():
    """Test the CNNModel."""
    # create a DataFrame with sample columns
    bs = 1
    X = pd.DataFrame({
        "T_10m": [10, 20, 30, 40, 50] * bs,
        "T_0m": [10, 20, 30, 40, 50] * bs,
    })

    with pytest.raises(AssertionError):
        _ = CNNModel(
            name="cnn_model",
            target_name="Cn2_15m",
            num_features=len(X.columns),  # single row
            window_size=0,  # single row
            in_channels=1,
            out_channels=1,
            kernel_size=99,
            stride=1,
            padding="valid",
            bias=False,
            nonlinearity=True,
            batch_size=bs,
            n_epochs=50,
            criterion = nn.MSELoss(),
            learning_rate=0.025,
            optimizer = optim.SGD,
            verbose=True,
        )

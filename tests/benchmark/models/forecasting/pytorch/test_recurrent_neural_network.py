import pytest

import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim

from otb.benchmark.models.forecasting.pytorch.recurrent_neural_network import RNNModel


@pytest.mark.slow
def test_rnn_model(task_api):
    """Test the RNNModel."""
    # create a DataFrame with sample columns
    bs = 32
    # create a DataFrame with the required columns
    X = pd.DataFrame({
        "T_10m": [10 for _ in range(50 * bs)],
        "T_0m": [10 for _ in range(50 * bs)],
    })
    y = pd.DataFrame({"Cn2_15m": [1.58e-16 for _ in range(50 * bs)]})

    X_train, X_test = X[0:40*bs], X[40*bs:]
    y_train, y_test = y[0:40*bs], y[40*bs:]

    # we need a task to prepare forecasting data
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
        criterion = nn.MSELoss(),
        learning_rate=0.025,
        optimizer = optim.SGD,
        verbose=True,
    )

    # check the model name
    assert model.name == "rnn_model"

    # train the model
    model.train(X_train, y_train)
    # predict
    predictions = model.predict(X_test)
    # check the predictions
    assert (isinstance(predictions, pd.Series) or isinstance(predictions, np.ndarray))
    assert len(predictions) == len(y_test)
    assert np.allclose(predictions, y_test.values.ravel())

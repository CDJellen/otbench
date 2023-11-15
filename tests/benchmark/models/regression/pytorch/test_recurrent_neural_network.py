import pytest

import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim

from otb.benchmark.models.regression.pytorch.recurrent_neural_network import RNNModel


@pytest.mark.slow
def test_rnn_model():
    """Test the RNNModel."""
    # create a DataFrame with sample columns
    bs = 32
    X = pd.DataFrame({
        "T_10m": [10, 20, 30, 40, 50] * bs,
        "T_0m": [10, 20, 30, 40, 50] * bs,
    })
    y = pd.DataFrame({"Cn2_15m": [1.58e-16, 1.58e-16, 1.58e-16, 1.58e-16, 1.58e-16] * bs})

    model = RNNModel(
        name="rnn_model",
        target_name="Cn2_15m",
        input_size=len(X.columns),  # single row
        window_size=1,  # single row
        hidden_size=32,
        num_layers=1,
        num_classes=1,
        batch_size=bs,
        n_epochs=50,
        criterion = nn.MSELoss(),
        learning_rate=0.025,
        optimizer = optim.SGD,
        verbose=False,
    )

    # check the model name
    assert model.name == "rnn_model"

    # train the model
    model.train(X[0:4*bs], y[0:4*bs])
    # predict
    predictions = model.predict(X[4*bs:])
    # check the predictions
    assert (isinstance(predictions, pd.Series) or isinstance(predictions, np.ndarray))
    assert len(predictions) == len(y[4*bs:])
    assert np.allclose(predictions, y[4*bs:].values.ravel())

import pandas as pd
import numpy as np

from otbench.benchmark.models.regression.air_water_temperature_difference import AWTModel


def test_awt_model():
    """Test the AWTModel."""
    # create a DataFrame with the required columns
    X = pd.DataFrame({
        "T_10m": [10, 20, 30, 40, 50],
        "T_0m": [10, 20, 30, 40, 50],
    })
    y = pd.DataFrame({"Cn2_15m": [1.58e-16, 1.58e-16, 1.58e-16, 1.58e-16, 1.58e-16]})

    # create the model
    model = AWTModel(name="AWT",
                     target_name="Cn2_15m",
                     air_temperature_col_name="T_10m",
                     water_temperature_col_name="T_0m",
                     use_log10=False)
    # check the model name
    assert model.name == "AWT"
    # check the model columns
    assert model.air_temperature_col_name == "T_10m"
    assert model.water_temperature_col_name == "T_0m"
    # check the model parameters
    assert model.use_log10 == False

    # train the model
    model.train(X[0:2], y[0:2])
    # predict
    predictions = model.predict(X[2:])
    # check the predictions
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(y[2:])
    assert np.allclose(predictions, y[2:].values.ravel())

    # create the model with log10
    model = AWTModel(name="AWT",
                     target_name="Cn2_15m",
                     air_temperature_col_name="T_10m",
                     water_temperature_col_name="T_0m",
                     use_log10=True)
    # check the model parameters
    assert model.use_log10 == True

    # train the model
    model.train(X[0:2], y[0:2])
    # predict
    predictions = model.predict(X[2:])
    # check the predictions
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(y[2:])
    assert np.allclose(predictions, np.log10(y[2:].values.ravel()))

import pytest

import pandas as pd
import numpy as np

from otbench.benchmark.models.regression.macro_meteorological import MacroMeteorologicalModel


@pytest.mark.slow
def test_macro_meteorological_model():
    """Test the MacroMeteorologicalModel."""
    # create a DataFrame with the required columns
    X = pd.DataFrame({
        "T_2m": [30 for _ in range(10)],
        "Spd_10m": [5 for _ in range(10)],
        "RH_2m": [50 for _ in range(10)],
        "time": [i for i in range(10)],
        "temporal_hour": [0 for _ in range(10)],
        "temporal_hour_weight": [1 for _ in range(10)],
    })
    y = pd.DataFrame({"Cn2_15m": [3.9925e-14 for _ in range(10)]})

    # create the model
    model = MacroMeteorologicalModel(name="macro_meteorological",
                                     target_name="Cn2_15m",
                                     timezone="UTC",
                                     obs_lat=0.0,
                                     obs_lon=0.0,
                                     air_temperature_col_name="T_2m",
                                     wind_speed_col_name="Spd_10m",
                                     humidity_col_name="RH_2m",
                                     time_col_name="time",
                                     temporal_hour_col_name="temporal_hour",
                                     temporal_hour_weight_col_name="temporal_hour_weight",
                                     height_of_observation=15.0,
                                     use_log10=False)
    # check the model name
    assert model.name == "macro_meteorological"
    # check the model columns
    assert model.air_temperature_col_name == "T_2m"
    assert model.wind_speed_col_name == "Spd_10m"
    assert model.humidity_col_name == "RH_2m"
    assert model.time_col_name == "time"
    assert model.temporal_hour_col_name == "temporal_hour"
    assert model.temporal_hour_weight_col_name == "temporal_hour_weight"

    # check the model parameters
    assert model.use_log10 == False

    # train the model (this model doesn't actually train)
    model.train(X, y)
    # predict
    predictions = model.predict(X)
    # check the predictions
    assert len(predictions) == len(y)
    assert np.allclose(predictions, y.values.ravel())

    # create the model with log10
    model = MacroMeteorologicalModel(name="macro_meteorological",
                                     target_name="Cn2_15m",
                                     timezone="UTC",
                                     obs_lat=0.0,
                                     obs_lon=0.0,
                                     air_temperature_col_name="T_2m",
                                     wind_speed_col_name="Spd_10m",
                                     humidity_col_name="RH_2m",
                                     time_col_name="time",
                                     temporal_hour_col_name="temporal_hour",
                                     temporal_hour_weight_col_name="temporal_hour_weight",
                                     height_of_observation=15.0,
                                     use_log10=True)
    # check the model parameters
    assert model.use_log10 == True

    # train the model
    model.train(X, y)
    # predict
    predictions = model.predict(X)
    # check the predictions
    assert len(predictions) == len(y)
    assert np.allclose(predictions, np.log10(y.values.ravel()))

import pytest
import pandas as pd
import numpy as np

from otb.benchmark.models.forecasting.climatology import ClimatologyForecastingModel


@pytest.mark.slow
def test_climatology_forecasting_model(task_api):
    """Test the LinearForecastingModel."""
    # create a DataFrame with the required columns
    periods = 50
    index = pd.date_range("2020-01-01", periods=periods, freq="1H")
    X = pd.DataFrame({
        "T_10m": [10 for _ in range(periods)],
        "T_0m": [10 for _ in range(periods)],
    }, index=index)
    y = pd.DataFrame({"Cn2_15m": [1.58e-16 for _ in range(periods)]}, index=index)

    # we need a task to prepare forecasting data
    task = task_api.get_task("forecasting.mlo_cn2.dropna.Cn2_15m")
    X, y = task.prepare_forecasting_data(X, y)

    # create the model
    model = ClimatologyForecastingModel(
        name="climatology_forecasting",
        target_name="Cn2_15m",
        window_size=task.window_size,
        forecast_horizon=task.forecast_horizon,
    )
    # check the model name
    assert model.name == "climatology_forecasting"

    # train the model (this model doesn't actually train)
    model.train(X, y)
    # predict
    predictions = model.predict(X)
    # check the predictions
    assert len(predictions) == len(y)
    assert np.allclose(predictions, y["Cn2_15m"].values)

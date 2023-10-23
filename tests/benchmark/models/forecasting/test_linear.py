import pytest
import pandas as pd
import numpy as np

from otb.benchmark.models.forecasting.linear import LinearForecastingModel


@pytest.mark.slow
def test_linear_forecasting_model(task_api):
    """Test the LinearForecastingModel."""
    # create a DataFrame with the required columns
    X = pd.DataFrame({
        "T_10m": [10 for _ in range(50)],
        "T_0m": [10 for _ in range(50)],
    })
    y = pd.DataFrame({"Cn2_15m": [1.58e-16 for _ in range(50)]})

    # we need a task to prepare forecasting data
    task = task_api.get_task("forecasting.mlo_cn2.dropna.Cn2_15m")
    X, y = task.prepare_forecasting_data(X, y)

    # create the model
    model = LinearForecastingModel(
        name="linear_forecasting",
        target_name="Cn2_15m",
        window_size=task.window_size,
        forecast_horizon=task.forecast_horizon,
    )
    # check the model name
    assert model.name == "linear_forecasting"

    # train the model (this model doesn't actually train)
    model.train(X, y)
    # predict
    predictions = model.predict(X)
    # check the predictions
    assert len(predictions) == len(y)
    assert np.allclose(predictions, y.values.ravel())

import pytest
import pandas as pd
import numpy as np

from otb.benchmark.models.forecasting.base_model import BaseForecastingModel


def test_base_model():
    """Test the BaseModel."""
    # create the model
    model = BaseForecastingModel(
        name="base",
        target_name="Cn2_15m",
        window_size=4,
        forecast_horizon=1,
        )
    # check the model name
    assert model.name == "base"
    assert model.target_name == "Cn2_15m"
    assert model.window_size == 4
    assert model.forecast_horizon == 1

    with pytest.raises(NotImplementedError):
        model.train(None, None)
    with pytest.raises(NotImplementedError):
        model.predict(None)

import pytest
import pandas as pd
import numpy as np

from otbench.benchmark.models.regression.base_model import BaseRegressionModel


def test_base_model():
    """Test the BaseModel."""
    # create the model
    model = BaseRegressionModel(
        name="base",
        target_name="Cn2_15m",
    )
    # check the model name
    assert model.name == "base"
    assert model.target_name == "Cn2_15m"

    with pytest.raises(NotImplementedError):
        model.train(None, None)
    with pytest.raises(NotImplementedError):
        model.predict(None)

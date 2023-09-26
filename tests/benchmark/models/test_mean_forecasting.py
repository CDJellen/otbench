import json

import pytest
import pandas as pd
import numpy as np

from otb.benchmark.models.mean_forecasting import MeanWindowForecastingModel
from tests import TESTS_BENCHMARK_FP


@pytest.mark.slow
def test_mean_window_forecasting_model(task_api):
    """Test the MeanWindowForecastingModel."""
    # create a DataFrame with the required columns
    X = pd.DataFrame({
        "T_10m": [10 for _ in range(50)],
        "T_0m": [10 for _ in range(50)],
    })
    y = pd.DataFrame({
        "Cn2_15m": [1.58e-16 for _ in range(50)]
    })
    
    # we need a task to prepare forecasting data
    task = task_api.get_task("forecasting.mlo_cn2.dropna.Cn2_15m", benchmark_fp=TESTS_BENCHMARK_FP)
    X, y = task.prepare_forecasting_data(X, y)

    # create the model
    model = MeanWindowForecastingModel(
        name="mean_window_forecasting",
        target_name="Cn2_15m",
        )
    # check the model name
    assert model.name == "mean_window_forecasting"

    # train the model (this model doesn't actually train)
    model.train(X, y)
    # predict
    predictions = model.predict(X)
    # check the predictions
    assert len(predictions) == len(y)
    assert np.allclose(predictions, y.values.ravel())

@pytest.mark.slow
def test_with_forecasting_evaluation(task_api):
    """Test the MeanWindowForecastingModel with model evaluation."""    
    model = MeanWindowForecastingModel(
        name="mean_window_forecasting",
        target_name="Cn2_15m",
        )
    # save current experiments.json contents
    with open(TESTS_BENCHMARK_FP, "r") as f:
        old_experiments = json.load(f)

    task = task_api.get_task("forecasting.mlo_cn2.dropna.Cn2_15m", benchmark_fp=TESTS_BENCHMARK_FP)    
    # test model evaluation
    _ = task.evaluate_model(model.predict, return_predictions=True)
    # test model evaluation with as a benchmark
    _ = task.evaluate_model(model.predict, return_predictions=True, include_as_benchmark=True, model_name="test", overwrite=False)
    _ = task.evaluate_model(model.predict, return_predictions=True, include_as_benchmark=True, model_name="test", overwrite=True)
    with pytest.raises(ValueError):
        _ = task.evaluate_model(model.predict, return_predictions=True, include_as_benchmark=True, model_name=None, overwrite=False)
    
    # restore experiments.json contents
    with open(TESTS_BENCHMARK_FP, "w") as f:
        json.dump(old_experiments, f, indent=4)

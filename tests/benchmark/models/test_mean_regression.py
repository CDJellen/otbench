import json

import pytest
import pandas as pd
import numpy as np

from otb.benchmark.models.mean_regression import MeanRegressionModel
from tests import TESTS_BENCHMARK_FP


def test_mean_regression_model():
    """Test the MeanRegressionModel."""
    # create a DataFrame
    X = pd.DataFrame({
        "T_10m": [10, 20, 30, 40, 50],
        "T_0m": [10, 20, 30, 40, 50],
    })
    y = pd.DataFrame({
        "Cn2_15m": [1.58e-16, 1.58e-16, 1.58e-16, 1.58e-16, 1.58e-16]
    })
    
    # create the model
    model = MeanRegressionModel(
        name="mean_regression",
        )
    # check the model name
    assert model.name == "mean_regression"

    # train the model
    model.train(X[0:2], y[0:2])
    # predict
    predictions = model.predict(X[2:])
    # check the predictions
    assert len(predictions) == len(y[2:])
    assert np.allclose(predictions, y[2:].values.ravel())


@pytest.mark.slow
def test_with_regression_evaluation(task_api):
    """Test the MeanWindowForecastingModel with model evaluation."""    
    model = MeanRegressionModel(
        name="mean_window_forecasting",
        target_name="Cn2_15m",
        )
    # save current experiments.json contents
    with open(TESTS_BENCHMARK_FP, "r") as f:
        old_experiments = json.load(f)

    task = task_api.get_task("regression.mlo_cn2.dropna.Cn2_15m", benchmark_fp=TESTS_BENCHMARK_FP)    
    # get train data
    X_train, y_train = task.get_train_data()
    # train the model
    model.train(X_train, y_train)    
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

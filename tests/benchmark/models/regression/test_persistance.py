import os
import json

import pytest
import pandas as pd
import numpy as np

from otbench.benchmark.models.regression.persistence import PersistenceRegressionModel
from tests import TESTS_BENCHMARK_FP


def test_persistence_regression_model():
    """Test the PersistenceRegressionModel."""
    # create a DataFrame
    X = pd.DataFrame({
        "T_10m": [10, 20, 30, 40, 50],
        "T_0m": [10, 20, 30, 40, 50],
    })
    y = pd.DataFrame({"Cn2_15m": [1.58e-16, 1.58e-16, 1.58e-16, 1.58e-16, 1.58e-16]})

    # create the model
    model = PersistenceRegressionModel(
        name="persistence_regression",
        target_name="Cn2_15m",
    )
    # check the model name
    assert model.name == "persistence_regression"

    # train the model
    model.train(X[0:2], y[0:2])
    # predict
    predictions = model.predict(X[2:])
    # check the predictions
    assert len(predictions) == len(y[2:])
    assert np.allclose(predictions, y[2:].values.ravel())


@pytest.mark.slow
def test_with_regression_evaluation(task_api):
    """Test the PersistenceForecastingModel with model evaluation."""
    model = PersistenceRegressionModel(
        name="persistence_regression",
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
    # test model evaluation with transforms
    _ = task.evaluate_model(model.predict,
                            return_predictions=True,
                            x_transforms=lambda x: x,
                            x_transform_kwargs={},
                            predict_call_kwargs={})
    # test model evaluation with as a benchmark
    _ = task.evaluate_model(model.predict,
                            return_predictions=True,
                            include_as_benchmark=True,
                            model_name="test",
                            overwrite=False)
    _ = task.evaluate_model(model.predict,
                            return_predictions=True,
                            include_as_benchmark=True,
                            model_name="test",
                            overwrite=True)
    with pytest.raises(ValueError):
        _ = task.evaluate_model(model.predict,
                                return_predictions=True,
                                include_as_benchmark=True,
                                model_name=None,
                                overwrite=False)
    # see where it shows up in the benchmark
    top_models = task.top_models(n=100, metric="")
    assert isinstance(top_models, dict)
    # see where it shows up in the r2_score metrics
    top_models = task.top_models(n=100, metric="r2_score")
    assert isinstance(top_models, dict)
    # remove the experiments.json file and then try to get the top models
    os.remove(TESTS_BENCHMARK_FP)
    with pytest.raises(FileNotFoundError):
        top_models = task.top_models(n=100, metric="r2_score")
    # restore experiments.json contents
    with open(TESTS_BENCHMARK_FP, "w") as f:
        json.dump(old_experiments, f, indent=4)

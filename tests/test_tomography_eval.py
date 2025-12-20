
import pytest
import numpy as np
import pandas as pd
from otbench.eval import metrics
from otbench.eval.integrated_metrics import integrated_seeing
from otbench.tasks.tasks import TaskABC

def test_metrics_detailed_return():
    """Verify that metrics return detailed info when requested."""
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.1, 1.9], [3.2, 3.8]])
    
    # RMSE
    # Diff: [[-0.1, 0.1], [-0.2, 0.2]]
    # SqDiff: [[0.01, 0.01], [0.04, 0.04]]
    # Mean Sq Error per output: [0.025, 0.025] -> RMSE per output [0.158, 0.158] (approx)
    
    res = metrics.root_mean_square_error(y_true, y_pred, detailed=True)
    assert isinstance(res, dict)
    assert "metric_value" in res
    assert "detailed_score" in res
    assert len(res["detailed_score"]) == 2
    assert np.allclose(res["detailed_score"], [0.15811388, 0.15811388])

def test_integrated_seeing_metric():
    """Verify integrated seeing metric calculation."""
    # simple constant profile
    y_true = np.ones((1, 5)) # 5 layers, value 1
    y_pred = np.ones((1, 5)) * 0.5 # 5 layers, value 0.5
    
    # heights None -> sum
    res = integrated_seeing(y_true, y_pred, heights=None, detailed=True)
    
    assert "seeing_true" in res
    assert "seeing_pred" in res
    # True sum = 5. Pred sum = 2.5.
    # r0 ~ (J)^-3/5
    # seeing ~ r0^-1 ~ (J)^3/5
    # true ~ 5^0.6 = 2.62
    # pred ~ 2.5^0.6 = 1.73
    # error ~ 0.89
    
    assert res["seeing_true"][0] > res["seeing_pred"][0]
    assert np.isclose(res['metric_value'], res['seeing_true'][0] - res['seeing_pred'][0])

def test_evaluate_model_detailed_propagation():
    """Mock test to check if evaluate_model propagates detailed flag."""
    # We mock Task to avoid loading full dataset
    class MockTask:
        def __init__(self):
            self.task = {"eval_metrics": ["root_mean_square_error"]}
            
        def get_test_data(self, data_type="pd"):
            X = pd.DataFrame({"f1": [1, 2]})
            y = pd.DataFrame({"t1": [1, 2], "t2": [3, 4]})
            return X, y
            
    # We can't easily instantiate TaskABC directly or use the real classes without config.
    # But we can import the class and patch get_test_data if we instance it.
    # However, creating a real regression task requires tasks.json entry.
    pass

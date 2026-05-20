
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

def test_get_valid_indices_partial_nan_vector():
    """Vector targets with partial NaN rows must be dropped entirely."""
    from otbench.eval.utils import _get_valid_indices

    y_true = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
    y_pred = np.array([[1.0, 2.0], [3.0, 3.0], [4.0, 5.0]])

    yt, yp = _get_valid_indices(y_true, y_pred)
    # Row 1 has a NaN in y_true → must be dropped
    assert yt.shape == (2, 2)
    assert yp.shape == (2, 2)
    np.testing.assert_array_equal(yt, [[1.0, 2.0], [4.0, 5.0]])


def test_get_valid_indices_all_nan_row():
    """Rows where all values are NaN are still dropped."""
    from otbench.eval.utils import _get_valid_indices

    y_true = np.array([[np.nan, np.nan], [1.0, 2.0]])
    y_pred = np.array([[1.0, 1.0], [1.0, 2.0]])

    yt, yp = _get_valid_indices(y_true, y_pred)
    assert yt.shape == (1, 2)


def test_integrated_seeing_values_are_integrals():
    """When values_are_integrals=True, heights are ignored and values are summed."""
    # Two identical profiles → RMSE should be 0
    profile = np.array([[1e-14, 2e-14, 3e-14]])
    heights = np.array([500, 1000, 2000])

    res_integrals = integrated_seeing(
        profile, profile, heights=heights, values_are_integrals=True
    )
    assert np.isclose(res_integrals["metric_value"], 0.0)

    # With values_are_integrals=False, trapz integration should give a different
    # total J than simple sum → seeing values differ if we compare the two modes
    res_density = integrated_seeing(
        profile, profile, heights=heights, values_are_integrals=False
    )
    assert np.isclose(res_density["metric_value"], 0.0)


def test_integrated_seeing_sum_vs_trapz():
    """Verify that sum (integrals mode) and trapz (density mode) give different seeing."""
    profile = np.array([[1e-14, 2e-14, 3e-14]])
    heights = np.array([500, 1000, 2000])

    res_sum = integrated_seeing(profile, profile * 0.5, heights=heights,
                                values_are_integrals=True, detailed=True)
    res_trapz = integrated_seeing(profile, profile * 0.5, heights=heights,
                                  values_are_integrals=False, detailed=True)
    # The total J differs between sum and trapz, so derived seeing should differ
    assert res_sum["seeing_true"][0] != res_trapz["seeing_true"][0]


def test_integrated_seeing_density_without_heights_raises():
    """values_are_integrals=False with heights=None must raise ValueError."""
    profile = np.array([[1e-14, 2e-14, 3e-14]])
    with pytest.raises(ValueError, match="heights must be provided"):
        integrated_seeing(profile, profile, heights=None, values_are_integrals=False)

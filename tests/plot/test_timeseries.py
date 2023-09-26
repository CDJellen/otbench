import pytest
import pandas as pd

from otb.plot.timeseries import plot_predictions


@pytest.mark.slow
@pytest.mark.private  # TODO: remove this tag when the test is ready
def test_plot_predictions():
    """Test plotting predictions."""
    y_true = pd.Series([1e-16 for _ in range(10)])
    y_pred = pd.Series([1.1e-16 for _ in range(10)])
    y_pred_mdl_1 = pd.Series([0.9e-16 for _ in range(10)])
    assert plot_predictions(y_true, y_pred, y_pred_mdl_1) == None

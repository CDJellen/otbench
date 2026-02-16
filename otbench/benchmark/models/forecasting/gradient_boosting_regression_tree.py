# otbench/benchmark/models/forecasting/gradient_boosting_regression_tree.py
from typing import Union
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from otbench.benchmark.models.forecasting.base_model import BaseForecastingModel

class GradientBoostingForecastingModel(BaseForecastingModel):
    """A model that uses gradient boosting regression trees for direct forecasting."""

    def __init__(self, name: str, target_name: str, window_size: int, forecast_horizon: int, output_size: int = 1, **kwargs):
        super().__init__(name, target_name, window_size, forecast_horizon, **kwargs)
        if "verbose" in kwargs:
            del kwargs["verbose"]
        
        # Base estimator
        lgb_estimator = lgb.LGBMRegressor(verbose=-1, **kwargs)
        
        # Handle Vector vs Scalar
        if output_size > 1:
            self._lgb_mdl = MultiOutputRegressor(lgb_estimator)
        else:
            self._lgb_mdl = lgb_estimator

    def _train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Internal training implementation."""
        # Guard against empty training data
        if len(X) == 0:
            return

        self._lgb_mdl.fit(X, y)

    def _predict(self, X: 'pd.DataFrame'):
        """Internal prediction implementation."""
        # Guard against empty prediction data
        if len(X) == 0:
            # Return empty array with correct dimensions
            if isinstance(self._lgb_mdl, MultiOutputRegressor):
                # We can't ask the model for output dim easily if not fitted, 
                # so we rely on the target_name list length if available or fallback
                n_targets = len(self.target_name) if isinstance(self.target_name, list) else 1
                return np.empty((0, n_targets))
            return np.array([])

        return self._lgb_mdl.predict(X)
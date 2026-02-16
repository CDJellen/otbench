from typing import Union

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor

from otbench.benchmark.models.regression.base_model import BaseRegressionModel


class GradientBoostingRegressionModel(BaseRegressionModel):
    """A model that uses gradient boosting regression trees to predict the target."""

    def __init__(self, name: str, target_name: str, time_col_name: Union[str, None] = None, output_size: int = 1, **kwargs):
        super().__init__(name, target_name, **kwargs)
        self.time_col_name = time_col_name if time_col_name is not None else None
        
        # Remove verbose from kwargs before passing to LGBM (it handles it differently)
        if "verbose" in kwargs:
            del kwargs["verbose"]

        # Base estimator
        lgb_estimator = lgb.LGBMRegressor(verbose=-1, **kwargs)

        # Handle Vector vs Scalar
        if output_size > 1:
            self._lgb_mdl = MultiOutputRegressor(lgb_estimator)
        else:
            self._lgb_mdl = lgb_estimator

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Train the underlying LightGBM regressor."""
        if len(X) == 0:
            return

        self._lgb_mdl.fit(X, y)

    def predict(self, X: 'pd.DataFrame'):
        """Use the underlying LightGBM regressor to generate predictions."""
        if len(X) == 0:
            return np.empty((0, len(self.target_name)))

        return self._lgb_mdl.predict(X)

from typing import Union

import pandas as pd
import numpy as np
import lightgbm as lgb

from otbench.benchmark.models.regression.base_model import BaseRegressionModel


class GradientBoostingRegressionModel(BaseRegressionModel):
    """A model that uses gradient boosting regression trees to predict the target."""

    def __init__(self, name: str, target_name: str, time_col_name: Union[str, None] = None, **kwargs):
        super().__init__(name, target_name, **kwargs)
        self.time_col_name = time_col_name if time_col_name is not None else None
        del kwargs["verbose"]
        self._lgb_mdl = lgb.LGBMRegressor(verbose=-1, **kwargs)

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Train the underlying LightGBM regressor."""
        self._lgb_mdl.fit(X, y)

    def predict(self, X: 'pd.DataFrame'):
        """Use the underlying LightGBM regressor to generate predictions."""
        return self._lgb_mdl.predict(X)

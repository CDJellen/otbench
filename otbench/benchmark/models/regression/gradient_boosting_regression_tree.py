from typing import Union

import numpy as np
import pandas as pd

from otbench.benchmark.models._lgbm_utils import (
    _NON_LGBM_KEYS,
    build_lgbm_estimator,
    filter_valid_target_rows,
)
from otbench.benchmark.models.regression.base_model import BaseRegressionModel


class GradientBoostingRegressionModel(BaseRegressionModel):
    """A model that uses gradient boosting regression trees to predict the target."""

    _NON_LGBM_KEYS = _NON_LGBM_KEYS

    def __init__(self,
                 name: str,
                 target_name: str,
                 time_col_name: Union[str, None] = None,
                 output_size: int = 1,
                 **kwargs):
        super().__init__(name, target_name, **kwargs)
        self.time_col_name = time_col_name
        self._lgb_mdl, self.output_size = build_lgbm_estimator(output_size, target_name, kwargs)

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Train the underlying LightGBM regressor."""
        if len(X) == 0:
            return
        X, y = filter_valid_target_rows(X, y)
        if len(X) == 0:
            return
        self._lgb_mdl.fit(X, y)

    def predict(self, X: 'pd.DataFrame'):
        """Use the underlying LightGBM regressor to generate predictions."""
        if len(X) == 0:
            return np.empty((0, self.output_size)) if self.output_size > 1 else np.empty((0,))
        return self._lgb_mdl.predict(X)

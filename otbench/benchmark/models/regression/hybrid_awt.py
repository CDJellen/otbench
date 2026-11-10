from typing import Union

import lightgbm as lgb
import numpy as np
import pandas as pd

from otbench.benchmark.models._lgbm_utils import _NON_LGBM_KEYS, filter_lgbm_kwargs
from otbench.benchmark.models.regression.air_water_temperature_difference import AWTModel
from otbench.benchmark.models.regression.base_model import BaseRegressionModel


class HybridAWTRegressionModel(BaseRegressionModel):
    """A model that uses a hybrid of the AWTModel and a gradient boosted decision tree to predict the target."""

    _NON_LGBM_KEYS = _NON_LGBM_KEYS

    def __init__(self, name: str, target_name: str, time_col_name: Union[str, None] = None, **kwargs):
        super().__init__(name, target_name, **kwargs)
        self.time_col_name = time_col_name
        self._awt_mdl = AWTModel(name=name, target_name=target_name, **kwargs)
        self._lgb_mdl = lgb.LGBMRegressor(verbose=-1, **filter_lgbm_kwargs(kwargs))

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Train the underlying LightGBM regressor to form a hybrid model."""
        y_pred_awt = self._awt_mdl.predict(X)
        target = y[self.target_name].values - y_pred_awt.values
        target[np.isnan(y_pred_awt.values)] = np.nan
        valid = ~np.isnan(target)
        if not valid.any():
            return
        self._lgb_mdl.fit(X[valid], target[valid])

    def predict(self, X: 'pd.DataFrame'):
        """Generate AWT predictions and correct with the trained LightGBM regressor."""
        y_pred_awt = self._awt_mdl.predict(X)
        hybrid_correction = self._lgb_mdl.predict(X)
        preds = y_pred_awt.values + hybrid_correction
        preds[np.isnan(y_pred_awt.values)] = np.nan
        return preds

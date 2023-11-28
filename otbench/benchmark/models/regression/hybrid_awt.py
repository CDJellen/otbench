from typing import Union

import pandas as pd
import numpy as np
import lightgbm as lgb

from otbench.benchmark.models.regression.base_model import BaseRegressionModel
from otbench.benchmark.models.regression.air_water_temperature_difference import AWTModel


class HybridAWTRegressionModel(BaseRegressionModel):
    """A model that uses a hybrid of the AWTModel and a random forest regressor to predict the target."""

    def __init__(self, name: str, target_name: str, time_col_name: Union[str, None] = None, **kwargs):
        super().__init__(name, target_name, **kwargs)
        self.time_col_name = time_col_name if time_col_name is not None else None
        del kwargs["verbose"]
        self._lgb_mdl = lgb.LGBMRegressor(verbose=-1, **kwargs)
        self._awt_mdl = AWTModel(name=name, target_name=target_name, **kwargs)

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Train the underlying LightGBM regressor to form a hybrid model."""
        y_pred_awt = self._awt_mdl.predict(X)
        target = y[self.target_name].values - y_pred_awt.values
        target[np.isnan(y_pred_awt)] = np.nan
        self._lgb_mdl.fit(X, target)

    def predict(self, X: 'pd.DataFrame'):
        """Use generate AWT predictions and correct with trained LightGBM regressor."""
        y_pred_awt = self._awt_mdl.predict(X)
        hybrid_correction = self._lgb_mdl.predict(X)
        preds = y_pred_awt.values + hybrid_correction
        preds[np.isnan(y_pred_awt)] = np.nan
        return preds

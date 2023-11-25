"""
Air-Water Temperature Difference Model from:

@article{chen2019climatological,
  title={Climatological analysis of the seeing at Fuxian Solar Observatory},
  author={Chen, Li-Hui and Liu, Zhong and Chen, Dong},
  journal={Research in Astronomy and Astrophysics},
  volume={19},
  number={1},
  pages={015},
  year={2019},
  publisher={IOP Publishing}
}
"""
import numpy as np
import pandas as pd

from otbench.benchmark.models.regression.base_model import BaseRegressionModel


class AWTModel(BaseRegressionModel):

    def __init__(self,
                 name: str,
                 target_name: str,
                 air_temperature_col_name: str,
                 water_temperature_col_name: str,
                 use_log10: bool = True,
                 **kwargs):
        super().__init__(name, target_name, **kwargs)
        self.air_temperature_col_name = air_temperature_col_name
        self.water_temperature_col_name = water_temperature_col_name
        self.use_log10 = use_log10

    def train(self, X: 'pd.DataFrame', y: 'pd.DataFrame'):
        """Maintain the same interface as the other models."""
        pass

    def predict(self, X):
        """Predict the log10 of the cn2 value using the air-water temperature difference."""
        X_ = X.loc[X[self.air_temperature_col_name].notna() & X[self.water_temperature_col_name].notna()].copy()

        awt = X_[self.air_temperature_col_name] - X_[self.water_temperature_col_name]
        awt_prediction = pd.Series((2.05 * awt**2 + 2.37 * awt + 1.58) * 1e-16, index=X.index, name=self.name)

        if self.use_log10:
            return np.log10(awt_prediction)
        return awt_prediction

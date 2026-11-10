r"""
Offshore Macro-Meteorological Model from:

@article{wang2015prediction,
  title={Prediction model of atmospheric refractive index structure parameter in coastal area},
  author={Wang, Hongxing and Li, Bifeng and Wu, Xiaojun and Liu, Chuanhui and Hu, Zhihui and Xu, Pengfei},
  journal={Journal of Modern Optics},
  volume={62},
  number={16},
  pages={1336--1346},
  year={2015},
  publisher={Taylor \& Francis}
}
"""
import pandas as pd

from otbench.benchmark.models.regression.macro_meteorological import MacroMeteorologicalModel


class OffshoreMacroMeteorologicalModel(MacroMeteorologicalModel):
    """A model which predicts the Cn2 value using macro-meteorological parameters, tuned for offshore cases."""

    _REFERENCE_HEIGHT: float = 62.0
    _TEMP_KELVIN_OFFSET: float = 0.0
    _COEFFS = (-1.58e-15, 2.74e-16, 8.3e-17, -2.22e-18, 1.42e-20, 3.37e-16, 1.92e-16, -2.8e-17, -7.44e-14)

    def _out_of_range_mask(self, X: pd.DataFrame) -> pd.Series:
        """Identify rows outside the empirical validity envelope of the offshore polynomial."""
        return ((X[self.wind_speed_col_name] > 15) | ((X[self.air_temperature_col_name] < -2) |
                                                      (X[self.air_temperature_col_name] > 24)) |
                (X[self.humidity_col_name] < 15))

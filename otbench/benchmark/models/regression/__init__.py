from otbench.benchmark.models.regression.air_water_temperature_difference import AWTModel
from otbench.benchmark.models.regression.macro_meteorological import MacroMeteorologicalModel
from otbench.benchmark.models.regression.offshore_macro_meteorological import OffshoreMacroMeteorologicalModel
from otbench.benchmark.models.regression.persistence import PersistenceRegressionModel
from otbench.benchmark.models.regression.minute_climatology import MinuteClimatologyRegressionModel
from otbench.benchmark.models.regression.climatology import ClimatologyRegressionModel
from otbench.benchmark.models.regression.hybrid_awt import HybridAWTRegressionModel
from otbench.benchmark.models.regression.gradient_boosting_regression_tree import GradientBoostingRegressionModel

__all__ = [
    "AWTModel", "MacroMeteorologicalModel", "OffshoreMacroMeteorologicalModel", "PersistenceRegressionModel",
    "MinuteClimatologyRegressionModel", "ClimatologyRegressionModel", "HybridAWTRegressionModel",
    "GradientBoostingRegressionModel"
]

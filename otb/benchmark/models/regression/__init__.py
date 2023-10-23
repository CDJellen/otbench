from otb.benchmark.models.regression.air_water_temperature_difference import AWTModel
from otb.benchmark.models.regression.macro_meteorological import MacroMeteorologicalModel
from otb.benchmark.models.regression.offshore_macro_meteorological import OffshoreMacroMeteorologicalModel
from otb.benchmark.models.regression.persistence import PersistenceRegressionModel
from otb.benchmark.models.regression.minute_climatology import MinuteClimatologyRegressionModel
from otb.benchmark.models.regression.climatology import ClimatologyRegressionModel
from otb.benchmark.models.regression.hybrid_awt import HybridAWTRegressionModel
from otb.benchmark.models.regression.gradient_boosting_regression_tree import GradientBoostingRegressionModel

__all__ = [
    "AWTModel", "MacroMeteorologicalModel", "OffshoreMacroMeteorologicalModel", "PersistenceRegressionModel",
    "MinuteClimatologyRegressionModel", "ClimatologyRegressionModel", "HybridAWTRegressionModel",
    "GradientBoostingRegressionModel"
]

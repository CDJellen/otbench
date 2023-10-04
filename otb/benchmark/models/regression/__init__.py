from otb.benchmark.models.regression.air_water_temperature_difference import AWTModel
from otb.benchmark.models.regression.macro_meterological import MacroMeterologicalModel
from otb.benchmark.models.regression.offshore_macro_meterological import OffshoreMacroMeterologicalModel
from otb.benchmark.models.regression.persistance import PersistanceRegressionModel
from otb.benchmark.models.regression.minute_climatology import MinuteClimatologyRegressionModel
from otb.benchmark.models.regression.climatology import ClimatologyRegressionModel
from otb.benchmark.models.regression.hybrid_awt import HybridAWTRegressionModel
from otb.benchmark.models.regression.random_forest import RandomForestRegressionModel


__all__ = [
    "AWTModel",  "MacroMeterologicalModel", "OffshoreMacroMeterologicalModel",
    "PersistanceRegressionModel", "MinuteClimatologyRegressionModel", "ClimatologyRegressionModel", "HybridAWTRegressionModel", "RandomForestRegressionModel"
]

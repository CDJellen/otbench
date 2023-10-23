from otb.benchmark.models.forecasting.linear import LinearForecastingModel
from otb.benchmark.models.forecasting.mean_window import MeanWindowForecastingModel
from otb.benchmark.models.forecasting.persistence import PersistenceForecastingModel
from otb.benchmark.models.forecasting.minute_climatology import MinuteClimatologyForecastingModel
from otb.benchmark.models.forecasting.climatology import ClimatologyForecastingModel
from otb.benchmark.models.forecasting.random_forest import RandomForestForecastingModel

__all__ = [
    "LinearForecastingModel", "MeanWindowForecastingModel", "PersistenceForecastingModel",
    "MinuteClimatologyForecastingModel", "ClimatologyForecastingModel"
]

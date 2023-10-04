from otb.benchmark.models.forecasting.linear import LinearForecastingModel
from otb.benchmark.models.forecasting.mean_window import MeanWindowForecastingModel
from otb.benchmark.models.forecasting.persistance import PersistanceForecastingModel
from otb.benchmark.models.forecasting.minute_climatology import MinuteClimatologyForecastingModel
from otb.benchmark.models.forecasting.climatology import ClimatologyForecastingModel
from otb.benchmark.models.forecasting.random_forest import RandomForestForecastingModel


__all__ = ["LinearForecastingModel", "MeanWindowForecastingModel", "PersistanceForecastingModel", "MinuteClimatologyForecastingModel", "ClimatologyForecastingModel"]

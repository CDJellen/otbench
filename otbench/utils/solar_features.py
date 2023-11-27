import warnings

import pandas as pd
from astral import LocationInfo
from astral.sun import sun

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def add_temporal_hour(X: pd.DataFrame, name, timezone, obs_lat, obs_lon, time_col_name, temporal_hour_col_name):
    """Get the difference between the current time and sunrise."""

    OBS = LocationInfo(name=name, region='', timezone=timezone, latitude=obs_lat, longitude=obs_lon)

    def _get_sunrise_sunset(dt):
        s = sun(OBS.observer, date=dt, tzinfo=OBS.timezone)
        return (s['sunrise'], s['sunset'])

    if isinstance(X.index, pd.DatetimeIndex):
        X['date'] = X.index.date
        if time_col_name not in X.columns:
            X[time_col_name] = X.index
    else:
        X['date'] = pd.to_datetime(X[time_col_name]).dt.date

    suntimes = X['date'].apply(lambda x: _get_sunrise_sunset(x))
    X['time_sunrise'] = suntimes.apply(lambda x: x[0])
    X['time_sunset'] = suntimes.apply(lambda x: x[1])

    X['time_sunrise'] = X['time_sunrise'].dt.tz_convert(timezone)
    X['time_sunrise'] = X['time_sunrise'].dt.tz_localize(tz=None)
    X['time_sunset'] = X['time_sunset'].dt.tz_convert(timezone)
    X['time_sunset'] = X['time_sunset'].dt.tz_localize(tz=None)

    X[temporal_hour_col_name] = ((X[time_col_name] - X['time_sunrise']).dt.total_seconds()) / (
        (1 / 12) * (X[time_col_name] - X['time_sunrise']).dt.total_seconds())

    X.drop(columns=['date', 'time_sunrise', 'time_sunset'], inplace=True)

    return X


def add_temporal_hour_weight(X: pd.DataFrame, temporal_hour_col_name: str, temporal_hour_weight_col_name: str):
    """Get the temporal hour weight from the temporal hour"""

    TEMPORAL_HOUR_DICT = {
        -99: 0.11,
        -4: 0.11,
        -3: 0.07,
        -2: 0.08,
        -1: 0.06,
        0: 0.05,
        1: 0.10,
        2: 0.51,
        3: 0.75,
        4: 0.95,
        5: 1,
        6: 0.9,
        7: 0.8,
        8: 0.59,
        9: 0.32,
        10: 0.22,
        11: 0.10,
        12: 0.08,
        13: 0.13,
        99: 0.13
    }

    X[temporal_hour_weight_col_name] = 0
    keys = list(TEMPORAL_HOUR_DICT.keys())

    for idx in range(len(keys) - 1):
        # read values (start, stop, W_th) from the dictionary
        start_temporal_hour = keys[idx]
        end_temporal_hour = keys[idx + 1]
        temporal_hour_weight_value = TEMPORAL_HOUR_DICT[keys[idx]]

        # assign DataFrame values based on temporal hour of observation and bin in dict
        X.loc[(X[temporal_hour_col_name] > start_temporal_hour) & (X[temporal_hour_col_name] < end_temporal_hour),
              temporal_hour_weight_col_name] = temporal_hour_weight_value

    return X

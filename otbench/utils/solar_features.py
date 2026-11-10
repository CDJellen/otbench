import pandas as pd
from astral import LocationInfo
from astral.sun import sun


def add_temporal_hour(X: pd.DataFrame, name, timezone, obs_lat, obs_lon, time_col_name, temporal_hour_col_name):
    """Get the difference between the current time and sunrise."""
    X = X.copy()
    OBS = LocationInfo(name=name, region='', timezone=timezone, latitude=obs_lat, longitude=obs_lon)

    if isinstance(X.index, pd.DatetimeIndex):
        dates = pd.Series(X.index.date, index=X.index)
        timestamps = pd.to_datetime(X[time_col_name]) if time_col_name in X.columns else pd.Series(X.index,
                                                                                                   index=X.index)
    else:
        timestamps = pd.to_datetime(X[time_col_name])
        dates = timestamps.dt.date

    if getattr(timestamps.dt, "tz", None) is not None:
        timestamps = timestamps.dt.tz_convert(timezone).dt.tz_localize(tz=None)

    unique_dates = pd.unique(dates)
    sunrise_map = {}
    sunset_map = {}
    for dt in unique_dates:
        if pd.isna(dt):
            sunrise_map[dt] = pd.NaT
            sunset_map[dt] = pd.NaT
        else:
            s = sun(OBS.observer, date=dt, tzinfo=OBS.timezone)
            sunrise_map[dt] = s['sunrise']
            sunset_map[dt] = s['sunset']

    time_sunrise = pd.to_datetime(dates.map(sunrise_map)).dt.tz_convert(timezone).dt.tz_localize(tz=None)
    time_sunset = pd.to_datetime(dates.map(sunset_map)).dt.tz_convert(timezone).dt.tz_localize(tz=None)

    X[temporal_hour_col_name] = ((timestamps - time_sunrise).dt.total_seconds()) / (
        (1 / 12) * (time_sunset - time_sunrise).dt.total_seconds())

    return X


def add_temporal_hour_weight(X: pd.DataFrame, temporal_hour_col_name: str, temporal_hour_weight_col_name: str):
    """Get the temporal hour weight from the temporal hour"""
    X = X.copy()
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

    X[temporal_hour_weight_col_name] = 0.0
    keys = list(TEMPORAL_HOUR_DICT.keys())

    for idx in range(len(keys) - 1):
        # read values (start, stop, W_th) from the dictionary
        start_temporal_hour = keys[idx]
        end_temporal_hour = keys[idx + 1]
        temporal_hour_weight_value = TEMPORAL_HOUR_DICT[keys[idx]]

        # assign DataFrame values based on temporal hour of observation and half-open bin [start, end)
        X.loc[(X[temporal_hour_col_name] >= start_temporal_hour) & (X[temporal_hour_col_name] < end_temporal_hour),
              temporal_hour_weight_col_name] = temporal_hour_weight_value

    X.loc[X[temporal_hour_col_name] < keys[0], temporal_hour_weight_col_name] = TEMPORAL_HOUR_DICT[keys[0]]
    X.loc[X[temporal_hour_col_name] >= keys[-1], temporal_hour_weight_col_name] = TEMPORAL_HOUR_DICT[keys[-1]]
    X.loc[X[temporal_hour_col_name].isna(), temporal_hour_weight_col_name] = float("nan")

    return X

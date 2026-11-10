import pytest
import pandas as pd

from otbench.utils import add_temporal_hour, add_temporal_hour_weight


@pytest.fixture(scope='module')
def sample_data():
    df = pd.DataFrame({
        "time": pd.date_range("2020-01-01", "2020-01-02", freq="1h"),
        "air_temperature": 10,
        "humidity": 0.5,
        "wind_speed": 10,
        "temporal_hour": 0,
        "temporal_hour_weight": 0,
    })

    yield df


def test_add_temporal_hour_and_temporal_hour_weight(sample_data):
    """Test adding temporal hour to a DataFrame."""
    df = add_temporal_hour(
        sample_data,
        name="test",
        timezone="America/New_York",
        obs_lat=40.7128,
        obs_lon=74.0060,
        time_col_name="time",
        temporal_hour_col_name="temporal_hour",
    )
    assert "temporal_hour" in df.columns
    # Test adding temporal hour weight to a DataFrame
    df = add_temporal_hour_weight(
        df,
        temporal_hour_col_name="temporal_hour",
        temporal_hour_weight_col_name="temporal_hour_weight",
    )
    assert "temporal_hour_weight" in df.columns


def test_add_temporal_hour_time_index(sample_data):
    """Test adding temporal hour to a DataFrame."""
    df = sample_data.set_index("time")
    df = add_temporal_hour(
        df,
        name="test",
        timezone="America/New_York",
        obs_lat=40.7128,
        obs_lon=74.0060,
        time_col_name="time",
        temporal_hour_col_name="temporal_hour",
    )
    assert "temporal_hour" in df.columns


def test_add_temporal_hour_sunrise_noon_sunset_boundaries():
    """Verify temporal hour equals 0 at sunrise, 6 at midpoint, and 12 at sunset."""
    from astral import LocationInfo
    from astral.sun import sun
    import numpy as np

    obs = LocationInfo(name="NYC", region="", timezone="America/New_York", latitude=40.7128, longitude=-74.0060)
    s = sun(obs.observer, date=pd.Timestamp("2020-06-15").date(), tzinfo=obs.timezone)
    sunrise = pd.Timestamp(s["sunrise"]).tz_convert("America/New_York").tz_localize(None)
    sunset = pd.Timestamp(s["sunset"]).tz_convert("America/New_York").tz_localize(None)
    midpoint = sunrise + (sunset - sunrise) / 2

    df_in = pd.DataFrame({"time": [sunrise, midpoint, sunset]})
    df_out = add_temporal_hour(
        df_in,
        name="NYC",
        timezone="America/New_York",
        obs_lat=40.7128,
        obs_lon=-74.0060,
        time_col_name="time",
        temporal_hour_col_name="temporal_hour",
    )
    assert "temporal_hour" not in df_in.columns
    np.testing.assert_allclose(df_out["temporal_hour"].to_numpy(), [0.0, 6.0, 12.0], atol=1e-6)
    df_weighted = add_temporal_hour_weight(df_out, "temporal_hour", "temporal_hour_weight")
    np.testing.assert_allclose(df_weighted["temporal_hour_weight"].to_numpy(), [0.05, 0.90, 0.08], atol=1e-6)

import pytest
import pandas as pd

from otb.utils import add_temporal_hour, add_temporal_hour_weight


@pytest.fixture(scope='module')
def sample_data():
    df = pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", "2020-01-02", freq="1H"),
            "air_temperature": 10,
            "humidity": 0.5,
            "wind_speed": 10,
            "temporal_hour": 0,
            "temporal_hour_weight": 0,
        }
    )

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

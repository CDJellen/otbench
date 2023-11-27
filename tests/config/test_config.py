import pytest

import otbench.config as otb_config


@pytest.fixture(scope='module')
def config():
    """Return a function that yields a pointer to the config."""
    yield otb_config


def test_config(config):
    """Test the attributes of the config."""
    assert isinstance(config.CACHE_DIR, str)
    assert isinstance(config.DATA_DIR, str)
    assert isinstance(config.DATASETS_FP, str)
    assert isinstance(config.RETURN_TYPES, list)
    assert isinstance(config.ROOT_DIR, str)
    assert isinstance(config.TASKS_FP, str)
    assert isinstance(config.BENCHMARK_FP, str)

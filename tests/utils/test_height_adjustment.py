import pytest
import pandas as pd

from otb.utils import apply_fried_height_adjustment, apply_oermann_height_adjustment


@pytest.fixture(scope='module')
def sample_data():
    cn2 = pd.Series([1e-16 for _ in range(10)])
    yield cn2


def test_apply_fried_height_adjustment(sample_data):
    """Test applying the Fried height adjustment."""
    cn2 = sample_data.copy()
    cn2 = apply_fried_height_adjustment(cn2, 1, 10)
    assert cn2.shape == (10,)

    with pytest.raises(ValueError):
        apply_fried_height_adjustment(cn2, 0, 0)
    with pytest.raises(ValueError):
        apply_fried_height_adjustment(cn2, -1, 0)


def test_apply_oermann_height_adjustment(sample_data):
    """Test applying the Fried height adjustment."""
    cn2 = sample_data.copy()
    cn2 = apply_oermann_height_adjustment(cn2, 1, 10)
    assert cn2.shape == (10,)

    with pytest.raises(ValueError):
        apply_oermann_height_adjustment(cn2, 0, 0)
    with pytest.raises(ValueError):
        apply_oermann_height_adjustment(cn2, -1, 0)

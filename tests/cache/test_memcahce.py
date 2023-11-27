import os

import pytest
import pandas as pd

from otbench.cache import InMemoryCache
from otbench.config import CACHE_DIR


@pytest.fixture(scope='module')
def in_memory_cache():
    yield InMemoryCache()


def test_add_dataset(in_memory_cache):
    """Test adding a dataset to the cache."""
    in_memory_cache.add_dataset(name='test', dataset=pd.DataFrame())
    assert 'test' in in_memory_cache.available_datasets()
    remove_from_cache(name='test')


def test_available_datasets(in_memory_cache):
    """Test listing available datasets."""
    assert isinstance(in_memory_cache.available_datasets(), list)


def test_mem_datasets(in_memory_cache):
    """Test listing datasets in memory."""
    assert isinstance(in_memory_cache.mem_datasets(), list)


def test_get_dataset(in_memory_cache):
    """Test getting a dataset from the cache."""
    in_memory_cache.add_dataset(name='test', dataset=pd.DataFrame())
    assert isinstance(in_memory_cache.get_dataset(key='test'), pd.DataFrame)
    with pytest.raises(NotImplementedError):
        in_memory_cache.get_dataset(key='not_in_cache')
    remove_from_cache(name='test')


def test_is_in_memory(in_memory_cache):
    """Test checking if a dataset is in memory."""
    in_memory_cache.add_dataset(name='test', dataset=pd.DataFrame())
    assert in_memory_cache._is_in_memory(key='test')
    assert not in_memory_cache._is_in_memory(key='not_in_cache')
    remove_from_cache(name='test')


def test_is_on_disk(in_memory_cache):
    """Test checking if a dataset is on disk."""
    in_memory_cache.add_dataset(name='test', dataset=pd.DataFrame())
    assert in_memory_cache._is_on_disk(key='test')
    assert not in_memory_cache._is_on_disk(key='not_in_cache')
    remove_from_cache(name='test')


def test_cache_dataset(in_memory_cache):
    """Test caching a dataset."""
    in_memory_cache.add_dataset(name='test', dataset=pd.DataFrame())
    assert in_memory_cache._is_on_disk(key='test')
    assert in_memory_cache._is_in_memory(key='test')
    with pytest.raises(KeyError):
        in_memory_cache._cache_dataset(key='not_in_cache')
    remove_from_cache(name='test')


def test_load_dataset(in_memory_cache, capfd):
    """Test loading a dataset."""
    in_memory_cache.add_dataset(name='test', dataset=pd.DataFrame())
    assert in_memory_cache._is_in_memory(key='test')
    in_memory_cache._load_dataset(key='test')
    assert in_memory_cache._is_in_memory(key='test')
    # assert notification in capfd.readouterr().out
    in_memory_cache._load_dataset(key='not_in_cache')
    out, _ = capfd.readouterr()
    assert f"failed to load dataset with key 'not_in_cache'" in out
    remove_from_cache(name='test')


def test_iter(in_memory_cache):
    """Test iterating over the cache."""
    in_memory_cache.add_dataset(name='test', dataset=pd.DataFrame())
    df = next(iter(in_memory_cache))
    assert isinstance(df, pd.DataFrame)
    remove_from_cache(name='test')


def test_contains(in_memory_cache):
    """Test checking if a dataset is in the cache."""
    in_memory_cache.add_dataset(name='test', dataset=pd.DataFrame())
    assert in_memory_cache.__contains__(key='test')
    assert not in_memory_cache.__contains__(key='not_in_cache')
    remove_from_cache(name='test')


def test_len(in_memory_cache):
    """Test checking the length of the cache."""
    in_memory_cache.add_dataset(name='test', dataset=pd.DataFrame())
    assert len(in_memory_cache) == 1
    remove_from_cache(name='test')


def test_getitem(in_memory_cache):
    """Test getting an item from the cache."""
    in_memory_cache.add_dataset(name='test', dataset=pd.DataFrame())
    assert isinstance(in_memory_cache['test'], pd.DataFrame)
    assert in_memory_cache['not_in_cache'] is None
    remove_from_cache(name='test')


def remove_from_cache(name: str) -> None:
    """Remove any datasets created during testing."""
    for fp in os.listdir(CACHE_DIR):
        if name in str(fp):
            os.remove(os.path.join(CACHE_DIR, fp))

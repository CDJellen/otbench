import os
import pickle
from typing import List, Union, Any

import pandas as pd

from otbench.config import CACHE_DIR


class InMemoryCache:

    def __init__(self, cache_dir: Union[str, os.PathLike, None] = CACHE_DIR) -> None:
        self._cache = dict()
        self._cache_dir = cache_dir

    def add_dataset(self, name: str, dataset: Any) -> None:
        """Adds a new dataset to the cache, persisting to disk if needed."""
        self._cache[name] = dataset
        self._cache_dataset(key=name)

    def available_datasets(self) -> List[str]:
        """List datasets available in memory or on disk"""
        in_memory = self.mem_datasets()
        on_disk = [fp.strip(".pickle") for fp in os.listdir(self._cache_dir) if ".gitkeep" not in str(fp)]
        return list(set(in_memory + on_disk))

    def mem_datasets(self) -> List[str]:
        """List the datasets available in memory."""
        return list(self._cache.keys())

    def get_dataset(self, key: str) -> Any:
        """Get a dataset from memory or disk."""
        if key not in self.available_datasets():
            raise KeyError(f"Dataset '{key}' not found in cache or on disk.")
        if self._is_in_memory(key):
            return self._cache[key]
        # Key exists on disk but not in memory — load it.
        self._load_dataset(key)
        if self._is_in_memory(key):
            return self._cache[key]
        raise RuntimeError(f"Failed to load dataset '{key}' from disk cache.")

    def _is_in_memory(self, key: str) -> bool:
        """Check if a dataset is available in memory."""
        return self.__contains__(key=key)

    def _is_on_disk(self, key: str) -> bool:
        """Check if a dataset was processed and is on disk."""
        on_disk = [fp for fp in os.listdir(self._cache_dir) if ".gitkeep" not in str(fp)]

        for fp in on_disk:
            if fp.split(".pickle")[0] == key:
                return True
        return False

    def _cache_dataset(self, key) -> None:
        """Save a dataset from memory to disk."""
        if key not in self._cache:
            raise KeyError(f"no dataset named {key}.")
        data = self._cache[key]
        with open(os.path.join(self._cache_dir, f"{key}.pickle"), "wb") as f:
            pickle.dump(data, f)

    def _load_dataset(self, key) -> None:
        """Load a dataset from disk to memory."""
        fp = os.path.join(self._cache_dir, f"{key}.pickle")
        try:
            with open(fp, "rb") as f:
                data = pickle.load(f)
            self._cache[key] = data
        except FileNotFoundError:
            raise
        except (pickle.UnpicklingError, EOFError, ModuleNotFoundError) as e:
            import warnings
            warnings.warn(
                f"Corrupted cache entry for '{key}' at {fp}: {e}. "
                f"Delete the file and re-run to regenerate."
            )

    def __iter__(self) -> pd.DataFrame:
        for key in list(self._cache.keys()):
            yield self._cache[key]

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, key) -> Union[pd.DataFrame, None]:
        if key in self._cache:
            return self._cache[key]
        else:
            return None

    def __contains__(self, key) -> bool:
        return True if key in self._cache else False

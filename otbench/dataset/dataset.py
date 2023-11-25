import os
import json
from typing import Any, Union, Tuple, Sequence

import pandas as pd
import numpy as np
import xarray as xr

from otbench.config import ROOT_DIR, DATA_DIR, DATASETS_FP, CACHE_DIR, RETURN_TYPES
from otbench.cache import CACHE


class Dataset(object):
    """A singleton helper for in-memory datasets."""

    def __init__(self,
                 name: str,
                 datasets_fp: Union[str, os.PathLike, None] = DATASETS_FP,
                 root_dir: Union[str, os.PathLike, None] = ROOT_DIR,
                 data_dir: Union[str, os.PathLike, None] = DATA_DIR,
                 cache_dir: Union[str, os.PathLike, None] = CACHE_DIR) -> None:
        """Read the currently-supported benchmarking task for loaders and evaluators."""
        self._name = name
        self._datasets_fp = datasets_fp
        self._root_dir = root_dir
        self._data_dir = data_dir
        self._cache_dir = cache_dir
        self._df = self._load_dataset()

    def get_slice(self, start_indices: Sequence[int], end_indices: Sequence[int]) -> pd.DataFrame:
        """Obtain a slice of the underlying dataset from start and end indices."""
        if len(start_indices) == 0 or len(start_indices) != len(end_indices):
            raise ValueError(f"malformed {start_indices}, {end_indices}.")
        ranges = []
        for start_idx, end_idx in zip(start_indices, end_indices):
            if start_idx >= 0 and end_idx <= len(self._df) and start_idx < end_idx:
                ranges.append(np.arange(start_idx, end_idx))
            else:
                raise ValueError(f"requested {start_idx}:{end_idx} out of bounds for df with len {len(self._df)}.")
        included = np.concatenate(ranges)
        return self._df.iloc[included, :].copy(deep=True)

    def get_all(self, data_type: str = "pd", device: str = "") -> Any:
        """Obtain the training data for this dataset from the supplied task."""
        return self._handle_return_type(data=self._df, return_type=data_type)

    def get_train(self, task: dict, data_type: str = "pd") -> Tuple[Any, Any]:
        """Obtain the training data for this dataset from the supplied task."""
        indices = [int(i) for i in task["train_idx"] for i in i.split(":")]
        starts, stops = indices[::2], indices[1::2]
        data = self.get_slice(starts, stops)
        X, y = self._handle_task(data=data, task=task)
        return self._handle_return_type(data=X, return_type=data_type), self._handle_return_type(data=y,
                                                                                                 return_type=data_type)

    def get_test(self, task: dict, data_type: str = "pd") -> Tuple[Any, Any]:
        """Obtain the test data for this dataset from the supplied task."""
        indices = [int(i) for i in task["test_idx"] for i in i.split(":")]
        starts, stops = indices[::2], indices[1::2]
        data = self.get_slice(starts, stops)
        X, y = self._handle_task(data=data, task=task)
        return self._handle_return_type(data=X, return_type=data_type), self._handle_return_type(data=y,
                                                                                                 return_type=data_type)

    def get_val(self, task: dict, data_type: str = "pd") -> Tuple[Any, Any]:
        """Obtain the validation data for this dataset from the supplied task."""
        indices = [int(i) for i in task["val_idx"] for i in i.split(":")]
        starts, stops = indices[::2], indices[1::2]
        data = self.get_slice(starts, stops)
        X, y = self._handle_task(data=data, task=task)
        return self._handle_return_type(data=X, return_type=data_type), self._handle_return_type(data=y,
                                                                                                 return_type=data_type)

    def _handle_task(self, data: pd.DataFrame, task: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split into features and target, dropping missing and transforming target if needed."""
        if task["dropna"]:
            data = data.dropna()
        X = data[[c for c in data.columns if c not in task["remove"]]]
        y = data[[task["target"]]]
        if task["log_transform"]:
            y = np.log10(y)
        return X, y

    def _handle_return_type(self, data: pd.DataFrame, return_type: str) -> Any:
        """Map the slice of underlying data to the requested type."""
        if return_type not in RETURN_TYPES:
            raise NotImplementedError(f"return type {return_type} not implemented.")
        # switch case
        if return_type == "pd":
            return data
        return getattr(self, f"_convert_to_{return_type}")(data)

    def _convert_to_np(self, data: pd.DataFrame) -> np.ndarray:
        """Map the slice of underlying data to np ndarray."""
        nd_arr = data.to_numpy()
        return nd_arr

    def _convert_to_xr(self, data: pd.DataFrame) -> xr.Dataset:
        """Map the slice of underlying data to xr xarray."""
        ds = data.set_index("time").to_xarray()
        return ds

    def _convert_to_nc(self, data: pd.DataFrame) -> xr.Dataset:
        """Map the slice of underlying data to netCDF (an alias for xr.DataSet)."""
        return self._convert_to_xr(data)

    def _load_dataset(self) -> pd.DataFrame:
        """Load the dataset from cache or disk."""
        if self._name in CACHE:
            return CACHE.get_dataset(self._name)
        else:
            return self._load_dataset_from_disk()

    def _load_dataset_from_disk(self) -> pd.DataFrame:
        """Load the dataset from disk."""
        supported_datasets = self._supported_datasets()
        file_name = supported_datasets[self._name]["local_data_path"]
        fp = os.path.join(self._data_dir, self._name, file_name)
        fp_str = str(fp)
        try:
            file_type = fp_str.split(".")[-1]
        except IndexError:
            raise NotImplementedError(f"unknown or unsupported file type {fp}.")
        # netcdf
        if file_type == "nc":
            ds = xr.load_dataset(fp)
            df = ds.to_dataframe()
        else:
            raise NotImplementedError(f"unknown or unsupported file type {fp}.")

        # update the cache
        CACHE.add_dataset(self._name, df)

        return df

    def _supported_datasets(self) -> dict:
        """Load the datasets configuration file."""
        supported_datasets = json.load(open(self._datasets_fp, 'rb'))

        return supported_datasets

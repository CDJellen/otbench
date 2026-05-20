import os
import json
from typing import Any, Union, Tuple, Sequence, List

import pandas as pd
import numpy as np
import xarray as xr

from otbench.config import settings, RETURN_TYPES
from otbench.cache import CACHE
from otbench.models import Task
from otbench.dataset.synthetic import SYNTHETIC_REGISTRY


class Dataset(object):
    """A singleton helper for in-memory datasets."""

    def __init__(self,
                 name: str,
                 datasets_fp: Union[str, os.PathLike, None] = settings.DATASETS_FP,
                 root_dir: Union[str, os.PathLike, None] = settings.ROOT_DIR,
                 data_dir: Union[str, os.PathLike, None] = settings.DATA_DIR,
                 cache_dir: Union[str, os.PathLike, None] = settings.CACHE_DIR) -> None:
        """Read the currently-supported benchmarking task for loaders and evaluators."""
        self._name = name
        self._datasets_fp = datasets_fp
        self._root_dir = root_dir
        self._data_dir = data_dir
        self._cache_dir = cache_dir
        self._data: Union[pd.DataFrame, xr.Dataset] = self._load_dataset()

    def get_slice(self, start_indices: Sequence[int], end_indices: Sequence[int]) -> Union[pd.DataFrame, xr.Dataset]:
        """Obtain a slice of the underlying dataset from start and end indices."""
        if len(start_indices) == 0 or len(start_indices) != len(end_indices):
            raise ValueError(f"malformed {start_indices}, {end_indices}.")

        # Handle pandas DataFrame
        if isinstance(self._data, pd.DataFrame):
            ranges = []
            for start_idx, end_idx in zip(start_indices, end_indices):
                if start_idx >= 0 and end_idx <= len(self._data) and start_idx < end_idx:
                    ranges.append(np.arange(start_idx, end_idx))
                else:
                    raise ValueError(
                        f"requested {start_idx}:{end_idx} out of bounds for df with len {len(self._data)}.")
            included = np.concatenate(ranges)
            return self._data.iloc[included, :].copy(deep=True)

        # Handle xarray Dataset
        elif isinstance(self._data, xr.Dataset):
            # Assumes 'time' is the primary dimension for slicing
            if 'time' not in self._data.dims:
                raise ValueError("xarray Dataset must have 'time' dimension for slicing.")

            slices = []
            max_len = len(self._data.time)
            for start_idx, end_idx in zip(start_indices, end_indices):
                if start_idx >= 0 and end_idx <= max_len and start_idx < end_idx:
                    # isel slicing is efficient
                    slices.append(self._data.isel(time=slice(start_idx, end_idx)))
                else:
                    raise ValueError(f"requested {start_idx}:{end_idx} out of bounds for ds with len {max_len}.")

            if not slices:
                return xr.Dataset()
            return xr.concat(slices, dim="time")

        else:
            raise NotImplementedError(f"Unsupported data type: {type(self._data)}")

    def get_context(self, indices: Any, columns: Union[str, List[str]], data_type: str = "pd") -> Any:
        """
        Retrieves "Contextual Metadata" (e.g., night_id) for the given indices from the raw dataset.
        
        This allows for the recovery of columns that were removed during task processing (via the 'remove' list)
        but are necessary for downstream visualization or analysis (e.g., grouping by observing session).
        
        Args:
            indices: The indices (row identifiers) corresponding to the data you currently have.
                     Usually X_train.index or X_test.index.
            columns: The list of column names to recover (e.g. ['night_id']).
            data_type: The return format ('pd', 'np', 'xr', 'nc').
        """
        if isinstance(columns, str):
            columns = [columns]

        context_slice = None

        # 1. Pandas Implementation
        if isinstance(self._data, pd.DataFrame):
            # Verify columns exist
            missing = [c for c in columns if c not in self._data.columns]
            if missing:
                raise ValueError(f"Context columns {missing} not found in source dataset.")

            # Use .loc to retrieve rows by the user's index (TimeIndex or RangeIndex)
            try:
                context_slice = self._data.loc[indices, columns]
            except KeyError:
                # If indices don't align, it might be a Type mismatch (Int vs DateTime)
                raise KeyError(
                    f"Provided indices could not be located in source dataset index ({type(self._data.index)}).")

        # 2. Xarray Implementation
        elif isinstance(self._data, xr.Dataset):
            # Verify variables exist
            missing = [c for c in columns if c not in self._data.data_vars and c not in self._data.coords]
            if missing:
                raise ValueError(f"Context variables {missing} not found in source dataset.")

            # Xarray selection requires values, not a Pandas Index object usually
            if hasattr(indices, 'values'):
                sel_indices = indices.values
            else:
                sel_indices = indices

            try:
                # Assumes 'time' is the indexing dimension
                context_slice = self._data.sel(time=sel_indices)[columns]
            except Exception as e:
                raise KeyError(f"Could not select indices from xarray dataset: {e}")

        else:
            raise NotImplementedError(f"Storage type {type(self._data)} not supported.")

        return self._handle_return_type(context_slice, return_type=data_type)

    def get_xarray(self) -> 'Optional[xr.Dataset]':
        """Return the underlying xr.Dataset if the cache holds one, else None.

        For datasets opened with ``"lazy": true`` in datasets.json (e.g. paranal_tomography)
        the backing store is a memory-mapped xr.Dataset.  This method exposes
        it for schema inspection and testing without materialising a flat DataFrame.
        """
        if isinstance(self._data, xr.Dataset):
            return self._data
        return None

    def get_sample_df(self, n: int = 5000) -> pd.DataFrame:
        """Return up to *n* rows as a flat pandas DataFrame.

        Safe to call on large datasets because it only materialises a bounded
        slice rather than the full dataset.  Intended for schema/column
        validation in tests and exploratory analysis.
        """
        if isinstance(self._data, xr.Dataset):
            n = min(n, len(self._data.time))
        else:
            n = min(n, len(self._data))
        sample = self.get_slice([0], [n])
        return self._convert_to_pd(sample)

    def get_all(self, data_type: str = "pd", device: str = "") -> Any:
        """Obtain the training data for this dataset from the supplied task."""
        return self._handle_return_type(data=self._data, return_type=data_type)

    def get_train(self, task: Union[dict, Task], data_type: str = "pd") -> Tuple[Any, Any]:
        """Obtain the training data for this dataset from the supplied task."""
        if isinstance(task, dict):
            task = Task(**task)

        indices = [int(i) for i in task.train_idx for i in i.split(":")]
        starts, stops = indices[::2], indices[1::2]
        data = self.get_slice(starts, stops)
        X, y = self._handle_task(data=data, task=task)
        return self._handle_return_type(data=X, return_type=data_type), self._handle_return_type(data=y,
                                                                                                 return_type=data_type)

    def get_test(self, task: Union[dict, Task], data_type: str = "pd") -> Tuple[Any, Any]:
        """Obtain the test data for this dataset from the supplied task."""
        if isinstance(task, dict):
            task = Task(**task)

        indices = [int(i) for i in task.test_idx for i in i.split(":")]
        starts, stops = indices[::2], indices[1::2]
        data = self.get_slice(starts, stops)
        X, y = self._handle_task(data=data, task=task)
        return self._handle_return_type(data=X, return_type=data_type), self._handle_return_type(data=y,
                                                                                                 return_type=data_type)

    def get_val(self, task: Union[dict, Task], data_type: str = "pd") -> Tuple[Any, Any]:
        """Obtain the validation data for this dataset from the supplied task."""
        if isinstance(task, dict):
            task = Task(**task)

        indices = [int(i) for i in task.val_idx for i in i.split(":")]
        starts, stops = indices[::2], indices[1::2]
        data = self.get_slice(starts, stops)
        X, y = self._handle_task(data=data, task=task)
        return self._handle_return_type(data=X, return_type=data_type), self._handle_return_type(data=y,
                                                                                                 return_type=data_type)

    def _handle_task(self, data: Union[pd.DataFrame, xr.Dataset],
                     task: Task) -> Tuple[Union[pd.DataFrame, xr.Dataset], Union[pd.DataFrame, xr.Dataset]]:
        """Split into features and target, dropping missing and transforming target if needed."""
        # Lazy flattening: if the cache stores an xr.Dataset (memory-mapped), flatten
        # only the requested slice here rather than pre-flattening the full dataset.
        if isinstance(data, xr.Dataset):
            data = self._flatten_dataset(data)

        # Handle pandas DataFrame
        if isinstance(data, pd.DataFrame):
            if task.dropna:
                data = data.dropna()
            X = data[[c for c in data.columns if c not in task.remove]]

            # Handle string or list target
            if isinstance(task.target, list):
                y = data[task.target]
            else:
                y = data[[task.target]]

            if task.log_transform:
                y = np.log10(y.clip(lower=1e-19))
            return X, y

        # Handle xarray Dataset
        elif isinstance(data, xr.Dataset):
            if task.dropna:
                # dropna along time dimension if any variable is nan
                data = data.dropna(dim='time', how='any')

            # Remove variables
            # drop_vars returns a new dataset
            X = data.drop_vars(task.remove, errors='ignore')

            # Select target(s)
            if isinstance(task.target, list):
                y = data[task.target]
            else:
                y = data[[task.target]]

            if task.log_transform:
                y = np.log10(y.clip(lower=1e-19))

            return X, y

        else:
            raise NotImplementedError(f"Unsupported data type: {type(data)}")

    def _handle_return_type(self, data: Union[pd.DataFrame, xr.Dataset], return_type: str) -> Any:
        """Map the slice of underlying data to the requested type."""
        if return_type not in RETURN_TYPES:
            raise NotImplementedError(f"return type {return_type} not implemented.")

        # If requested pd and data is pd, return
        if return_type == "pd" and isinstance(data, pd.DataFrame):
            return data

        # If requested xr/nc and data is xr, return
        if return_type in ["xr", "nc"] and isinstance(data, xr.Dataset):
            return data

        return getattr(self, f"_convert_to_{return_type}")(data)

    def _convert_to_pd(self, data: Union[pd.DataFrame, xr.Dataset]) -> pd.DataFrame:
        """Convert data to a flat pandas DataFrame."""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, xr.Dataset):
            needs_flattening = any(len(data[v].dims) > 1 for v in data.data_vars)
            return self._flatten_dataset(data) if needs_flattening else data.to_dataframe()
        raise NotImplementedError(f"Cannot convert {type(data)} to pd.DataFrame")

    def _convert_to_np(self, data: Union[pd.DataFrame, xr.Dataset]) -> np.ndarray:
        """Map the slice of underlying data to np ndarray."""
        if isinstance(data, pd.DataFrame):
            return data.to_numpy()
        elif isinstance(data, xr.Dataset):
            return data.to_array().values
        else:
            raise NotImplementedError

    def _convert_to_xr(self, data: Union[pd.DataFrame, xr.Dataset]) -> xr.Dataset:
        """Map the slice of underlying data to xr xarray."""
        if isinstance(data, pd.DataFrame):
            # Safely check if time is already the index
            if "time" in data.columns:
                ds = data.set_index("time").to_xarray()
            else:
                ds = data.to_xarray()
            return ds
        elif isinstance(data, xr.Dataset):
            return data
        else:
            raise NotImplementedError

    def _convert_to_nc(self, data: pd.DataFrame) -> xr.Dataset:
        """Map the slice of underlying data to netCDF (an alias for xr.DataSet)."""
        return self._convert_to_xr(data)

    def _load_dataset(self) -> Union[pd.DataFrame, xr.Dataset]:
        """Load the dataset from cache or disk."""
        if self._name in CACHE:
            return CACHE.get_dataset(self._name)
        else:
            data = self._load_dataset_from_disk()
            # update the cache
            CACHE.add_dataset(self._name, data)
            return data

    def _flatten_dataset(self, ds: xr.Dataset) -> pd.DataFrame:
        """
        Flattens a multi-dimensional xarray Dataset into a 2D DataFrame.
        Variables with extra dimensions (e.g. height) are pivoted into columns.
        """
        dfs = []
        for var_name, da in ds.data_vars.items():
            if 'time' not in da.dims:
                continue

            if len(da.dims) == 1:
                # 1D variable (time,) -> Column name is just 'var_name'
                dfs.append(da.to_dataframe())
            else:
                # Multi-dimensional variable (time, dim1, ...)
                # Stack all non-time dimensions
                other_dims = [d for d in da.dims if d != 'time']

                # Unstack creates a MultiIndex column: (var_name, dim_val1, dim_val2...)
                temp_df = da.to_dataframe().unstack(level=other_dims)

                # Flatten MultiIndex columns
                new_columns = []
                for col in temp_df.columns:
                    # 'col' is a tuple, e.g., ('cn2_free_atmos', 500)
                    if isinstance(col, tuple):
                        # Join them directly. This preserves "varname_dimval"
                        new_columns.append("_".join(map(str, col)))
                    else:
                        # Fallback for simple indexes
                        new_columns.append(str(col))

                temp_df.columns = new_columns
                dfs.append(temp_df)

        if not dfs:
            return pd.DataFrame()

        # Concatenate all parts along columns (axis=1), aligning on index (time)
        return pd.concat(dfs, axis=1)

    def _load_dataset_from_disk(self) -> Union[pd.DataFrame, xr.Dataset]:
        """Load the dataset from disk."""
        supported_datasets = self._supported_datasets()

        if settings.USE_SYNTHETIC_DATA:
            if self._name in SYNTHETIC_REGISTRY:
                # Bypass disk loading completely for synthetic data
                ds = SYNTHETIC_REGISTRY[self._name]()

                # Check if we need to flatten (mirroring disk logic)
                dataset_config = supported_datasets.get(self._name, {})
                should_flatten = dataset_config.get("flatten", True)

                if should_flatten:
                    # Check if it actually needs flattening (dims > 1)
                    needs_flattening = any(len(ds[v].dims) > 1 for v in ds.data_vars)
                    if needs_flattening:
                        return self._flatten_dataset(ds)
                    else:
                        return ds.to_dataframe()
                return ds
            else:
                pass
        file_name = supported_datasets[self._name]["local_data_path"]
        fp = settings.DATA_DIR / self._name / file_name
        fp_str = str(fp)
        try:
            file_type = fp_str.split(".")[-1]
        except IndexError:
            raise NotImplementedError(f"unknown or unsupported file type {fp}.")
        # netcdf
        if file_type == "nc":
            dataset_config = supported_datasets.get(self._name, {})
            should_flatten = dataset_config.get("flatten", True)
            is_lazy = dataset_config.get("lazy", False)

            if is_lazy:
                # Lazy path (e.g. paranal_tomography): open_dataset is memory-mapped.
                # Data is not resident in RAM until a slice is requested via get_slice.
                # Flattening is deferred to _handle_task so only the active split
                # is ever materialised as a flat DataFrame.
                return xr.open_dataset(fp)

            # Eager path: load and flatten the full dataset into a DataFrame.
            ds = xr.load_dataset(fp)
            if should_flatten:
                needs_flattening = any(len(ds[v].dims) > 1 for v in ds.data_vars)
                if needs_flattening:
                    return self._flatten_dataset(ds)
                else:
                    return ds.to_dataframe()
            else:
                return ds
        else:
            raise NotImplementedError(f"unknown or unsupported file type {fp}.")

    def _supported_datasets(self) -> dict:
        """Load the datasets configuration file."""
        supported_datasets = json.load(open(self._datasets_fp, 'rb'))

        return supported_datasets

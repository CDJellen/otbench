import os
from typing import Optional

import pandas as pd
import xarray as xr


class Datasets(object):
    """A singleton helper for in-memory datasets."""

    def __init__(self, root_dir: Optional[str] = None) -> None:
        """Read the currently-supported benchmarking task for loaders and evaluators."""
        self.data_sets = {}

        if root_dir is None:
            root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))  # obtain root path
        ds_dir = os.path.join(root_dir, 'datasets')
        data_dir = os.path.join(root_dir, 'data')

        # remember the data dir abspath
        self.data_dir = data_dir
        
        # load cached datasets
        for ds_fn in os.listdir(ds_dir):
            if '.parquet.gzip' in ds_fn:
                ds = pd.read_parquet(os.path.join(ds_dir, ds_fn))
                ds_name = ds_fn.split('.parquet')[0]
                self.data_sets[ds_name] = ds
        
    def is_in_memory(self, ds_name: str) -> bool:
        """Check whether a dataset is already in memory."""
        return ds_name in self.data_sets
    
    def load_dataset_to_cache(self, ds_name: str) -> None:
        """Read data from disk and persist to in-memory cache."""
        if os.path.exists(os.path.join(self.data_dir, ds_name)):
            # check if the data is serialized as netcdf (`.nc`)
            if os.path.exists(os.path.join(self.data_dir, ds_name, f'{ds_name}.nc')):
                ds = xr.load_dataset(os.path.join(self.data_dir, ds_name, f'{ds_name}.nc'))
                df = ds.to_dataframe()
            # check if the data is serialized as `parquet.gzip`
            elif os.path.exists(os.path.join(self.data_dir, ds_name, f'{ds_name}.parquet.gzip')):
                df = pd.read_parquet(os.path.join(self.data_dir, ds_name, f'{ds_name}.parquet.gzip'))
            # check if the data is serialized as `mat`
            elif os.path.exists(os.path.join(self.data_dir, ds_name, f'{ds_name}.mat')):
                raise NotImplementedError('TODO implement loading from `.mat`.')
            
            self.data_sets[ds_name] = df

        else:
            raise ValueError(f'Missing data dependencies for dataset {ds_name}.')

    def get_train(self, task: dict) -> pd.DataFrame:
        """"""
        if not self.is_in_memory(task['ds_name']):
            try:
                self.load_dataset_to_cache(task['ds_name'])
            except ValueError as e:
                raise ValueError('Failed to obtain training data due to missing dependency') from e
        
        # obtain the task-specific indices for the dataset
        train_idx = []

        for train_range in task['train_idx']:
            range_start, range_end = train_range.split(':')
            range_start, range_end = int(range_start), int(range_end)
            train_idx.extend(list(range(range_start, range_end)))

        df = self.data_sets[task['ds_name']]

        df_train = df.iloc[train_idx]
        return df_train.drop(columns=task['remove'])

    def get_test(self, task: dict) -> pd.DataFrame:
        """"""
        if not self.is_in_memory(task['ds_name']):
            try:
                self.load_dataset_to_cache(task['ds_name'])
            except ValueError as e:
                raise ValueError('Failed to obtain training data due to missing dependency') from e
        
        # obtain the task-specific indices for the dataset
        test_idx = []

        for test_range in task['test_idx']:
            range_start, range_end = test_range.split(':')
            range_start, range_end = int(range_start), int(range_end)
            test_idx.extend(list(range(range_start, range_end)))

        df = self.data_sets[task['ds_name']]

        df_test = df.iloc[test_idx]
        return df_test.drop(columns=task['remove'])

    def dataset_full(self, task: dict) -> pd.DataFrame:
        """"""
        if not self.is_in_memory(task['ds_name']):
            try:
                self.load_dataset_to_cache(task['ds_name'])
            except ValueError as e:
                raise ValueError('Failed to obtain training data due to missing dependency') from e
        
        return self.data_sets[task['ds_name']]

    def dataset_train(self, task: dict) -> pd.DataFrame:
        """"""
        if not self.is_in_memory(task['ds_name']):
            try:
                self.load_dataset_to_cache(task['ds_name'])
            except ValueError as e:
                raise ValueError('Failed to obtain training data due to missing dependency') from e
        
        # obtain the task-specific indices for the dataset
        train_idx = []

        for train_range in task['train_idx']:
            range_start, range_end = train_range.split(':')
            range_start, range_end = int(range_start), int(range_end)
            train_idx.extend(list(range(range_start, range_end)))

        df = self.data_sets[task['ds_name']]

        return df.iloc[train_idx]

    def dataset_test(self, task: dict) -> pd.DataFrame:
        """"""
        if not self.is_in_memory(task['ds_name']):
            try:
                self.load_dataset_to_cache(task['ds_name'])
            except ValueError as e:
                raise ValueError('Failed to obtain training data due to missing dependency') from e
        
        # obtain the task-specific indices for the dataset
        test_idx = []

        for test_range in task['test_idx']:
            range_start, range_end = test_range.split(':')
            range_start, range_end = int(range_start), int(range_end)
            test_idx.extend(list(range(range_start, range_end)))

        df = self.data_sets[task['ds_name']]

        return df.iloc[test_idx]

    def dataset_from_task_name(self, task_name: str) -> pd.DataFrame:
        """"""
        if not self.is_in_memory(task_name):
            try:
                self.load_dataset_to_cache(task_name)
            except ValueError as e:
                raise ValueError('Failed to obtain training data due to missing dependency') from e
        
        return self.data_sets[task_name]

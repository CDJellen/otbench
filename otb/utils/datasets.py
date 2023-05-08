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

    def get_train():
        pass

    def get_test():
        pass

    def remove_target():
        pass

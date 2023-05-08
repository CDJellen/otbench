import os
from typing import Optional

import numpy as np
import pandas as pd

from ..utils.eval_metrics import *
from ..utils.tasks import Tasks

# try import torch

t = Tasks()

class RegressionEvaluator:

    def __init__(self, task_name: str, root_dir: Optional[str]) -> None:
        """The base evaluator for regression tasks."""
        # check if task_name is a supported task
        if not t.is_supported_task(task_name):
            raise ValueError(f'Task {task_name} is not in supported tasks {t.task_names}')

        # load the values for `y_true` as an `np.ndarray
        if root_dir is None:
            # assume a root directory
            root_dir = os.path.dirname(os.path.dirname(__file__))
        
        task_ds_path = task_name.split('.')[0:2]
        ds_dir = os.path.join(root_dir, *task_ds_path)
        ds_fp = os.path.join(ds_dir, "ds.nc")

        # check if the dataset is already cached
        if os.path.exists(ds_fp):
            
        # if not, create from source
        

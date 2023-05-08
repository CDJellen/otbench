import os
from abc import ABCMeta
from typing import List, Optional, Callable

import numpy as np


class EvaluatorABC(metaclass=ABCMeta):
    """TODO"""

    def evaluate(self, dataset_name: str, eval_metric_name: str, preds: np.ndarray) -> dict:
        """take as input the dataset name, eval metric, and model predictions, return a dict summarizing results."""
        pass

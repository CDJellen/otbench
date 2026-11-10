from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from otbench.benchmark.models._pytorch_base import BasePyTorchMixin
from otbench.benchmark.models.regression.base_model import BaseRegressionModel


class BasePyTorchRegressionModel(BasePyTorchMixin, BaseRegressionModel):
    """Base class for PyTorch regression models."""

    def __init__(
        self,
        name: str,
        target_name: Union[str, List[str]],
        window_size: int = 1,
        batch_size: int = 32,
        n_epochs: int = 500,
        learning_rate: float = 0.025,
        criterion: Optional['torch.nn.modules.loss'] = None,
        optimizer: 'torch.optim' = optim.SGD,
        random_state: int = 2020,
        verbose: bool = False,
        weight_decay: float = 0.0,
        patience: int = None,
        val_fraction: float = 0.0,
        use_scheduler: bool = False,
    ):
        BaseRegressionModel.__init__(self, name=name, target_name=target_name)
        self.window_size = window_size
        self._init_pytorch_state(
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            criterion=criterion,
            optimizer=optimizer,
            random_state=random_state,
            verbose=verbose,
            weight_decay=weight_decay,
            patience=patience,
            val_fraction=val_fraction,
            use_scheduler=use_scheduler,
        )

    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]):
        raise NotImplementedError

    def predict(self, X: Union[pd.DataFrame, np.ndarray]):
        raise NotImplementedError

    def _set_dataloader_from_data(self,
                                  X: Union[pd.DataFrame, np.ndarray],
                                  y: Union[pd.DataFrame, np.ndarray],
                                  mode: str = "val") -> None:
        """Use the data supplied to create train or validation DataLoaders."""
        y = self._ensure_y_for_dataloader(X, y)

        if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
            n_features = len(X.columns) // self.window_size
            X = X.to_numpy().reshape(-1, self.window_size, n_features)
            if self.window_size > 1:
                X = np.ascontiguousarray(X[:, ::-1, :])
            y = y.to_numpy()
        elif isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
            X = X.copy()
            y = y.copy()
            if y.ndim == 1:
                y = y[:, np.newaxis]
            if X.ndim == 2:
                n_features = X.shape[1] // self.window_size
                X = X.reshape(-1, self.window_size, n_features)
                if self.window_size > 1:
                    X = np.ascontiguousarray(X[:, ::-1, :])
        else:
            raise ValueError("X and y must be both be pd.DataFrame objects or np.ndarray objects.")

        self._finalize_dataloader(X, y, mode=mode)

    def _run_training_loop(self, X: 'pd.DataFrame', y: 'pd.DataFrame', output_dim: int) -> None:
        """Shared training loop supporting temporal validation splits, LR warmup/decay,
        gradient clipping, and early stopping with best-checkpoint restoration.
        """
        if len(X) == 0:
            return

        y_arr = y.to_numpy() if isinstance(y, (pd.DataFrame, pd.Series)) else np.asarray(y)
        y_valid_mask = ~np.isnan(y_arr).any(axis=1) if y_arr.ndim > 1 else ~np.isnan(y_arr)
        if len(y_valid_mask) == len(X) and not np.all(y_valid_mask):
            X = X.iloc[y_valid_mask] if isinstance(X, pd.DataFrame) else X[y_valid_mask]
            y = y.iloc[y_valid_mask] if isinstance(y, (pd.DataFrame, pd.Series)) else y[y_valid_mask]
            if len(X) == 0:
                return

        if y.ndim > 1 and y.shape[1] != output_dim:
            if self.verbose:
                print(f"Warning: Target dimension {y.shape[1]} does not match model output size {output_dim}.")
            if output_dim == 1 and y.shape[1] > 1:
                raise ValueError(f"Model initialized with output_size=1 but target has {y.shape[1]} columns. "
                                 f"Ensure input_size and output_size are set correctly.")

        n_features = (len(X.columns) // self.window_size if isinstance(X, pd.DataFrame) else
                      (X.shape[-1] if X.ndim == 3 else X.shape[1] // self.window_size))
        if self.verbose:
            print(f"training data contains {n_features} features.")

        self._prepare_train_val_dataloaders(X, y)
        self._execute_epoch_loop()

    def _run_prediction(self, X: Union['pd.DataFrame', np.ndarray], output_dim: int) -> np.ndarray:
        """Shared batched inference across validation/test data."""
        if len(X) == 0:
            return np.empty((0, output_dim)) if output_dim > 1 else np.empty((0,))

        n_features = (len(X.columns) // self.window_size if isinstance(X, pd.DataFrame) else
                      (X.shape[-1] if X.ndim == 3 else X.shape[1] // self.window_size))
        if self.verbose:
            print(f"validation data contains {n_features} features.")
        self.set_validation_data(X=X, y=None)
        return self._collect_batched_predictions(output_dim=output_dim, empty_1d_for_scalar=True)

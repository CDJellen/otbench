from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from otb.benchmark.models.regression.base_model import BaseRegressionModel


class BasePyTorchRegressionModel(BaseRegressionModel):
    """A the base class for PyTorch models."""

    def __init__(
        self,
        name: str,
        batch_size: int = 32,
        n_epochs: int = 500,
        learning_rate: float = 0.025,
        criterion: 'torch.nn.modules.loss' = nn.MSELoss(),
        optimizer: 'torch.optim' = optim.SGD,
        random_state: int = 2020,
        verbose: bool = False,
    ):
        self.name = name
        self.batch_size = batch_size
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.criterion = criterion
        self._optimizer_callable = optimizer  # persist callable for later use
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.normlaize_data = False
        self.optimizer = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.val_dataloader = None

    def set_model(self,
                  model: 'torch.nn.Module',
                  normalize_data: bool = False,
                  set_optimizer_callable_params: bool = False) -> None:
        """Pass model architecture before training, optionally set optimizer params using model params."""
        self.model = model
        if self.verbose:
            print(f"model: {model}.")
        self.normlaize_data = normalize_data
        if self.verbose:
            if self.normalize_data:
                print("will normalize data before training")
            else:
                print("will not normalize data before training.")
        if set_optimizer_callable_params:
            self.optimizer = self._optimizer_callable(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = self._optimizer_callable(lr=self.learning_rate)

    def set_training_data(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]) -> None:
        """Pass training data to set model's DataLoader."""
        self._set_dataloader_from_data(X=X, y=y, mode="train")

    def set_test_data(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]) -> None:
        """Pass training data to set model's DataLoader."""
        self._set_dataloader_from_data(X=X, y=y, mode="test")

    def set_validation_data(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]) -> None:
        """Pass training data to set model's DataLoader."""
        self._set_dataloader_from_data(X=X, y=y, mode="val")

    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]):
        # maintain the same interface as the other models
        raise NotImplementedError

    def predict(self, X: Union[pd.DataFrame, np.ndarray]):
        # maintain the same interface as the other models
        raise NotImplementedError

    def _set_dataloader_from_data(self,
                                  X: Union[pd.DataFrame, np.ndarray],
                                  y: Union[pd.DataFrame, np.ndarray],
                                  mode: str = "val") -> None:
        """Use the data supplied to create train or validation DataLoaders."""
        if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
            if self.normlaize_data:
                if mode == "train":
                    X, y = self._normlaize_data(X=X, y=y)
                else:
                    X, y = self._apply_normalization(X=X, y=y)
            else:
                X, y = X.to_numpy(), y.to_numpy()
        else:
            if not isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
                raise ValueError("X and y must be both be pd.DataFrame objectrs or np.ndarray objects.")
        X, y = self._map_to_tensor(X, y)
        if mode == "train":
            dataloader = self._create_dataloader(X=X, y=y, batch_size=self.batch_size, shuffle=True)
            self.train_dataloader = dataloader
        elif mode == "test":
            dataloader = self._create_dataloader(X=X, y=y, batch_size=self.batch_size, shuffle=True)
            self.test_dataloader = dataloader
        else:
            dataloader = self._create_dataloader(X=X, y=y, batch_size=1, shuffle=False)
            self.val_dataloader = dataloader

    def _normlaize_data(self, X: 'pd.DataFrame', y: 'pd.DataFrame') -> Tuple[np.ndarray, np.ndarray]:
        """Normalize the data before training."""
        if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
            # map X and y to numpy arrays
            X = X.to_numpy()
            y = y.to_numpy()

        # normalize the training data
        X_mean = np.nanmean(X, axis=0)
        X_std = np.nanstd(X, axis=0)
        y_mean = np.nanmean(y, axis=0)
        y_std = np.nanstd(y, axis=0)

        # save the mean and std
        self.X_mean = X_mean
        self.X_std = X_std
        self.y_mean = y_mean
        self.y_std = y_std

        return self._apply_normalization(X, y)

    def _apply_normalization(self, X: Union['pd.DataFrame', np.ndarray],
                             y: Union['pd.DataFrame', np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Apply normalization learned during training for test or validation."""
        if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
            # map X and y to numpy arrays
            X = X.to_numpy()
            y = y.to_numpy()

        # replace missing values with the mean of that column
        X[np.isnan(X)] = np.take(self.X_mean, np.where(np.isnan(X))[1])
        y[np.isnan(y)] = np.take(self.y_mean, np.where(np.isnan(y))[1])

        # normalize the data before training
        X = (X - self.X_mean) / self.X_std
        y = (y - self.y_mean) / self.y_std

        return X, y

    def _map_to_tensor(self, X: 'np.ndarray', y: 'np.ndarray') -> Tuple['torch.Tensor', 'torch.Tensor']:
        """Convert the data to torch tensors."""
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)

        return X, y

    def _create_dataloader(self,
                           X: 'torch.Tensor',
                           y: 'torch.Tensor',
                           batch_size: int = 1,
                           shuffle: bool = True) -> 'torch.utils.data.TensorDataset':
        """Create the dataset and dataloader."""
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        if self.verbose:
            print(f"dataloader created with length {len(dataloader)}.")

        return dataloader

import sys
import warnings
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from otbench.benchmark.models.regression.base_model import BaseRegressionModel


class BasePyTorchRegressionModel(BaseRegressionModel):
    """Base class for PyTorch regression models."""

    def __init__(
        self,
        name: str,
        target_name: Union[str, List[str]],
        window_size: int = 1,  # default to single row
        batch_size: int = 32,
        n_epochs: int = 500,
        learning_rate: float = 0.025,
        criterion: 'torch.nn.modules.loss' = nn.MSELoss(),
        optimizer: 'torch.optim' = optim.SGD,
        random_state: int = 2020,
        verbose: bool = False,
        weight_decay: float = 0.0,
        patience: int = None,
        val_fraction: float = 0.0,
        use_scheduler: bool = False,
    ):
        super().__init__(name=name, target_name=target_name)
        self.window_size = window_size
        self.batch_size = batch_size
        self.random_state = random_state
        torch.manual_seed(self.random_state)
        self.n_epochs = n_epochs
        self.criterion = criterion
        self._optimizer_callable = optimizer  # persist callable for later use
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.verbose = verbose
        self.normalize_data = False
        self.optimizer = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.val_dataloader = None
        self._val_train_dataloader = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.patience = patience
        self.val_fraction = val_fraction
        self.use_scheduler = use_scheduler

    def set_model(self,
                  model: 'torch.nn.Module',
                  normalize_data: bool = False,
                  set_optimizer_callable_params: bool = False) -> None:
        """Pass model architecture before training, optionally set optimizer params using model params."""
        self.model = model
        if self.verbose:
            print(f"model: {model}.")
        self.normalize_data = normalize_data
        if self.verbose:
            if self.normalize_data:
                print("will normalize data before training")
            else:
                print("will not normalize data before training.")

        self.model.to(self.device)

        if set_optimizer_callable_params:
            self.optimizer = self._optimizer_callable(self.model.parameters(),
                                                      lr=self.learning_rate,
                                                      weight_decay=self.weight_decay)
        else:
            self.optimizer = self._optimizer_callable(lr=self.learning_rate, weight_decay=self.weight_decay)

    def set_training_data(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]) -> None:
        """Pass training data to set model's DataLoader."""
        self._set_dataloader_from_data(X=X, y=y, mode="train")

    def set_test_data(self,
                      X: Union[pd.DataFrame, np.ndarray],
                      y: Union[pd.DataFrame, np.ndarray, None] = None) -> None:
        """Pass test data to set model's DataLoader."""
        self._set_dataloader_from_data(X=X, y=y, mode="test")

    def set_validation_data(self,
                            X: Union[pd.DataFrame, np.ndarray],
                            y: Union[pd.DataFrame, np.ndarray, None] = None) -> None:
        """Pass validation data to set model's DataLoader."""
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
        if y is None and isinstance(X, pd.DataFrame):
            y = X.iloc[:, [0]]
        elif y is None and isinstance(X, np.ndarray):
            y = np.zeros((len(X), 1), dtype=X.dtype)
        elif y is None:
            raise ValueError("y must be supplied if X is not a pd.DataFrame  or np.ndarray object.")

        if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
            n_features = len(X.columns) // self.window_size
            X = X.to_numpy()
            y = y.to_numpy()
            X = X.reshape(-1, self.window_size, n_features)

            if self.normalize_data:
                if mode == "train":
                    X, y = self._normalize_data(X=X, y=y)
                else:
                    X, y = self._apply_normalization(X=X, y=y)
        elif isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
            X = X.copy()
            y = y.copy()
            if y.ndim == 1:
                y = y[:, np.newaxis]
            if X.ndim == 2:
                n_features = X.shape[1] // self.window_size
                X = X.reshape(-1, self.window_size, n_features)
            if self.normalize_data:
                if mode == "train":
                    X, y = self._normalize_data(X=X, y=y)
                else:
                    X, y = self._apply_normalization(X=X, y=y)
        else:
            raise ValueError("X and y must be both be pd.DataFrame objects or np.ndarray objects.")
        X, y = self._map_to_tensor(X, y)
        if mode == "train":
            dataloader = self._create_dataloader(X=X, y=y, batch_size=self.batch_size, shuffle=True)
            self.train_dataloader = dataloader
        elif mode == "test":
            dataloader = self._create_dataloader(X=X, y=y, batch_size=self.batch_size, shuffle=False)
            self.test_dataloader = dataloader
        elif mode == "val_train":
            dataloader = self._create_dataloader(X=X, y=y, batch_size=self.batch_size, shuffle=False)
            self._val_train_dataloader = dataloader
        else:
            dataloader = self._create_dataloader(X=X, y=y, batch_size=max(self.batch_size, 256), shuffle=False)
            self.val_dataloader = dataloader

    def _normalize_data(self, X: 'pd.DataFrame', y: 'pd.DataFrame') -> Tuple[np.ndarray, np.ndarray]:
        """Normalize the data before training."""
        # normalize the training data
        x_axes = (0, 1) if np.ndim(X) == 3 else 0
        X_mean = np.nanmean(X, axis=x_axes)
        X_std = np.nanstd(X, axis=x_axes)
        X_std = np.where(X_std == 0, sys.float_info.epsilon, X_std)
        y_mean = np.nanmean(y, axis=0)
        y_std = np.nanstd(y, axis=0)
        y_std = np.where(y_std == 0, sys.float_info.epsilon, y_std)

        # save the mean and std
        self.X_mean = X_mean
        self.X_std = X_std
        self.y_mean = y_mean
        self.y_std = y_std
        if hasattr(self.criterion, "set_normalization_stats"):
            self.criterion.set_normalization_stats(y_mean, y_std)

        return self._apply_normalization(X, y)

    def _apply_normalization(self, X: Union['pd.DataFrame', np.ndarray],
                             y: Union['pd.DataFrame', np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Apply normalization learned during training for test or validation."""
        # Replace NaN with training mean before normalizing.
        # X may be 3-D [samples, window, features] — broadcast X_mean over the
        # sample axis so that each (window, feature) position is filled correctly.
        if np.any(np.isnan(X)):
            X = np.where(np.isnan(X), self.X_mean, X)

        # y is 2-D [samples, output_size]; broadcast y_mean over sample axis.
        if np.any(np.isnan(y)):
            y = np.where(np.isnan(y), self.y_mean, y)

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
                           shuffle: bool = True) -> 'torch.utils.data.DataLoader':
        """Create the dataset and dataloader."""
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        if self.verbose:
            print(f"dataloader created with length {len(dataloader)}.")

        return dataloader

    def _split_data(self, X, y, val_fraction: float):
        """Split data into train/val, preserving temporal ordering."""
        n = len(X)
        n_val = max(1, int(n * val_fraction))
        n_train = n - n_val

        if isinstance(X, pd.DataFrame):
            X_tr, X_val = X.iloc[:n_train], X.iloc[n_train:]
        else:
            X_tr, X_val = X[:n_train], X[n_train:]

        if isinstance(y, (pd.DataFrame, pd.Series)):
            y_tr, y_val = y.iloc[:n_train], y.iloc[n_train:]
        else:
            y_tr, y_val = y[:n_train], y[n_train:]

        return X_tr, y_tr, X_val, y_val

    def _compute_val_loss(self):
        """Evaluate the model on the validation-for-training dataloader."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for X_batch, y_batch in self._val_train_dataloader:
                X_batch = X_batch.to(self.device).float()
                y_batch = y_batch.to(self.device).float()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                n_batches += 1
        self.model.train()
        return total_loss / max(n_batches, 1)

    def _create_scheduler(self, warmup_fraction: float = 0.1):
        """Create a warmup + cosine annealing LR scheduler."""
        warmup_epochs = max(1, int(self.n_epochs * warmup_fraction))
        main_epochs = self.n_epochs - warmup_epochs

        warmup = torch.optim.lr_scheduler.LinearLR(self.optimizer,
                                                   start_factor=0.01,
                                                   end_factor=1.0,
                                                   total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(main_epochs, 1), eta_min=1e-6)
        return torch.optim.lr_scheduler.SequentialLR(self.optimizer,
                                                     schedulers=[warmup, cosine],
                                                     milestones=[warmup_epochs])

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

        self._val_train_dataloader = None
        if self.val_fraction > 0 and self.patience is not None and len(X) >= 2:
            X_tr, y_tr, X_val, y_val = self._split_data(X, y, self.val_fraction)
            self.set_training_data(X=X_tr, y=y_tr)
            self._set_dataloader_from_data(X=X_val, y=y_val, mode="val_train")
        else:
            self.set_training_data(X=X, y=y)

        scheduler = self._create_scheduler() if self.use_scheduler else None
        best_val_loss = float('inf')
        best_state_dict = None
        epochs_no_improve = 0

        torch.manual_seed(self.random_state)
        for i in range(self.n_epochs):
            self.model.train()
            total_loss = 0.0
            for _, (X_batch, y_batch) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                X_batch, y_batch = X_batch.to(self.device).float(), y_batch.to(self.device).float()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(len(self.train_dataloader), 1)

            if scheduler is not None:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore",
                                            message=".*epoch parameter.*scheduler.step.*",
                                            category=UserWarning)
                    scheduler.step()

            if self._val_train_dataloader is not None and self.patience is not None:
                val_loss = self._compute_val_loss()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= self.patience:
                    break
            else:
                if self.verbose and self.n_epochs >= 10 and (i % (self.n_epochs // 10) == 0):
                    print(f"at epoch {i}. loss: {avg_loss:.6f}")

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

    def _run_prediction(self, X: Union['pd.DataFrame', np.ndarray], output_dim: int) -> np.ndarray:
        """Shared batched inference across validation/test data."""
        if len(X) == 0:
            if output_dim > 1:
                return np.empty((0, output_dim))
            return np.empty((0,))

        n_features = (len(X.columns) // self.window_size if isinstance(X, pd.DataFrame) else
                      (X.shape[-1] if X.ndim == 3 else X.shape[1] // self.window_size))
        if self.verbose:
            print(f"validation data contains {n_features} features.")
        self.set_validation_data(X=X, y=None)

        pred = []
        with torch.no_grad():
            self.model.eval()
            for _, (X_batch, _) in enumerate(self.val_dataloader):
                X_batch = X_batch.to(self.device).float()
                y_pred = self.model(X_batch).cpu().numpy()
                if self.normalize_data:
                    y_pred = y_pred * self.y_std + self.y_mean
                pred.append(y_pred)
            self.model.train()

        if not pred:
            if output_dim > 1:
                return np.empty((0, output_dim))
            return np.empty((0,))

        result = np.concatenate(pred, axis=0)
        if output_dim == 1:
            return result.ravel()
        return result

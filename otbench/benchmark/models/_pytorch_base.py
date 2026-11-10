import sys
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class RNN(nn.Module):
    """Configurable recurrent module supporting RNN, LSTM, and GRU cells."""

    _CELL_TYPES = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int, cell_type: str = 'rnn'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type

        if input_size <= 0:
            raise AssertionError("input size must be greater than 0")
        if cell_type not in self._CELL_TYPES:
            raise ValueError(f"Unsupported cell_type '{cell_type}'. Must be one of {list(self._CELL_TYPES.keys())}")

        cell_cls = self._CELL_TYPES[cell_type]
        self.rnn = cell_cls(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        if self.cell_type == 'lstm':
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])


class BasePyTorchMixin:
    """Shared PyTorch training, normalization, DataLoader, and inference lifecycle primitives."""

    def _init_pytorch_state(
        self,
        batch_size: int = 32,
        n_epochs: int = 500,
        learning_rate: float = 0.025,
        criterion: Optional['torch.nn.modules.loss'] = None,
        optimizer: 'torch.optim' = optim.SGD,
        random_state: int = 2020,
        verbose: bool = False,
        weight_decay: float = 0.0,
        patience: Union[int, None] = None,
        val_fraction: float = 0.0,
        use_scheduler: bool = False,
    ) -> None:
        """Initialize shared PyTorch hyperparameters, device, and DataLoader handles."""
        self.batch_size = batch_size
        self.random_state = random_state
        torch.manual_seed(self.random_state)
        self.n_epochs = n_epochs
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        self._optimizer_callable = optimizer
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

    @staticmethod
    def _ensure_y_for_dataloader(
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray, None],
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Populate a placeholder y when None is supplied for inference DataLoaders."""
        if y is None and isinstance(X, pd.DataFrame):
            return X.iloc[:, [0]]
        if y is None and isinstance(X, np.ndarray):
            return np.zeros((len(X), 1), dtype=X.dtype)
        if y is None:
            raise ValueError("y must be supplied if X is not a pd.DataFrame  or np.ndarray object.")
        return y

    def _finalize_dataloader(self, X: np.ndarray, y: np.ndarray, mode: str) -> None:
        """Normalize if configured, convert to tensors, and bind the DataLoader for mode."""
        if self.normalize_data:
            if mode == "train":
                X, y = self._normalize_data(X=X, y=y)
            else:
                X, y = self._apply_normalization(X=X, y=y)

        X_tensor, y_tensor = self._map_to_tensor(X, y)
        if mode == "train":
            self.train_dataloader = self._create_dataloader(X=X_tensor,
                                                            y=y_tensor,
                                                            batch_size=self.batch_size,
                                                            shuffle=True)
        elif mode == "test":
            self.test_dataloader = self._create_dataloader(X=X_tensor,
                                                           y=y_tensor,
                                                           batch_size=self.batch_size,
                                                           shuffle=False)
        elif mode == "val_train":
            self._val_train_dataloader = self._create_dataloader(X=X_tensor,
                                                                 y=y_tensor,
                                                                 batch_size=self.batch_size,
                                                                 shuffle=False)
        else:
            self.val_dataloader = self._create_dataloader(X=X_tensor,
                                                          y=y_tensor,
                                                          batch_size=max(self.batch_size, 256),
                                                          shuffle=False)

    def _normalize_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize the data before training."""
        x_axes = (0, 1) if np.ndim(X) == 3 else 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            X_mean = np.nanmean(X, axis=x_axes)
            X_std = np.nanstd(X, axis=x_axes)
            y_mean = np.nanmean(y, axis=0)
            y_std = np.nanstd(y, axis=0)
        X_mean = np.where(np.isnan(X_mean), 0.0, X_mean)
        X_std = np.where((X_std == 0) | np.isnan(X_std), sys.float_info.epsilon, X_std)
        y_mean = np.where(np.isnan(y_mean), 0.0, y_mean)
        y_std = np.where((y_std == 0) | np.isnan(y_std), sys.float_info.epsilon, y_std)

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
        if np.any(np.isnan(X)):
            X = np.where(np.isnan(X), self.X_mean, X)
        if np.any(np.isnan(y)):
            y = np.where(np.isnan(y), self.y_mean, y)
        X = (X - self.X_mean) / self.X_std
        y = (y - self.y_mean) / self.y_std
        return X, y

    def _map_to_tensor(self, X: 'np.ndarray', y: 'np.ndarray') -> Tuple['torch.Tensor', 'torch.Tensor']:
        """Convert the data to torch tensors."""
        return torch.from_numpy(X), torch.from_numpy(y)

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
        X_tr, X_val = (X.iloc[:n_train], X.iloc[n_train:]) if isinstance(X, pd.DataFrame) else (X[:n_train],
                                                                                                X[n_train:])
        y_tr, y_val = (y.iloc[:n_train],
                       y.iloc[n_train:]) if isinstance(y, (pd.DataFrame, pd.Series)) else (y[:n_train], y[n_train:])
        return X_tr, y_tr, X_val, y_val

    def _compute_loss(self, outputs: torch.Tensor, y_batch: torch.Tensor, X_batch: torch.Tensor) -> torch.Tensor:
        """Evaluate criterion on a batch."""
        return self.criterion(outputs, y_batch)

    def _compute_val_loss(self) -> float:
        """Evaluate the model on the validation-for-training dataloader using sample-weighted mean loss."""
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        with torch.no_grad():
            for X_batch, y_batch in self._val_train_dataloader:
                X_batch = X_batch.to(self.device).float()
                y_batch = y_batch.to(self.device).float()
                outputs = self.model(X_batch)
                loss = self._compute_loss(outputs, y_batch, X_batch)
                batch_len = int(X_batch.size(0))
                total_loss += float(loss.item()) * batch_len
                n_samples += batch_len
        self.model.train()
        return total_loss / max(n_samples, 1)

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

    def _prepare_train_val_dataloaders(self, X, y) -> None:
        """Configure train_dataloader and optional _val_train_dataloader for validation checkpointing/early stopping."""
        self._val_train_dataloader = None
        if self.val_fraction > 0 and len(X) >= 2:
            X_tr, y_tr, X_val, y_val = self._split_data(X, y, self.val_fraction)
            self.set_training_data(X=X_tr, y=y_tr)
            self._set_dataloader_from_data(X=X_val, y=y_val, mode="val_train")
        else:
            self.set_training_data(X=X, y=y)

    def _execute_epoch_loop(self) -> None:
        """Run epochs with gradient clipping, optional LR scheduling, and early stopping."""
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
                loss = self._compute_loss(outputs, y_batch, X_batch)
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

            if self._val_train_dataloader is not None:
                val_loss = self._compute_val_loss()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if self.patience is not None and epochs_no_improve >= self.patience:
                    break
            else:
                if self.verbose and self.n_epochs >= 10 and (i % (self.n_epochs // 10) == 0):
                    print(f"at epoch {i}. loss: {avg_loss:.6f}")

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

    def _collect_batched_predictions(self, output_dim: int, empty_1d_for_scalar: bool = False) -> np.ndarray:
        """Evaluate self.model across self.val_dataloader and un-normalize predictions."""
        preds = []
        with torch.no_grad():
            self.model.eval()
            for _, (X_batch, _) in enumerate(self.val_dataloader):
                X_batch = X_batch.to(self.device).float()
                y_pred = self.model(X_batch).cpu().numpy()
                if self.normalize_data:
                    y_pred = y_pred * self.y_std + self.y_mean
                preds.append(y_pred)

        if not preds:
            if output_dim > 1 or not empty_1d_for_scalar:
                return np.empty((0, output_dim))
            return np.empty((0,))

        result = np.concatenate(preds, axis=0)
        return result.ravel() if output_dim == 1 else result

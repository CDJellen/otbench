import sys
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from otbench.benchmark.models.forecasting.base_model import BaseForecastingModel


class BasePyTorchForecastingModel(BaseForecastingModel):
    """A the base class for PyTorch models."""

    def __init__(
        self,
        name: str,
        target_name: Union[str, List[str]],
        window_size: int,
        forecast_horizon: int,
        batch_size: int = 32,
        n_epochs: int = 500,
        learning_rate: float = 0.025,
        criterion: 'torch.nn.modules.loss' = nn.MSELoss(),
        optimizer: 'torch.optim' = optim.SGD,
        random_state: int = 2020,
        verbose: bool = False,
        **kwargs
    ):
        # Pass kwargs (predict_residuals, use_log10, etc.) to BaseForecastingModel
        super().__init__(name=name, target_name=target_name, window_size=window_size, forecast_horizon=forecast_horizon, **kwargs)
        self.batch_size = batch_size
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.criterion = criterion
        self._optimizer_callable = optimizer  # persist callable for later use
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.normalize_data = False
        self.optimizer = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.val_dataloader = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            self.optimizer = self._optimizer_callable(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = self._optimizer_callable(lr=self.learning_rate)

    def set_training_data(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]) -> None:
        """Pass training data to set model's DataLoader."""
        self._set_dataloader_from_data(X=X, y=y, mode="train")

    def set_test_data(self,
                      X: Union[pd.DataFrame, np.ndarray],
                      y: Union[pd.DataFrame, np.ndarray, None] = None) -> None:
        """Pass training data to set model's DataLoader."""
        self._set_dataloader_from_data(X=X, y=y, mode="test")

    def set_validation_data(self,
                            X: Union[pd.DataFrame, np.ndarray],
                            y: Union[pd.DataFrame, np.ndarray, None] = None) -> None:
        """Pass training data to set model's DataLoader."""
        self._set_dataloader_from_data(X=X, y=y, mode="val")

    def _train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]):
        # Implementation hook required by BaseForecastingModel
        raise NotImplementedError

    def _predict(self, X: Union[pd.DataFrame, np.ndarray]):
        # Implementation hook required by BaseForecastingModel
        raise NotImplementedError

    def _set_dataloader_from_data(self,
                                  X: Union[pd.DataFrame, np.ndarray],
                                  y: Union[pd.DataFrame, np.ndarray],
                                  mode: str = "val") -> None:
        """Use the data supplied to create train or validation DataLoaders."""
        if y is None and isinstance(X, pd.DataFrame):
            y = X.iloc[:, [0]]
        elif y is None and isinstance(X, np.ndarray):
            y = X[:, [0]]
        elif y is None:
            raise ValueError("y must be supplied if X is not a pd.DataFrame  or np.ndarray object.")

        if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
            n_cols = len(X.columns)
            X = X.to_numpy()
            y = y.to_numpy()
            # Temporal mode: columns were built by lagging *all* features, so
            # len(columns) == window_size × n_features_per_step exactly.
            # Flat mode: selective lag_features produces a mixed contemporaneous +
            # lag column layout that doesn't divide evenly; treat the full feature
            # vector as a single-step sequence (seq_len=1) instead.
            if self.window_size > 1 and n_cols % self.window_size == 0:
                n_features = n_cols // self.window_size
                X = X.reshape(-1, self.window_size, n_features)
                # _add_lags produces columns ordered [feat(t-0), feat(t-1), ..., feat(t-W+1)],
                # so after reshape step-0 = newest and step-(W-1) = oldest.  Reverse the time
                # axis so the sequence is chronological: step-0 = oldest, step-(W-1) = current.
                # This ensures the RNN final hidden state and Transformer output[:, -1, :] both
                # encode the most recent observation rather than the oldest one.
                X = np.ascontiguousarray(X[:, ::-1, :])
            else:
                X = X.reshape(-1, 1, n_cols)

            if self.normalize_data:
                if mode == "train":
                    X, y = self._normalize_data(X=X, y=y)
                else:
                    X, y = self._apply_normalization(X=X, y=y)
        else:
            if not isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
                raise ValueError("X and y must be both be pd.DataFrame objects or np.ndarray objects.")
        
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

    def _normalize_data(self, X: 'pd.DataFrame', y: 'pd.DataFrame') -> Tuple[np.ndarray, np.ndarray]:
        """Normalize the data before training."""
        # X is 3-D [samples, window, features]: compute one mean/std per feature,
        # averaged over both the sample and time-step axes.  Shape: (n_features,).
        X_mean = np.nanmean(X, axis=(0, 1))
        X_std = np.nanstd(X, axis=(0, 1)) + sys.float_info.epsilon
        # y is 2-D [samples, n_targets]: compute per-target mean/std over samples.
        # Using axis=0 preserves the target dimension → shape (n_targets,).
        y_mean = np.nanmean(y, axis=0)
        y_std = np.nanstd(y, axis=0) + sys.float_info.epsilon

        # save the mean and std
        self.X_mean = X_mean
        self.X_std = X_std
        self.y_mean = y_mean
        self.y_std = y_std

        return self._apply_normalization(X, y)

    def _apply_normalization(self, X: Union['pd.DataFrame', np.ndarray],
                             y: Union['pd.DataFrame', np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Apply normalization learned during training for test or validation."""
        # Replace NaN with training mean before normalizing.
        # X may be 3-D [samples, window, features] — broadcast X_mean (shape [features])
        # over the sample and window axes so each feature gets the correct fill value.
        if np.any(np.isnan(X)):
            X = np.where(np.isnan(X), self.X_mean, X)

        # y is 2-D [samples, n_targets] — broadcast y_mean (shape [n_targets]).
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
                           shuffle: bool = True) -> 'torch.utils.data.TensorDataset':
        """Create the dataset and dataloader."""
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        if self.verbose:
            print(f"dataloader created with length {len(dataloader)}.")

        return dataloader
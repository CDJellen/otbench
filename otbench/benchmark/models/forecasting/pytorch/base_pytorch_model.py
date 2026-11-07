import math
import sys
import warnings
from typing import Optional, Sequence, Tuple, Union, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from otbench.benchmark.models.forecasting.base_model import BaseForecastingModel

_LN10 = math.log(10.0)


class AOPhysicsLoss(nn.Module):
    """Differentiable Adaptive Optics profile + optical-moment composite loss.

    Combines standard per-layer log10(J_i) Mean Squared Error with numerically stable
    ``torch.logsumexp`` penalties on the 0th optical turbulence moment (total column
    turbulence J_total -> Fried parameter r_0 / integrated seeing) and the 5/3rd
    altitude-weighted turbulence moment (effective altitude H_eff -> isoplanatic angle theta_0):

        L_AO(y_hat, y) = MSE(y_hat, y)
                       + lambda_r0 * MSE(log10 sum_i 10^{y_hat_i}, log10 sum_i 10^{y_i})
                       + lambda_theta0 * MSE(log10 sum_i h_tilde_i^{5/3} 10^{y_hat_i},
                                             log10 sum_i h_tilde_i^{5/3} 10^{y_i})

    When attached to a normalized PyTorch model, ``set_normalization_stats(y_mean, y_std)``
    is called automatically so moment integrals are evaluated in physical log10(J_i) space.
    When ``predict_residuals=True``, ``y_base`` (the current persistence profile log10(J_i(t)))
    is added to the un-normalized residual predictions and targets before evaluating moments.
    """

    accepts_y_base: bool = True

    def __init__(
        self,
        heights: Sequence[float] = (0.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0),
        lambda_r0: float = 0.15,
        lambda_theta0: float = 0.15,
        min_height_m: float = 100.0,
    ) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_r0 = float(lambda_r0)
        self.lambda_theta0 = float(lambda_theta0)

        h_arr = np.maximum(np.asarray(heights, dtype=np.float32), float(min_height_m))
        h_tilde = h_arr / np.max(h_arr)
        log10_h53 = (5.0 / 3.0) * np.log10(h_tilde)
        self.register_buffer("log10_h53", torch.tensor(log10_h53, dtype=torch.float32))
        self.register_buffer("y_mean", None)
        self.register_buffer("y_std", None)

    def set_normalization_stats(self, y_mean: np.ndarray, y_std: np.ndarray) -> None:
        """Register target normalization statistics so optical moments are evaluated in physical log10(J_i) space."""
        self.y_mean = torch.as_tensor(y_mean, dtype=torch.float32)
        self.y_std = torch.as_tensor(y_std, dtype=torch.float32)

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        y_base: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss_profile = self.mse(y_pred, y_true)
        if y_pred.ndim < 2 or y_pred.shape[-1] <= 1 or (self.lambda_r0 == 0.0 and self.lambda_theta0 == 0.0):
            return loss_profile

        if self.y_mean is not None and self.y_std is not None:
            mean = self.y_mean.to(device=y_pred.device, dtype=y_pred.dtype)
            std = self.y_std.to(device=y_pred.device, dtype=y_pred.dtype)
            yp_phys = y_pred * std + mean
            yt_phys = y_true * std + mean
        else:
            yp_phys = y_pred
            yt_phys = y_true

        if y_base is not None:
            base = y_base.to(device=y_pred.device, dtype=y_pred.dtype)
            yp_phys = yp_phys + base
            yt_phys = yt_phys + base

        yp_clamped = torch.clamp(yp_phys, min=-20.0, max=-5.0)
        yt_clamped = torch.clamp(yt_phys, min=-20.0, max=-5.0)

        total_loss = loss_profile
        if self.lambda_r0 > 0.0:
            log10_j_pred = torch.logsumexp(yp_clamped * _LN10, dim=-1) / _LN10
            log10_j_true = torch.logsumexp(yt_clamped * _LN10, dim=-1) / _LN10
            total_loss = total_loss + self.lambda_r0 * self.mse(log10_j_pred, log10_j_true)

        if self.lambda_theta0 > 0.0 and yp_clamped.shape[-1] == self.log10_h53.numel():
            h_shift = self.log10_h53.to(device=y_pred.device, dtype=y_pred.dtype)
            log10_m53_pred = torch.logsumexp((yp_clamped + h_shift) * _LN10, dim=-1) / _LN10
            log10_m53_true = torch.logsumexp((yt_clamped + h_shift) * _LN10, dim=-1) / _LN10
            total_loss = total_loss + self.lambda_theta0 * self.mse(log10_m53_pred, log10_m53_true)

        return total_loss


class BasePyTorchForecastingModel(BaseForecastingModel):
    """Base class for PyTorch forecasting models."""

    def __init__(self,
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
                 weight_decay: float = 0.0,
                 patience: int = None,
                 val_fraction: float = 0.0,
                 use_scheduler: bool = False,
                 **kwargs):
        # Pass kwargs (predict_residuals, use_log10, etc.) to BaseForecastingModel
        super().__init__(name=name,
                         target_name=target_name,
                         window_size=window_size,
                         forecast_horizon=forecast_horizon,
                         **kwargs)
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
        self._val_train_dataloader = None  # validation dataloader for early stopping
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Early stopping and scheduling
        self.patience = patience
        self.val_fraction = val_fraction
        self.use_scheduler = use_scheduler
        self._target_feature_indices: Optional[List[int]] = None
        self._target_time_index: int = -1

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
            y = np.zeros((len(X), 1), dtype=X.dtype)
        elif y is None:
            raise ValueError("y must be supplied if X is not a pd.DataFrame  or np.ndarray object.")

        if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
            n_cols = len(X.columns)
            col_names = list(X.columns)
            targets = self.target_name if isinstance(self.target_name, list) else [self.target_name]
            p_cols = self._persistence_cols
            if p_cols is None:
                try:
                    p_cols = self._find_persistence_cols(X, pd.DataFrame(columns=targets))
                except ValueError:
                    p_cols = []
            X = X.to_numpy().copy()
            y = y.to_numpy().copy()

            # When use_log10 is active, transform linear target-history columns in X
            # into log10 space so autoregressive neural features share the target domain.
            if self.use_log10 and len(X) > 0:
                target_col_idx = [
                    idx for idx, c in enumerate(col_names)
                    if any(c == t or c.startswith(f"{t} (t-") or c.startswith(f"{t}_t-") for t in targets)
                ]
                if target_col_idx:
                    sub = X[:, target_col_idx]
                    valid = sub[np.isfinite(sub)]
                    if len(valid) > 0 and np.min(valid) >= 0.0 and np.median(valid) < 1e-5:
                        X[:, target_col_idx] = np.log10(np.maximum(sub, 1e-19))

            # Temporal mode: columns were built by lagging *all* features, so
            # len(columns) == window_size × n_features_per_step exactly.
            # Flat mode: selective lag_features produces a mixed contemporaneous +
            # lag column layout that doesn't divide evenly; treat the full feature
            # vector as a single-step sequence (seq_len=1) instead.
            if self.window_size > 1 and n_cols % self.window_size == 0:
                n_features = n_cols // self.window_size
                if p_cols:
                    first_c_idx = col_names.index(p_cols[0])
                    self._target_time_index = (self.window_size - 1) - (first_c_idx // n_features)
                    self._target_feature_indices = [col_names.index(c) % n_features for c in p_cols]
                X = X.reshape(-1, self.window_size, n_features)
                # _add_lags produces columns ordered [feat(t-0), feat(t-1), ..., feat(t-W+1)],
                # so after reshape step-0 = newest and step-(W-1) = oldest.  Reverse the time
                # axis so the sequence is chronological: step-0 = oldest, step-(W-1) = current.
                # This ensures the RNN final hidden state and Transformer output[:, -1, :] both
                # encode the most recent observation rather than the oldest one.
                X = np.ascontiguousarray(X[:, ::-1, :])
            else:
                if p_cols:
                    self._target_time_index = 0
                    self._target_feature_indices = [col_names.index(c) for c in p_cols]
                X = X.reshape(-1, 1, n_cols)

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
                n_cols = X.shape[1]
                if self.window_size > 1 and n_cols % self.window_size == 0:
                    n_features = n_cols // self.window_size
                    X = np.ascontiguousarray(X.reshape(-1, self.window_size, n_features)[:, ::-1, :])
                else:
                    X = X.reshape(-1, 1, n_cols)
            if X.ndim == 3:
                target_idx = self._persistence_indices_3d or self._find_persistence_indices_3d(X.shape[2], strict=False)
                if target_idx:
                    self._target_time_index = -1
                    self._target_feature_indices = list(target_idx)
                    if self.use_log10 and len(X) > 0:
                        sub = X[:, :, target_idx]
                        valid = sub[np.isfinite(sub)]
                        if len(valid) > 0 and np.min(valid) >= 0.0 and np.median(valid) < 1e-5:
                            X[:, :, target_idx] = np.log10(np.maximum(sub, 1e-19))
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
            # Validation dataloader for early stopping during training:
            # uses training batch_size for efficiency, no shuffle for reproducibility.
            dataloader = self._create_dataloader(X=X, y=y, batch_size=self.batch_size, shuffle=False)
            self._val_train_dataloader = dataloader
        else:
            dataloader = self._create_dataloader(X=X, y=y, batch_size=max(self.batch_size, 256), shuffle=False)
            self.val_dataloader = dataloader

    def _normalize_data(self, X: 'pd.DataFrame', y: 'pd.DataFrame') -> Tuple[np.ndarray, np.ndarray]:
        """Normalize the data before training."""
        # X is 3-D [samples, window, features]: compute one mean/std per feature,
        # averaged over both the sample and time-step axes.  Shape: (n_features,).
        X_mean = np.nanmean(X, axis=(0, 1))
        X_std = np.nanstd(X, axis=(0, 1))
        X_std = np.where(X_std == 0, sys.float_info.epsilon, X_std)
        # y is 2-D [samples, n_targets]: compute per-target mean/std over samples.
        # Using axis=0 preserves the target dimension → shape (n_targets,).
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
                           shuffle: bool = True) -> 'torch.utils.data.DataLoader':
        """Create the dataset and dataloader."""
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        if self.verbose:
            print(f"dataloader created with length {len(dataloader)}.")

        return dataloader

    def _split_data(self, X, y, val_fraction: float):
        """Split data into train/val, preserving temporal ordering.

        Takes the last ``val_fraction`` of samples as validation to prevent
        data leakage from future observations into the training set.
        """
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

    def _extract_y_base(self, X_batch: torch.Tensor) -> Optional[torch.Tensor]:
        """Reconstruct physical log10(J_i(t)) persistence baseline from X_batch when predicting residuals."""
        if not self.predict_residuals or not self._target_feature_indices or X_batch.ndim < 3:
            return None
        t_idx = getattr(self, "_target_time_index", -1)
        f_idx = self._target_feature_indices
        x_base = X_batch[:, t_idx, f_idx]
        if self.normalize_data and getattr(self, "X_mean", None) is not None and getattr(self, "X_std", None) is not None:
            mean_arr = np.asarray(self.X_mean)
            std_arr = np.asarray(self.X_std)
            mean_t = torch.as_tensor(
                mean_arr[f_idx] if mean_arr.ndim == 1 else mean_arr[t_idx, f_idx],
                device=X_batch.device,
                dtype=X_batch.dtype,
            )
            std_t = torch.as_tensor(
                std_arr[f_idx] if std_arr.ndim == 1 else std_arr[t_idx, f_idx],
                device=X_batch.device,
                dtype=X_batch.dtype,
            )
            return x_base * std_t + mean_t
        return x_base

    def _compute_loss(self, outputs: torch.Tensor, y_batch: torch.Tensor, X_batch: torch.Tensor) -> torch.Tensor:
        """Evaluate criterion, supplying the physical persistence baseline when AOPhysicsLoss is used with residuals."""
        if getattr(self.criterion, "accepts_y_base", False) and self.predict_residuals:
            y_base = self._extract_y_base(X_batch)
            return self.criterion(outputs, y_batch, y_base=y_base)
        return self.criterion(outputs, y_batch)

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
                loss = self._compute_loss(outputs, y_batch, X_batch)
                total_loss += loss.item()
                n_batches += 1
        self.model.train()
        return total_loss / max(n_batches, 1)

    def _create_scheduler(self, warmup_fraction: float = 0.1):
        """Create a warmup + cosine annealing LR scheduler.

        Linear warmup over the first ``warmup_fraction`` of epochs,
        then cosine decay to ``eta_min`` for the remainder.
        """
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

    def _run_training_loop(self, X: Union['pd.DataFrame', np.ndarray], y: Union['pd.DataFrame', np.ndarray],
                           input_size: int) -> None:
        """Shared training loop supporting temporal validation splits, LR warmup/decay,
        gradient clipping, and early stopping with best-checkpoint restoration.
        """
        if len(X) == 0:
            return

        if isinstance(X, np.ndarray) and X.ndim == 3:
            n_features_in_data = X.shape[2]
            n_cols = X.shape[1] * X.shape[2]
            if isinstance(y, (pd.DataFrame, pd.Series)):
                y = y.to_numpy()
        else:
            n_cols = X.shape[1] if isinstance(X, np.ndarray) else len(X.columns)
            if self.window_size > 1 and n_cols % self.window_size == 0:
                n_features_in_data = n_cols // self.window_size
            else:
                n_features_in_data = n_cols

        if n_features_in_data != input_size:
            raise ValueError(f"Dimension Mismatch: Model initialized with input_size={input_size}, "
                             f"but training data has {n_features_in_data} features per timestep "
                             f"(Total columns: {n_cols}, Window: {self.window_size}).")

        if self.verbose:
            mode = "temporal" if (self.window_size > 1 and n_cols % self.window_size == 0) else "flat"
            print(f"training data: {mode} mode, {n_features_in_data} features per timestep, "
                  f"{self.window_size} timestep(s) per sample.")

        self._val_train_dataloader = None
        if self.val_fraction > 0 and self.patience is not None and len(X) >= 2:
            X_tr, y_tr, X_val, y_val = self._split_data(X, y, self.val_fraction)
            self.set_training_data(X=X_tr, y=y_tr)
            self._set_dataloader_from_data(X=X_val, y=y_val, mode="val_train")
            if self.verbose:
                print(f"  Train/Val split: {len(X_tr)}/{len(X_val)} samples")
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

            if self._val_train_dataloader is not None and self.patience is not None:
                val_loss = self._compute_val_loss()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if self.verbose and self.n_epochs >= 10 and (i % (self.n_epochs // 10) == 0):
                    lr = self.optimizer.param_groups[0]['lr']
                    print(f"  Epoch {i}: Train={avg_loss:.6f} | Val={val_loss:.6f} | LR={lr:.2e}")

                if epochs_no_improve >= self.patience:
                    if self.verbose:
                        print(f"  Early stopping at epoch {i}. Best val loss: {best_val_loss:.6f}")
                    break
            else:
                if self.verbose and self.n_epochs >= 10 and (i % (self.n_epochs // 10) == 0):
                    print(f"  at epoch {i}. loss: {avg_loss:.6f}")

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

    def _run_prediction(self, X: Union['pd.DataFrame', np.ndarray], output_dim: int) -> np.ndarray:
        """Shared batched inference across validation/test data."""
        if len(X) == 0:
            return np.empty((0, output_dim))

        if isinstance(X, np.ndarray) and X.ndim == 3:
            n_features = X.shape[2]
            n_cols = X.shape[1] * X.shape[2]
        else:
            n_cols = X.shape[1] if isinstance(X, np.ndarray) else len(X.columns)
            n_features = n_cols // self.window_size if (self.window_size > 1
                                                        and n_cols % self.window_size == 0) else n_cols
        if self.verbose:
            mode = "temporal" if (self.window_size > 1 and n_cols % self.window_size == 0) else "flat"
            print(f"validation data: {mode} mode, {n_features} features per timestep.")

        self.set_validation_data(X=X, y=None)
        preds = []

        with torch.no_grad():
            self.model.eval()
            for _, (X_batch, _) in enumerate(self.val_dataloader):
                X_batch = X_batch.to(self.device).float()
                y_pred = self.model(X_batch).cpu().numpy()
                if self.normalize_data:
                    y_pred = y_pred * self.y_std + self.y_mean
                preds.append(y_pred)
            self.model.train()

        if not preds:
            return np.empty((0, output_dim))

        result = np.concatenate(preds, axis=0)
        if output_dim == 1:
            return result.ravel()
        return result

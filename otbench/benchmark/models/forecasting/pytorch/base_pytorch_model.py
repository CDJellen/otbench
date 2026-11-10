import math
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from otbench.benchmark.models._pytorch_base import BasePyTorchMixin
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
        jensen_correction: bool = True,
    ) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_r0 = float(lambda_r0)
        self.lambda_theta0 = float(lambda_theta0)
        self.jensen_correction = bool(jensen_correction)

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

        # Second-order LogSumExp Jensen curvature correction:
        # E[log10 sum_k w_k 10^{y_k} | X] - log10 sum_k w_k 10^{E[y_k|X]}
        #   ~= (ln 10 / 2) * sum_k p_k * (1 - p_k) * sigma_k^2
        var_k = None
        if self.jensen_correction:
            var_k = torch.clamp(((yt_clamped - yp_clamped).detach()**2).mean(dim=0, keepdim=True), min=0.0, max=2.0)

        total_loss = loss_profile
        if self.lambda_r0 > 0.0:
            log10_j_pred = torch.logsumexp(yp_clamped * _LN10, dim=-1) / _LN10
            log10_j_true = torch.logsumexp(yt_clamped * _LN10, dim=-1) / _LN10
            if var_k is not None:
                p_r0 = torch.softmax(yp_clamped.detach() * _LN10, dim=-1)
                jensen_r0 = 0.5 * _LN10 * torch.sum(p_r0 * (1.0 - p_r0) * var_k, dim=-1)
                log10_j_pred = log10_j_pred + jensen_r0
            total_loss = total_loss + self.lambda_r0 * self.mse(log10_j_pred, log10_j_true)

        if self.lambda_theta0 > 0.0:
            if yp_clamped.shape[-1] != self.log10_h53.numel():
                raise ValueError(f"AOPhysicsLoss expected {self.log10_h53.numel()} profile layers matching heights, "
                                 f"got {yp_clamped.shape[-1]}.")
            h_shift = self.log10_h53.to(device=y_pred.device, dtype=y_pred.dtype)
            log10_m53_pred = torch.logsumexp((yp_clamped + h_shift) * _LN10, dim=-1) / _LN10
            log10_m53_true = torch.logsumexp((yt_clamped + h_shift) * _LN10, dim=-1) / _LN10
            if var_k is not None:
                p_theta0 = torch.softmax((yp_clamped.detach() + h_shift) * _LN10, dim=-1)
                jensen_theta0 = 0.5 * _LN10 * torch.sum(p_theta0 * (1.0 - p_theta0) * var_k, dim=-1)
                log10_m53_pred = log10_m53_pred + jensen_theta0
            total_loss = total_loss + self.lambda_theta0 * self.mse(log10_m53_pred, log10_m53_true)

        return total_loss


class BasePyTorchForecastingModel(BasePyTorchMixin, BaseForecastingModel):
    """Base class for PyTorch forecasting models."""

    def __init__(self,
                 name: str,
                 target_name: Union[str, List[str]],
                 window_size: int,
                 forecast_horizon: int,
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
                 use_physics_loss: bool = False,
                 heights: Optional[Sequence[float]] = None,
                 lambda_r0: float = 0.15,
                 lambda_theta0: float = 0.15,
                 jensen_correction: bool = True,
                 **kwargs):
        BaseForecastingModel.__init__(self,
                                      name=name,
                                      target_name=target_name,
                                      window_size=window_size,
                                      forecast_horizon=forecast_horizon,
                                      **kwargs)
        if use_physics_loss and (criterion is None or type(criterion) is nn.MSELoss):
            criterion = AOPhysicsLoss(
                heights=heights if heights is not None else (0.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0),
                lambda_r0=lambda_r0,
                lambda_theta0=lambda_theta0,
                jensen_correction=jensen_correction,
            )
        self.use_physics_loss = bool(use_physics_loss or isinstance(criterion, AOPhysicsLoss))
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
        self._target_feature_indices: Optional[List[int]] = None
        self._target_time_index: int = -1

    def _train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]):
        raise NotImplementedError

    def _predict(self, X: Union[pd.DataFrame, np.ndarray]):
        raise NotImplementedError

    def _set_dataloader_from_data(self,
                                  X: Union[pd.DataFrame, np.ndarray],
                                  y: Union[pd.DataFrame, np.ndarray],
                                  mode: str = "val") -> None:
        """Use the data supplied to create train or validation DataLoaders."""
        y = self._ensure_y_for_dataloader(X, y)

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

            if self.use_log10 and len(X) > 0:
                target_col_idx = [
                    idx for idx, c in enumerate(col_names)
                    if any(c == t or c.startswith(f"{t} (t-") or c.startswith(f"{t}_t-") for t in targets)
                ]
                if target_col_idx:
                    X[:, target_col_idx] = self._maybe_log10_array(X[:, target_col_idx])

            if self.window_size > 1 and n_cols % self.window_size == 0:
                n_features = n_cols // self.window_size
                if p_cols:
                    first_c_idx = col_names.index(p_cols[0])
                    self._target_time_index = (self.window_size - 1) - (first_c_idx // n_features)
                    self._target_feature_indices = [col_names.index(c) % n_features for c in p_cols]
                X = np.ascontiguousarray(X.reshape(-1, self.window_size, n_features)[:, ::-1, :])
            else:
                if p_cols:
                    self._target_time_index = 0
                    self._target_feature_indices = [col_names.index(c) for c in p_cols]
                X = X.reshape(-1, 1, n_cols)
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
                        X[:, :, target_idx] = self._maybe_log10_array(X[:, :, target_idx])
        else:
            raise ValueError("X and y must be both be pd.DataFrame objects or np.ndarray objects.")

        self._finalize_dataloader(X, y, mode=mode)

    def _extract_y_base(self, X_batch: torch.Tensor) -> Optional[torch.Tensor]:
        """Reconstruct physical log10(J_i(t)) persistence baseline from X_batch when predicting residuals."""
        if not self.predict_residuals or not self._target_feature_indices or X_batch.ndim < 3:
            return None
        t_idx = getattr(self, "_target_time_index", -1)
        f_idx = self._target_feature_indices
        x_base = X_batch[:, t_idx, f_idx]
        if self.normalize_data and getattr(self, "X_mean", None) is not None and getattr(self, "X_std",
                                                                                         None) is not None:
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
            n_features_in_data = n_cols // self.window_size if (self.window_size > 1
                                                                and n_cols % self.window_size == 0) else n_cols

        if n_features_in_data != input_size:
            raise ValueError(f"Dimension Mismatch: Model initialized with input_size={input_size}, "
                             f"but training data has {n_features_in_data} features per timestep "
                             f"(Total columns: {n_cols}, Window: {self.window_size}).")

        if self.verbose:
            mode = "temporal" if (self.window_size > 1 and n_cols % self.window_size == 0) else "flat"
            print(f"training data: {mode} mode, {n_features_in_data} features per timestep, "
                  f"{self.window_size} timestep(s) per sample.")

        self._prepare_train_val_dataloaders(X, y)
        self._execute_epoch_loop()

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
        return self._collect_batched_predictions(output_dim=output_dim, empty_1d_for_scalar=False)

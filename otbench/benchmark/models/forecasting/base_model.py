from typing import List, Optional, Union

import numpy as np
import pandas as pd


class BaseForecastingModel:
    """Base class for forecasting models."""

    def __init__(self,
                 name: str,
                 target_name: Union[str, List[str]],
                 window_size: int,
                 forecast_horizon: int,
                 predict_residuals: bool = False,
                 use_log10: bool = False,
                 feature_names: Optional[List[str]] = None,
                 **kwargs):
        self.name = name
        self.target_name = target_name
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.predict_residuals = predict_residuals
        self.use_log10 = use_log10
        self.feature_names = list(feature_names) if feature_names is not None else None
        self._persistence_cols = None
        self._persistence_indices_3d: Optional[List[int]] = None
        self._trained_on_3d: bool = False

    def train(
        self,
        X: Union['pd.DataFrame', np.ndarray],
        y: Union['pd.DataFrame', 'pd.Series', np.ndarray],
        feature_names: Optional[List[str]] = None,
    ):
        """Prepare targets (computing residuals if configured) and train the model."""
        if feature_names is not None:
            self.feature_names = list(feature_names)

        # 3D sequence tensor path: X has shape (N, window_size, n_features)
        if isinstance(X, np.ndarray) and X.ndim == 3:
            self._trained_on_3d = True
            y_arr = np.asarray(y, dtype=float)
            if y_arr.ndim == 1:
                y_arr = y_arr[:, np.newaxis]

            if not self.use_log10 and len(y_arr) > 0:
                p_idx = self._find_persistence_indices_3d(X.shape[2], strict=False)
                if p_idx:
                    self._check_auto_log10(y_arr, X[:, -1, p_idx])

            if self.predict_residuals:
                self._persistence_indices_3d = self._find_persistence_indices_3d(X.shape[2], strict=True)
                y_current = self._maybe_log10_array(X[:, -1, self._persistence_indices_3d].copy())
                y_training = y_arr - y_current
            else:
                y_training = y_arr

            self._train(X, y_training)
            return

        if isinstance(y, pd.Series):
            y = y.to_frame()
        elif isinstance(y, np.ndarray):
            y = pd.DataFrame(y, index=X.index if isinstance(X, pd.DataFrame) else None)

        if (self.feature_names is None and isinstance(X, pd.DataFrame) and self.window_size >= 1
                and len(X.columns) % self.window_size == 0):
            self.feature_names = list(X.columns[:len(X.columns) // self.window_size])

        if not self.use_log10 and isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame) and len(y) > 0:
            try:
                p_cols = self._find_persistence_cols(X, y)
                self._check_auto_log10(y.values, X[p_cols].values)
            except ValueError:
                pass

        if self.predict_residuals:
            self._persistence_cols = self._find_persistence_cols(X, y)
            y_current = self._maybe_log10_array(X[self._persistence_cols].values.copy())
            y_target = y.values - y_current
            y_training = pd.DataFrame(y_target, index=y.index, columns=y.columns)
        else:
            y_training = y

        self._train(X, y_training)

    def predict(self, X: Union['pd.DataFrame', np.ndarray], feature_names: Optional[List[str]] = None):
        """Generate predictions and reconstruct absolute values when predicting residuals."""
        if feature_names is not None:
            self.feature_names = list(feature_names)

        y_pred = self._predict(X)

        if self.predict_residuals:
            if isinstance(X, np.ndarray) and X.ndim == 3:
                if self._persistence_indices_3d is None:
                    self._persistence_indices_3d = self._find_persistence_indices_3d(X.shape[2], strict=True)
                y_current = self._maybe_log10_array(X[:, -1, self._persistence_indices_3d].copy())
                if y_pred.ndim == 1 and y_current.shape[1] == 1:
                    return y_current.ravel() + y_pred
                if y_pred.ndim != y_current.ndim:
                    y_pred = y_pred.reshape(y_current.shape)
                return y_current + y_pred

            if self._persistence_cols is None:
                dummy_y = pd.DataFrame(
                    columns=self.target_name if isinstance(self.target_name, list) else [self.target_name])
                self._persistence_cols = self._find_persistence_cols(X, dummy_y)

            y_current = self._maybe_log10_array(X[self._persistence_cols].values.copy())
            if y_pred.ndim != y_current.ndim:
                y_pred = y_pred.reshape(y_current.shape)

            return y_current + y_pred

        return y_pred

    def _check_auto_log10(self, y_vals: np.ndarray, x_vals: np.ndarray) -> None:
        """Promote use_log10=True when targets are in log10 space (< -5) and inputs are positive linear (< 1e-5)."""
        y_valid = y_vals[np.isfinite(y_vals)]
        if len(y_valid) > 0 and np.median(y_valid) < -5.0:
            x_valid = x_vals[np.isfinite(x_vals)]
            if len(x_valid) > 0 and np.min(x_valid) >= 0.0 and np.median(x_valid) < 1e-5:
                self.use_log10 = True

    def _maybe_log10_array(self, arr: np.ndarray) -> np.ndarray:
        """Convert non-negative linear values to log10 space when use_log10 is active, avoiding double-logging."""
        if not self.use_log10 or len(arr) == 0:
            return arr
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0 or np.min(valid) >= 0.0:
            return np.log10(np.maximum(arr, 1e-19))
        return arr

    def _resolve_target_lag_columns(
        self,
        X: pd.DataFrame,
        target: str,
        all_targets: Optional[List[str]] = None,
    ) -> List[str]:
        """Resolve all lag columns in X corresponding to target, ordered newest-first [t-0, t-1, ...]."""
        cols = [c for c in X.columns if c == target or c.startswith(f"{target} (t-") or c.startswith(f"{target}_t-")]
        if not cols:
            targets = all_targets or (self.target_name if isinstance(self.target_name, list) else [self.target_name])
            other_targets = [ot for ot in targets if ot != target and ot.startswith(target)]
            cols = [c for c in X.columns if c.startswith(target) and not any(c.startswith(ot) for ot in other_targets)]
        if not cols:
            return []
        p_col_idx = (cols.index(f"{target} (t-0)") if f"{target} (t-0)" in cols else
                     (cols.index(f"{target}_t-0") if f"{target}_t-0" in cols else 0))
        if p_col_idx > 0:
            cols = list(reversed(cols))
        return cols

    def _extract_target_window(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        target: str,
        all_targets: Optional[List[str]] = None,
    ) -> Optional[np.ndarray]:
        """Extract the (N, n_lags) target history matrix for target (ordered newest-first, with optional log10)."""
        if isinstance(X, np.ndarray) and X.ndim == 3:
            if self.feature_names is not None and target in self.feature_names:
                idx = self.feature_names.index(target)
                X_t = np.asarray(X[:, ::-1, idx], dtype=float).copy()
                return self._maybe_log10_array(X_t)
            return None
        cols = self._resolve_target_lag_columns(X, target, all_targets=all_targets)
        if not cols:
            return None
        X_t = X[cols].to_numpy(dtype=float).copy()
        return self._maybe_log10_array(X_t)

    def _find_persistence_indices_3d(self, n_features: int, strict: bool = True) -> List[int]:
        """Resolve feature-axis indices corresponding to target_name in a 3D (N, W, F) sequence array."""
        targets = self.target_name if isinstance(self.target_name, list) else [self.target_name]
        if self.feature_names is not None and len(self.feature_names) != n_features:
            filtered = [f for f in self.feature_names if f.lower() not in ("night_id", "session_id")]
            if len(filtered) == n_features:
                self.feature_names = filtered
        if self.feature_names is not None and len(self.feature_names) == n_features:
            indices: List[int] = []
            for t in targets:
                if t in self.feature_names:
                    indices.append(self.feature_names.index(t))
                elif strict:
                    raise ValueError(
                        f"Residual Learning Error: Target '{t}' not found in feature_names {self.feature_names}.")
            return indices
        if self.feature_names is None and len(targets) <= n_features:
            try:
                from otbench.tasks import TaskApi
                api = TaskApi()
                for t_name in api.list_tasks():
                    if not t_name.startswith("forecasting."):
                        continue
                    t_cfg = api._get_task(t_name)
                    t_targets = t_cfg["target"] if isinstance(t_cfg["target"], list) else [t_cfg["target"]]
                    if t_targets == targets:
                        task_obj = api.get_task(t_name)
                        cand_features = task_obj.feature_names
                        if len(cand_features) == n_features and all(t in cand_features for t in targets):
                            self.feature_names = cand_features
                            return [cand_features.index(t) for t in targets]
            except Exception:
                pass
        if strict:
            raise ValueError(
                "feature_names of length n_features must be provided when using 3D sequence arrays with predict_residuals=True."
            )
        return []

    def _find_persistence_cols(self, X: pd.DataFrame, y: pd.DataFrame) -> List[str]:
        """Resolve the lag-0 feature columns in X corresponding to the target columns in y."""
        found_cols = []
        targets = y.columns.tolist()

        for t in targets:
            if t in X.columns:
                found_cols.append(t)
            elif f"{t} (t-0)" in X.columns:
                found_cols.append(f"{t} (t-0)")
            elif f"{t}_t-0" in X.columns:
                found_cols.append(f"{t}_t-0")
            else:
                raise ValueError(
                    f"Residual Learning Error: Could not find current value feature for target '{t}' in X. "
                    f"Available features: {list(X.columns)}")
        return found_cols

    def _train(self, X: 'pd.DataFrame', y: 'pd.DataFrame'):
        """Concrete implementation of training logic."""
        raise NotImplementedError

    def _predict(self, X: 'pd.DataFrame'):
        """Concrete implementation of prediction logic."""
        raise NotImplementedError

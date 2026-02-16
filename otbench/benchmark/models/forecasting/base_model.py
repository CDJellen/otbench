# otbench/benchmark/models/forecasting/base_model.py
from typing import Union, List
import pandas as pd
import numpy as np

class BaseForecastingModel:
    """
    Base class for forecasting models.
    
    Features:
    1. Template Method Pattern: Defines the skeleton of train/predict operations.
    2. Residual Learning: Can automatically transform the problem to predict changes (y_t+k - y_t).
    3. Domain Adaptation: Handles the mismatch between Linear Features (X) and Log Targets (y).
    """

    def __init__(self, 
                 name: str, 
                 target_name: Union[str, List[str]], 
                 window_size: int, 
                 forecast_horizon: int, 
                 predict_residuals: bool = False, 
                 use_log10: bool = False, 
                 **kwargs):
        self.name = name
        self.target_name = target_name
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.predict_residuals = predict_residuals
        self.use_log10 = use_log10
        self._persistence_cols = None  # Cache for feature columns corresponding to y_t

    def train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """
        Public API: Prepares targets (calculating residuals if needed) and calls _train().
        """
        # 1. Standardize y to DataFrame
        if isinstance(y, pd.Series):
            y = y.to_frame()
        elif isinstance(y, np.ndarray):
            y = pd.DataFrame(y, index=X.index)

        # 2. Handle Residual Transformation
        if self.predict_residuals:
            # A. Identify "Current Value" columns in X
            self._persistence_cols = self._find_persistence_cols(X, y)
            
            # B. Extract y_t (Current State)
            y_current = X[self._persistence_cols].values
            
            # C. Domain Check: If Task uses Log10, y is Log, but X is Linear.
            # We must Log X to compare apples-to-apples.
            if self.use_log10:
                y_current = np.log10(np.maximum(y_current, 1e-19))
            
            # D. Compute Residuals (Delta = Future - Current)
            y_target = y.values - y_current
            
            # E. Wrap for model consumption
            y_training = pd.DataFrame(y_target, index=y.index, columns=y.columns)
            
            # Optional: Logging
            # print(f"[{self.name}] Residual Mode: Mean Delta = {np.nanmean(y_training):.4e}")
        else:
            y_training = y

        # 3. Delegate to Concrete Implementation
        self._train(X, y_training)

    def predict(self, X: 'pd.DataFrame'):
        """
        Public API: Generates predictions and reconstructs absolute values if needed.
        """
        # 1. Get raw prediction (Deltas or Absolutes) from subclass
        y_pred = self._predict(X)

        # 2. Reconstruct Absolute Values if needed
        if self.predict_residuals:
            # Fallback for inference-only (if train wasn't called on this instance)
            if self._persistence_cols is None:
                dummy_y = pd.DataFrame(columns=self.target_name if isinstance(self.target_name, list) else [self.target_name])
                self._persistence_cols = self._find_persistence_cols(X, dummy_y)

            # A. Extract y_t
            y_current = X[self._persistence_cols].values
            
            # B. Domain Check (Linear X -> Log Y)
            if self.use_log10:
                y_current = np.log10(np.maximum(y_current, 1e-19))
            
            # C. Reconstruct: y_{t+k} = y_t + Delta
            # Handle broadcasting if shapes mismatch (e.g. 1D prediction vs 2D history)
            if y_pred.ndim != y_current.ndim:
                y_pred = y_pred.reshape(y_current.shape)
                
            return y_current + y_pred
            
        return y_pred

    def _find_persistence_cols(self, X: pd.DataFrame, y: pd.DataFrame) -> List[str]:
        """
        Heuristic to find the feature columns in X that correspond to the targets in y at lag 0.
        """
        found_cols = []
        targets = y.columns.tolist()
        
        for t in targets:
            # Case 1: Exact Name (e.g. 'cn2_500')
            if t in X.columns:
                found_cols.append(t)
            # Case 2: Explicit Lag Name (e.g. 'cn2_500 (t-0)')
            elif f"{t} (t-0)" in X.columns:
                found_cols.append(f"{t} (t-0)")
            else:
                # Failure
                raise ValueError(f"Residual Learning Error: Could not find current value feature for target '{t}' in X. "
                                 f"Available features: {list(X.columns)}")
        return found_cols

    def _train(self, X: 'pd.DataFrame', y: 'pd.DataFrame'):
        """Concrete implementation of training logic."""
        raise NotImplementedError

    def _predict(self, X: 'pd.DataFrame'):
        """Concrete implementation of prediction logic."""
        raise NotImplementedError

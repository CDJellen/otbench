# otbench/benchmark/models/forecasting/persistence.py
from typing import Union, List
import numpy as np
import pandas as pd
from otbench.benchmark.models.forecasting.base_model import BaseForecastingModel

class PersistenceForecastingModel(BaseForecastingModel):
    """
    A model which predicts the most recent value of the target variable.
    
    Modes:
    1. Dynamic (Preferred): Uses the most recent value from X (y_t) for each sample.
    2. Static (Fallback): Uses the last value seen during training (y_last).
    """

    def __init__(self, name: str, target_name: Union[str, List[str]], **kwargs):
        super().__init__(name, target_name, **kwargs)
        self.static_persistence = np.nan
        self.output_size = kwargs.get("output_size", 1)
        self.target_cols = target_name if isinstance(target_name, list) else [target_name]

    def _train(self, X: 'pd.DataFrame', y: Union['pd.DataFrame', 'pd.Series', np.ndarray]):
        """Store the last observed value(s) as a static fallback."""
        y_arr = np.asarray(y)
        if len(y_arr) == 0:
            self.static_persistence = np.nan
            return

        # Store the last row of the training target
        # Note: If use_log10=True, 'y' passed here is already Log10 (from Task).
        self.static_persistence = y_arr[-1]

    def _predict(self, X: 'pd.DataFrame'):
        """
        Predict using Dynamic Persistence if features available, else Static.
        """
        n_samples = len(X)
        
        # 1. Handle Empty Input Edge Case
        if n_samples == 0:
            if self.output_size > 1:
                return np.empty((0, self.output_size))
            return np.array([])

        # 2. Residual Mode Bypass
        # If we are predicting residuals, the "Persistence" guess for a delta is 0.
        # (i.e. No change from t to t+k).
        if self.predict_residuals:
            if self.output_size > 1:
                return np.zeros((n_samples, self.output_size))
            return np.zeros(n_samples)

        # 3. Dynamic Persistence (Standard Mode)
        try:
            feature_cols = self._find_features_for_targets(X)
            vals = X[feature_cols].values
            
            # CRITICAL: Transform Linear X -> Log Y
            if self.use_log10:
                vals = np.log10(np.maximum(vals, 1e-19))
                
            return vals
            
        except ValueError:
            # 4. Static Fallback
            if np.all(np.isnan(self.static_persistence)):
                 raise RuntimeError("Model has not been trained and X does not contain target history.")

            # Reshape static persistence for broadcasting
            if np.ndim(self.static_persistence) == 0 or (np.ndim(self.static_persistence) == 1 and len(self.static_persistence) == 1):
                return np.full(n_samples, self.static_persistence)
            
            return np.tile(self.static_persistence, (n_samples, 1))

    def _find_features_for_targets(self, X: pd.DataFrame) -> List[str]:
        """Identifies columns in X that correspond to the target variable(s) at t=0."""
        found_cols = []
        for t in self.target_cols:
            if t in X.columns:
                found_cols.append(t)
            elif f"{t} (t-0)" in X.columns:
                found_cols.append(f"{t} (t-0)")
            else:
                raise ValueError(f"Target '{t}' not found in X features.")
        return found_cols
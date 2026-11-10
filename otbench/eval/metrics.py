from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import sklearn.metrics as sk_m

from otbench.eval.utils import _get_valid_indices, _format_metric, _bootstrap_ci, bootstrap_ci, paired_bootstrap_test
from otbench.eval.integrated_metrics import (
    integrated_seeing,
    isoplanatic_angle,
    coherence_time,
    greenwood_frequency,
    profile_to_fried_parameter,
    ao_wavefront_error_budget,
    strehl_ratio,
    ground_layer_fraction,
)

__all__ = [
    "MetricSpec",
    "get_metric_spec",
    "is_implemented_metric",
    "is_lower_better",
    "coefficient_of_determination",
    "root_mean_square_error",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "integrated_seeing",
    "isoplanatic_angle",
    "coherence_time",
    "greenwood_frequency",
    "per_layer_rmse",
    "profile_to_fried_parameter",
    "ao_wavefront_error_budget",
    "strehl_ratio",
    "ground_layer_fraction",
    "bootstrap_ci",
    "paired_bootstrap_test",
]


@dataclass(frozen=True)
class MetricSpec:
    """Declarative specification of a callable metric's signature and bootstrap semantics."""
    fn: Callable
    lower_is_better: bool = True
    needs_heights: bool = False
    needs_layer_names: bool = False
    needs_wind_speed: bool = False
    bootstrap_reduction: Optional[str] = None
    uses_detailed_score: bool = False

    @property
    def requires_profile_context(self) -> bool:
        """Return True if this metric accepts profile heights, layer names, or wind speed."""
        return self.needs_heights or self.needs_layer_names or self.needs_wind_speed

    def extract_sample_losses(
        self,
        y_true_valid: Optional[np.ndarray],
        y_pred_valid: Optional[np.ndarray],
        eval_result: Optional[dict] = None,
    ) -> Optional[np.ndarray]:
        """Extract per-sample losses aligned with bootstrap_reduction, or None if non-decomposable."""
        if self.uses_detailed_score:
            if isinstance(eval_result, dict) and "detailed_score" in eval_result:
                return np.asarray(eval_result["detailed_score"], dtype=float)**2
            return None
        if y_true_valid is None or y_pred_valid is None or len(y_true_valid) == 0:
            return None
        if self.bootstrap_reduction == "rmse":
            return (y_true_valid - y_pred_valid)**2
        if self.bootstrap_reduction == "mae":
            return np.abs(y_true_valid - y_pred_valid)
        return None


_HIGHER_IS_BETTER_METRICS = frozenset({
    "coefficient_of_determination",
    "skill_score",
})


def is_implemented_metric(metric_name: str) -> bool:
    """Return True if metric_name is a callable metric in this module."""
    return metric_name in _METRIC_NAMES


def is_lower_better(metric_name: str) -> bool:
    """Return True if lower values of metric_name indicate better model performance."""
    spec = _METRIC_SPECS.get(metric_name)
    if spec is not None:
        return spec.lower_is_better
    return metric_name not in _HIGHER_IS_BETTER_METRICS


def get_metric_spec(metric_name: str) -> Optional[MetricSpec]:
    """Return the declarative MetricSpec for metric_name, or None if not registered."""
    return _METRIC_SPECS.get(metric_name)


def coefficient_of_determination(y_true: Sequence, y_pred: Sequence, detailed: bool = False) -> dict:
    """Calculates R2 score using `sklearn.metrics.r2_score`."""
    y_true, y_pred = _get_valid_indices(y_true=y_true, y_pred=y_pred)
    if len(y_pred) == 0:
        return _format_metric(np.nan, 0)

    if detailed:
        r2_raw = sk_m.r2_score(y_true, y_pred, multioutput="raw_values")
        r2_avg = np.mean(r2_raw)
        res = _format_metric(float(r2_avg), len(y_pred))
        res["detailed_score"] = r2_raw.tolist()
        return res

    r2 = sk_m.r2_score(y_true, y_pred, multioutput="uniform_average")
    return _format_metric(float(r2), len(y_pred))


def root_mean_square_error(y_true: Sequence, y_pred: Sequence, detailed: bool = False) -> dict:
    """Calculate RMSE from `sklearn.metrics.root_mean_squared_error`."""
    y_true, y_pred = _get_valid_indices(y_true=y_true, y_pred=y_pred)
    if len(y_pred) == 0:
        return _format_metric(np.nan, 0)

    if detailed:
        rmse_raw = sk_m.root_mean_squared_error(y_true, y_pred, multioutput="raw_values")
        rmse_avg = np.mean(rmse_raw)
        res = _format_metric(float(rmse_avg), len(y_pred))
        res["detailed_score"] = rmse_raw.tolist()
        return res

    return _format_metric(
        float(sk_m.root_mean_squared_error(y_true=y_true, y_pred=y_pred, multioutput="uniform_average")), len(y_pred))


def mean_absolute_error(y_true: Sequence, y_pred: Sequence, detailed: bool = False) -> dict:
    """An alias for `sklearn.metrics.mean_absolute_error`."""
    y_true, y_pred = _get_valid_indices(y_true=y_true, y_pred=y_pred)
    if len(y_pred) == 0:
        return _format_metric(np.nan, 0)

    if detailed:
        mae_raw = sk_m.mean_absolute_error(y_true, y_pred, multioutput="raw_values")
        mae_avg = np.mean(mae_raw)
        res = _format_metric(float(mae_avg), len(y_pred))
        res["detailed_score"] = mae_raw.tolist()
        return res

    return _format_metric(float(sk_m.mean_absolute_error(y_true=y_true, y_pred=y_pred, multioutput="uniform_average")),
                          len(y_pred))


def mean_absolute_percentage_error(y_true: Sequence, y_pred: Sequence, detailed: bool = False) -> dict:
    """Compute mean absolute percentage error via `sklearn.metrics.mean_absolute_percentage_error`.

    Note: for tasks that apply a base-10 log transform to the target (``log_transform: true``
    in the task specification), both ``y_true`` and ``y_pred`` are in log10 space.  MAPE
    computed in log10 space is *not* the standard percentage error on the raw values — it
    measures the relative error of the log10 quantities.  Use with care when comparing
    across tasks with different transform settings.
    """
    y_true, y_pred = _get_valid_indices(y_true=y_true, y_pred=y_pred)
    if len(y_pred) == 0:
        return _format_metric(np.nan, 0)

    if detailed:
        mape_raw = sk_m.mean_absolute_percentage_error(y_true, y_pred, multioutput="raw_values")
        mape_avg = np.mean(mape_raw)
        res = _format_metric(float(mape_avg), len(y_pred))
        res["detailed_score"] = mape_raw.tolist()
        return res

    return _format_metric(
        float(sk_m.mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred, multioutput="uniform_average")),
        len(y_pred))


def per_layer_rmse(
    y_true: Sequence,
    y_pred: Sequence,
    layer_names: Optional[List[str]] = None,
    detailed: bool = False,
) -> dict:
    """Per-layer RMSE for vector (profile) targets.

    Returns the aggregate RMSE as ``metric_value`` and a ``per_layer``
    dictionary mapping each layer name to its individual RMSE.  This
    prevents a model that excels at one layer from masking failures at
    another.

    Args:
        y_true: True profile (samples x layers).
        y_pred: Predicted profile (samples x layers).
        layer_names: Optional list of human-readable layer names
            (e.g. ["500m", "1000m", ...]). If None, layers are
            numbered 0, 1, 2, ...
        detailed: If True, also return per-sample errors per layer.
    """
    y_true, y_pred = _get_valid_indices(y_true=y_true, y_pred=y_pred)
    if len(y_pred) == 0:
        return _format_metric(np.nan, 0)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        # Scalar target — degenerate case, just return RMSE
        rmse = float(sk_m.root_mean_squared_error(y_true, y_pred))
        return _format_metric(rmse, len(y_pred))

    n_layers = y_true.shape[1]
    if layer_names is None:
        layer_names = [str(i) for i in range(n_layers)]
    elif len(layer_names) != n_layers:
        raise ValueError(f"layer_names length ({len(layer_names)}) must match number of target layers ({n_layers}).")

    rmse_per = sk_m.root_mean_squared_error(y_true, y_pred, multioutput="raw_values")
    rmse_avg = float(np.mean(rmse_per))

    res = _format_metric(rmse_avg, len(y_pred))
    res["per_layer"] = {name: float(val) for name, val in zip(layer_names, rmse_per)}

    if detailed:
        # Per-sample absolute error per layer
        res["detailed_score"] = (np.abs(y_true - y_pred)).tolist()

    return res


_METRIC_SPECS: Dict[str, MetricSpec] = {
    "coefficient_of_determination":
        MetricSpec(fn=coefficient_of_determination, lower_is_better=False),
    "root_mean_square_error":
        MetricSpec(fn=root_mean_square_error, bootstrap_reduction="rmse"),
    "mean_absolute_error":
        MetricSpec(fn=mean_absolute_error, bootstrap_reduction="mae"),
    "mean_absolute_percentage_error":
        MetricSpec(fn=mean_absolute_percentage_error),
    "per_layer_rmse":
        MetricSpec(fn=per_layer_rmse, needs_layer_names=True, bootstrap_reduction="rmse"),
    "integrated_seeing":
        MetricSpec(
            fn=integrated_seeing,
            needs_heights=True,
            bootstrap_reduction="rmse",
            uses_detailed_score=True,
        ),
    "isoplanatic_angle":
        MetricSpec(
            fn=isoplanatic_angle,
            needs_heights=True,
            bootstrap_reduction="rmse",
            uses_detailed_score=True,
        ),
    "coherence_time":
        MetricSpec(
            fn=coherence_time,
            needs_heights=True,
            needs_wind_speed=True,
            bootstrap_reduction="rmse",
            uses_detailed_score=True,
        ),
    "greenwood_frequency":
        MetricSpec(
            fn=greenwood_frequency,
            needs_heights=True,
            needs_wind_speed=True,
            bootstrap_reduction="rmse",
            uses_detailed_score=True,
        ),
    "strehl_ratio":
        MetricSpec(
            fn=strehl_ratio,
            needs_heights=True,
            needs_wind_speed=True,
            bootstrap_reduction="rmse",
            uses_detailed_score=True,
        ),
    "ground_layer_fraction":
        MetricSpec(
            fn=ground_layer_fraction,
            needs_heights=True,
            bootstrap_reduction="rmse",
            uses_detailed_score=True,
        ),
}

_METRIC_NAMES = frozenset(_METRIC_SPECS.keys())

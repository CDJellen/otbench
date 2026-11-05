import os
import json
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

import otbench.eval as eval_metrics
from otbench.dataset import Dataset
from otbench.config import BENCHMARK_FP, settings

_SUMMARY_METRIC_KEYS = frozenset({
    "metric_value",
    "valid_predictions",
    "per_layer",
    "ci_lower",
    "ci_upper",
    "mean_opd_rms_error_nm",
    "layer_weight_kl_divergence",
})


def _compact_metrics_for_benchmark(model_metrics: dict, max_summary_list_len: int = 64) -> dict:
    """Strip sample-level arrays from a metric dictionary before persisting to experiments.json.

    Preserves scalar metrics, confidence intervals, per-layer dictionaries, and
    per-output summary lists (e.g. 7-layer ``detailed_score`` from sklearn
    ``multioutput='raw_values'``), while removing O(N_samples) time-series arrays
    (e.g. ``seeing_true``, ``seeing_pred``, ``theta0_true``, or N x L error matrices).
    """
    compacted: dict = {}
    for metric_name, m_val in model_metrics.items():
        if not isinstance(m_val, dict):
            compacted[metric_name] = m_val
            continue
        entry = {k: v for k, v in m_val.items() if k in _SUMMARY_METRIC_KEYS}
        if "detailed_score" in m_val:
            ds = m_val["detailed_score"]
            if (isinstance(ds, (list, np.ndarray)) and len(ds) <= max_summary_list_len
                    and (len(ds) == 0 or not isinstance(ds[0], (list, np.ndarray)))):
                entry["detailed_score"] = ds.tolist() if isinstance(ds, np.ndarray) else ds
        compacted[metric_name] = entry
    return compacted


class TaskTypes(Enum):
    REGRESSION = "regression"
    FORECASTING = "forecasting"


class TaskABC(ABC):

    @abstractmethod
    def get_info(self, keys: Union[List[str], None] = None) -> dict:
        """Returns the full task information dictionary."""
        ...

    @abstractmethod
    def get_description(self) -> str:
        """Return the short description of the task."""
        ...

    @abstractmethod
    def get_long_description(self) -> str:
        """Return the long-form description of the task."""
        ...

    @abstractmethod
    def get_benchmark_info(self, task_name: Union[str, None]) -> dict:
        """Returns the benchmark information dictionary."""
        ...

    @abstractmethod
    def top_models(self, n: int = 5, metric: str = "") -> dict:
        """Returns the top n models for this task."""
        ...

    @abstractmethod
    def get_transforms(self) -> dict:
        """Return the description of the transforms applied to the X and y data."""
        ...

    @abstractmethod
    def get_target_name(self) -> Union[str, List[str]]:
        """Return the target feature name for this task."""
        ...

    @abstractmethod
    def get_unavailable_features(self) -> List[str]:
        """Return the names of features which are unavailable for training and inference in this task."""
        ...

    @abstractmethod
    def get_metric_names(self) -> List[str]:
        """Return the evaluation metric names for this task."""
        ...

    @abstractmethod
    def get_dataset(self) -> Dataset:
        """Return the underlying dataset."""
        ...

    @abstractmethod
    def get_citation(self) -> str:
        """Return the BibTeX citation text for this task's dataset."""
        ...

    @abstractmethod
    def get_df(self) -> pd.DataFrame:
        """Return the underlying pd.DataFrame for this task's dataset."""
        ...

    @abstractmethod
    def get_data(self, data_type: str) -> Any:
        """Return the underlying data."""
        ...

    @abstractmethod
    def get_train_data(self, data_type: str) -> Any:
        """Return the underlying training data for this task."""
        ...

    @abstractmethod
    def get_test_data(self, data_type: str) -> Any:
        """Return the underlying test data for this task."""
        ...

    @abstractmethod
    def get_validation_data(self, data_type: str) -> Any:
        """Return the underlying validation data for this task."""
        ...

    @abstractmethod
    def evaluate_model(
        self,
        predict_call: Callable,
        data_type: str,
        x_transforms: Union[Callable, None] = None,
        x_transform_kwargs: Union[dict, None] = None,
        eval_metric_names: Union[List[str], None] = None,
        return_predictions: bool = False,
        include_as_benchmark: bool = False,
        model_name: Union[str, None] = None,
        detailed_metrics: bool = False,
        overwrite: bool = False,
    ) -> Union[dict, Tuple[dict, 'np.ndarray']]:
        """Evaluate a model against this task's transformed test set, default against all metrics."""
        ...

    @abstractmethod
    def compare_models(
        self,
        y_true: Any,
        y_pred_a: Any,
        y_pred_b: Any,
        eval_metric_names: Optional[List[str]] = None,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        bootstrap_block_size: int = 1,
        block_size: Optional[int] = None,
        seed: int = 42,
    ) -> dict:
        """Run paired bootstrap significance tests comparing Model A vs. Model B."""
        ...

    @abstractmethod
    def evaluate_by_regime(
        self,
        y_true: Any,
        y_pred: Any,
        eval_metric_names: Optional[List[str]] = None,
        compute_ci: bool = False,
        n_bootstrap: int = 200,
        ci: float = 0.95,
        bootstrap_block_size: int = 1,
    ) -> dict:
        """Stratify evaluation across physical optical turbulence regimes derived from y_true."""
        ...


class BaseTask(TaskABC):

    def __init__(self, task_type: str, task_name: str, task: dict, benchmark_fp: str = BENCHMARK_FP) -> None:
        super().__init__()
        self.task_type = task_type
        self.task_name = task_name
        self.task = task
        self.benchmark_fp = benchmark_fp
        self._init_dataset_for_task()

    def get_info(self, keys: Union[List[str], None] = None) -> dict:
        """Returns the task information dictionary."""
        task = self.task
        return {k: task[k] for k in keys} if keys is not None else task

    def get_benchmark_info(self, task_name: Union[str, None] = None) -> dict:
        """Returns the benchmark information dictionary."""
        if os.path.exists(self.benchmark_fp):
            with open(self.benchmark_fp, "r") as f:
                benchmark_info = json.load(f)
            if task_name is None:
                task_name = self.task_name
            if task_name == "*":
                return benchmark_info
            return benchmark_info[task_name]
        else:
            raise FileNotFoundError(
                f"benchmark file {self.benchmark_fp} not found. Please run `otb benchmark` to generate the benchmark file."
            )

    def top_models(self, n: int = 5, metric: str = "") -> dict:
        """Returns the top n models for this task."""
        benchmark_info = self.get_benchmark_info(task_name=self.task_name)
        if metric == "":
            metrics = self.get_metric_names()
            if len(metrics) < 1:
                raise ValueError(f"task {self.task_name} has no metrics defined.")
            metric = metrics[0]
        model_benchmark_info = {k: v for k, v in benchmark_info.items() if k != "possible_predictions" and metric in v}
        model_benchmark_info_keys = sorted(model_benchmark_info.keys(),
                                           key=lambda x: model_benchmark_info[x][metric]["metric_value"])[:n]
        model_benchmark_info = {k: model_benchmark_info[k] for k in model_benchmark_info_keys}
        model_benchmark_info["possible_predictions"] = benchmark_info["possible_predictions"]

        return model_benchmark_info

    def get_description(self) -> str:
        """Return the description of the task."""
        return self.task["description"]

    def get_long_description(self) -> str:
        """Return the description of the task."""
        return self.task["description_long"]

    def get_transforms(self) -> dict:
        """Return the description of the transforms applied to the X and y data."""
        transform_info = {
            "log_transform": {
                "description": "Whether to apply a base-10 log transform to the target.",
                "value": self.task["log_transform"]
            },
            "dropna": {
                "description":
                    "Whether to drop observations with any missing values across both the features and target.",
                "value":
                    self.task["dropna"]
            }
        }

        return transform_info

    def get_target_name(self) -> Union[str, List[str]]:
        """Return the target feature name for this task."""
        return self.task["target"]

    def get_unavailable_features(self) -> List[str]:
        """Return the names of features which are unavailable for training and inference in this task."""
        return self.task["remove"]

    def get_metric_names(self) -> List[str]:
        """Return the evaluation metric names for this task."""
        return self.task["eval_metrics"]

    def get_dataset(self) -> Dataset:
        """Return the underlying dataset."""
        return self._ds

    def get_citation(self) -> str:
        """Return the BibTeX / markdown citation text for this task's dataset."""
        return self._ds.get_citation()

    def get_df(self) -> pd.DataFrame:
        """Return the underlying pd.DataFrame for this task's dataset."""
        return self._ds.get_all(data_type="pd")

    def get_data(self, data_type: str = "pd") -> Any:
        """Return the underlying data."""
        return self._ds.get_all(data_type=data_type)

    def get_train_data(self, data_type: str = "pd") -> Any:
        """Return the underlying training data for this task."""
        return self._ds.get_train(task=self.task, data_type=data_type)

    def get_test_data(self, data_type: str = "pd") -> Any:
        """Return the underlying test data for this task."""
        return self._ds.get_test(task=self.task, data_type=data_type)

    def get_validation_data(self, data_type: str = "pd") -> Any:
        """Return the underlying validation data for this task."""
        return self._ds.get_val(task=self.task, data_type=data_type)

    def evaluate_model(
        self,
        predict_call: Callable,
        data_type: str = "pd",
        x_transforms: Union[Callable, None] = None,
        x_transform_kwargs: Union[dict, None] = None,
        eval_metric_names: Union[List[str], None] = None,
        return_predictions: bool = False,
        include_as_benchmark: bool = False,
        model_name: Union[str, None] = None,
        detailed_metrics: bool = False,
        overwrite: bool = True,
        compute_ci: bool = False,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        bootstrap_block_size: int = 1,
    ) -> Union[dict, Tuple[dict, 'np.ndarray']]:
        """Evaluate a model against this task's transformed test set, default against all metrics."""
        raise NotImplementedError

    def _init_dataset_for_task(self) -> None:
        """Get the underlying dataset for the given task"""
        self._ds = Dataset(name=self.task["ds_name"])

    def _add_experiment_to_benchmarks(self, model_name: str, model_metrics: dict, overwrite: bool) -> None:
        if model_name is None:
            raise ValueError("model_name must be provided if include_as_benchmark is True.")
        benchmark_info = self.get_benchmark_info(task_name="*")
        task_benchmarks = benchmark_info.setdefault(self.task_name, {})
        if not overwrite and model_name in task_benchmarks:
            model_name = f"{model_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        task_benchmarks[model_name] = _compact_metrics_for_benchmark(model_metrics)
        with open(self.benchmark_fp, "w") as f:
            json.dump(
                benchmark_info,
                f,
                indent=4,
                default=lambda o: o.item() if hasattr(o, "item") else o.tolist() if hasattr(o, "tolist") else str(o),
            )

    # Metrics requiring profile context (heights, wind speed, layer names).
    _PROFILE_METRICS = frozenset({
        "integrated_seeing",
        "isoplanatic_angle",
        "coherence_time",
        "greenwood_frequency",
        "per_layer_rmse",
        "strehl_ratio",
    })

    # Subset of profile metrics that accept a 'heights' parameter.
    _HEIGHT_METRICS = frozenset({
        "integrated_seeing",
        "isoplanatic_angle",
        "coherence_time",
        "greenwood_frequency",
        "strehl_ratio",
    })

    def _build_metric_kwargs(self, metric_name: str) -> dict:
        """Build additional kwargs for profile-aware metrics.

        Injects ``heights``, ``wind_speed``, and ``layer_names`` from the
        task/dataset configuration so that profile metrics can be used
        declaratively in ``eval_metrics`` lists without custom code.

        Each kwarg is only injected for metrics whose signature accepts it:
          - ``heights``: AO metrics (integrated_seeing, isoplanatic_angle, etc.)
          - ``layer_names``: per_layer_rmse
          - ``wind_speed``: coherence_time, greenwood_frequency, strehl_ratio
        """
        if metric_name not in self._PROFILE_METRICS:
            return {}

        kwargs: dict = {}

        # Resolve heights from dataset xarray coords or target column names.
        ds = self.get_dataset()
        xr_ds = ds.get_xarray()

        ds_config = ds._supported_datasets().get(self._name_to_ds(), {})
        feature_map = ds_config.get("feature_map", {})
        height_coord = feature_map.get("height_mass_coord", "height_mass")

        target = self.task.get("target", [])
        heights: list = []

        if xr_ds is not None and height_coord in xr_ds.coords:
            heights = [
                int(h) if float(h).is_integer() else float(h) for h in xr_ds.coords[height_coord].values.tolist()
            ]
            if isinstance(target, list) and "cn2_ground_scalar" in target and 0 not in heights:
                heights = [0] + heights

        if isinstance(target, list) and len(heights) != len(target):
            # Derive heights from target column names (e.g. cn2_free_atmos_500)
            heights = []
            has_ground = False
            for t in target:
                if t == "cn2_ground_scalar":
                    has_ground = True
                    continue
                parts = t.rsplit("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    heights.append(int(parts[1]))
            # Prepend ground layer at h=0 if target includes it.
            if has_ground:
                heights = [0] + heights

        # Inject heights only for metrics that accept them.
        if heights and metric_name in self._HEIGHT_METRICS:
            kwargs["heights"] = heights

        # Layer names for per_layer_rmse (derived from heights).
        if metric_name == "per_layer_rmse" and heights:
            kwargs["layer_names"] = [f"{h}m" for h in heights]

        # Effective wind speed for tau_0, f_G, and strehl_ratio:
        # Hold v_bar at the canonical site reference constant (10.0 m/s for Paranal,
        # overridable via task["effective_wind_speed"]) during automated benchmark
        # evaluation so profile forecast errors at t+H are not conflated with
        # unmodeled 30 m surface anemometer fluctuations at observation time t.
        if metric_name in ("coherence_time", "greenwood_frequency", "strehl_ratio"):
            wind_col = feature_map.get("wind_speed_col_name")
            if wind_col or "effective_wind_speed" in self.task:
                kwargs["wind_speed"] = float(self.task.get("effective_wind_speed", 10.0))

        return kwargs

    def _compute_metrics_dict(
        self,
        y_true: Any,
        y_pred: Any,
        eval_metric_names: Optional[List[str]] = None,
        detailed_metrics: bool = False,
        compute_ci: bool = False,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        bootstrap_block_size: int = 1,
    ) -> dict:
        """Compute all requested metrics (and optional vectorized bootstrap CIs) for (y_true, y_pred)."""
        from otbench.eval.utils import _get_valid_indices

        if eval_metric_names is None:
            eval_metric_names = self.task["eval_metrics"]
        model_metrics = {k: -1 for k in eval_metric_names}

        yt_valid, yp_valid = (_get_valid_indices(y_true, y_pred) if compute_ci else (None, None))

        for m in eval_metric_names:
            if eval_metrics.is_implemented_metric(m):
                extra_kwargs = self._build_metric_kwargs(m)
                metric_fn = getattr(eval_metrics, m)
                need_sample_detail = compute_ci and m in self._HEIGHT_METRICS
                val = metric_fn(
                    y_true,
                    y_pred,
                    detailed=(detailed_metrics or need_sample_detail),
                    **extra_kwargs,
                )
                if compute_ci and isinstance(val, dict) and val.get("valid_predictions", 0) > 1:
                    sample_losses = None
                    reduction = "rmse"
                    if m in ("root_mean_square_error", "per_layer_rmse") and yt_valid is not None:
                        sample_losses = (yt_valid - yp_valid)**2
                        reduction = "rmse"
                    elif m == "mean_absolute_error" and yt_valid is not None:
                        sample_losses = np.abs(yt_valid - yp_valid)
                        reduction = "mae"
                    elif m in self._HEIGHT_METRICS and "detailed_score" in val:
                        sample_losses = np.asarray(val["detailed_score"], dtype=float)**2
                        reduction = "rmse"

                    if sample_losses is not None:
                        ci_low, ci_high = eval_metrics.bootstrap_ci(
                            y_true,
                            y_pred,
                            sample_losses=sample_losses,
                            reduction=reduction,
                            n_bootstrap=n_bootstrap,
                            ci=ci,
                            block_size=bootstrap_block_size,
                        )
                    else:
                        ci_low, ci_high = eval_metrics.bootstrap_ci(
                            y_true,
                            y_pred,
                            metric_fn=lambda yt, yp, _fn=metric_fn, _kw=extra_kwargs: _fn(yt, yp, detailed=False, **_kw)
                            ["metric_value"],
                            n_bootstrap=n_bootstrap,
                            ci=ci,
                            block_size=bootstrap_block_size,
                        )
                    val["ci_lower"] = ci_low
                    val["ci_upper"] = ci_high

                if need_sample_detail and not detailed_metrics and isinstance(val, dict):
                    val = _compact_metrics_for_benchmark({m: val})[m]

                model_metrics[m] = val
            else:
                warnings.warn(f"Metric '{m}' is not in the implemented metrics registry. Skipping.")
        return model_metrics

    def compare_models(
        self,
        y_true: Any,
        y_pred_a: Any,
        y_pred_b: Any,
        eval_metric_names: Optional[List[str]] = None,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        bootstrap_block_size: int = 1,
        block_size: Optional[int] = None,
        seed: int = 42,
    ) -> dict:
        """Run paired (i.i.d. or moving-block) bootstrap significance tests comparing Model A vs. Model B."""
        if eval_metric_names is None:
            eval_metric_names = self.task["eval_metrics"]
        eff_block_size = block_size if block_size is not None else bootstrap_block_size

        ao_detail_keys = {
            "integrated_seeing": ("seeing_true", "seeing_pred"),
            "isoplanatic_angle": ("theta0_true", "theta0_pred"),
            "coherence_time": ("tau0_true", "tau0_pred"),
            "greenwood_frequency": ("fG_true", "fG_pred"),
            "strehl_ratio": ("strehl_true", "strehl_pred"),
        }

        comparisons: dict = {}
        for m in eval_metric_names:
            if not eval_metrics.is_implemented_metric(m):
                continue
            extra_kwargs = self._build_metric_kwargs(m)
            metric_fn = getattr(eval_metrics, m)
            lower_is_better = m != "coefficient_of_determination"

            sample_losses_a = None
            sample_losses_b = None
            reduction = "rmse"
            fn_arg = None

            if m in ("root_mean_square_error", "per_layer_rmse"):
                reduction = "rmse"
            elif m == "mean_absolute_error":
                from otbench.eval.utils import _get_valid_indices

                yt_a, ypa = _get_valid_indices(y_true, y_pred_a)
                _, ypb = _get_valid_indices(y_true, y_pred_b)
                if len(yt_a) == len(ypb):
                    sample_losses_a = np.abs(yt_a - ypa)
                    sample_losses_b = np.abs(yt_a - ypb)
                    reduction = "mae"
            elif m in ao_detail_keys:
                true_k, pred_k = ao_detail_keys[m]
                val_a = metric_fn(y_true=y_true, y_pred=y_pred_a, detailed=True, **extra_kwargs)
                val_b = metric_fn(y_true=y_true, y_pred=y_pred_b, detailed=True, **extra_kwargs)
                if true_k in val_a and pred_k in val_a and true_k in val_b and pred_k in val_b:
                    la = (np.asarray(val_a[true_k], dtype=float) - np.asarray(val_a[pred_k], dtype=float))**2
                    lb = (np.asarray(val_b[true_k], dtype=float) - np.asarray(val_b[pred_k], dtype=float))**2
                    if len(la) == len(lb):
                        sample_losses_a = la
                        sample_losses_b = lb
                        reduction = "rmse"
            else:
                fn_arg = lambda yt, yp, _fn=metric_fn, _kw=extra_kwargs: _fn(yt, yp, detailed=False, **_kw)

            comparisons[m] = eval_metrics.paired_bootstrap_test(
                y_true=y_true,
                y_pred_a=y_pred_a,
                y_pred_b=y_pred_b,
                metric_fn=fn_arg,
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed,
                block_size=eff_block_size,
                lower_is_better=lower_is_better,
                sample_losses_a=sample_losses_a,
                sample_losses_b=sample_losses_b,
                reduction=reduction,
            )
        return comparisons

    def evaluate_by_regime(
        self,
        y_true: Any,
        y_pred: Any,
        eval_metric_names: Optional[List[str]] = None,
        compute_ci: bool = False,
        n_bootstrap: int = 200,
        ci: float = 0.95,
        bootstrap_block_size: int = 1,
    ) -> dict:
        """Stratify evaluation across physical optical turbulence regimes derived from y_true.

        For multi-layer profile tasks (ndim == 2 and n_layers > 1), regimes are partitioned
        by ground-truth integrated seeing epsilon_0 and free-atmosphere fraction:
          - ``exquisite_seeing``: epsilon_0 <= Q_0.25 (bottom quartile — prime AO science windows)
          - ``nominal_seeing``: Q_0.25 < epsilon_0 <= Q_0.90 (standard operations)
          - ``turbulence_burst``: epsilon_0 > Q_0.90 (top decile extreme turbulence events)
          - ``free_atmos_dominated``: J_free / J_total > 0.50 (upper-tropospheric jet events)

        For scalar targets, regimes are ``quiescent`` (<= Q_0.25), ``nominal`` (Q_0.25..Q_0.90),
        and ``turbulence_burst`` (> Q_0.90).
        """
        from otbench.eval.utils import _get_valid_indices

        yt, yp = _get_valid_indices(y_true, y_pred)
        n_valid = len(yt)
        if n_valid == 0:
            return {}

        regime_masks: Dict[str, np.ndarray] = {}
        if yt.ndim == 2 and yt.shape[1] > 1:
            heights = self._build_metric_kwargs("integrated_seeing").get("heights")
            j_total, r0 = eval_metrics.profile_to_fried_parameter(yt, heights=heights)
            seeing_arcsec = (0.98 * 500e-9 / r0) * 206265.0
            q25, q90 = np.nanpercentile(seeing_arcsec, [25.0, 90.0])

            regime_masks["exquisite_seeing"] = seeing_arcsec <= q25
            regime_masks["nominal_seeing"] = (seeing_arcsec > q25) & (seeing_arcsec <= q90)
            regime_masks["turbulence_burst"] = seeing_arcsec > q90

            lin_yt = np.power(10.0, yt) if np.nanmedian(yt) < 0 else yt
            j_free = np.sum(lin_yt[:, 1:], axis=1)
            regime_masks["free_atmos_dominated"] = (j_free / np.clip(j_total, 1e-30, None)) > 0.50
        else:
            q25, q90 = np.nanpercentile(yt, [25.0, 90.0])
            regime_masks["quiescent"] = yt <= q25
            regime_masks["nominal"] = (yt > q25) & (yt <= q90)
            regime_masks["turbulence_burst"] = yt > q90

        results: dict = {}
        for regime_name, mask in regime_masks.items():
            count = int(np.sum(mask))
            if count == 0:
                continue
            results[regime_name] = {
                "n_samples":
                    count,
                "fraction":
                    float(count / n_valid),
                "metrics":
                    self._compute_metrics_dict(
                        yt[mask],
                        yp[mask],
                        eval_metric_names=eval_metric_names,
                        detailed_metrics=False,
                        compute_ci=compute_ci,
                        n_bootstrap=n_bootstrap,
                        ci=ci,
                        bootstrap_block_size=bootstrap_block_size,
                    ),
            }
        return results

    def _name_to_ds(self) -> str:
        """Return the dataset name from the task config."""
        return self.task["ds_name"]


class RegressionTask(BaseTask):

    def __init__(self, task_type: str, task_name: str, task: dict, benchmark_fp: str = BENCHMARK_FP) -> None:
        super().__init__(task_type=task_type, task_name=task_name, task=task, benchmark_fp=benchmark_fp)

    def evaluate_model(
        self,
        predict_call: Callable,
        predict_call_kwargs: Union[dict, None] = None,
        data_type: str = "pd",
        x_transforms: Union[Callable, None] = None,
        x_transform_kwargs: Union[dict, None] = None,
        eval_metric_names: Union[List[str], None] = None,
        return_predictions: bool = False,
        include_as_benchmark: bool = False,
        model_name: Union[str, None] = None,
        detailed_metrics: bool = False,
        overwrite: bool = True,
        compute_ci: bool = False,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        bootstrap_block_size: int = 1,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[Union[pd.DataFrame, pd.Series]] = None,
    ) -> Union[dict, Tuple[dict, 'np.ndarray']]:
        """Evaluate a model against this task's transformed test set, default against all metrics."""
        # obtain evaluation data
        if X_test is None or y_test is None:
            X_test, y_test = self.get_test_data(data_type=data_type)

        # Exclude rows where the ground truth is absent — metrics are undefined
        # on those rows and including them would silently corrupt aggregate
        # statistics (e.g. NaN propagation into RMSE).  The filtered count
        # matches bench_runner's possible_predictions exactly.
        if isinstance(y_test, pd.DataFrame):
            valid = y_test.notna().all(axis=1)
        else:
            valid = y_test.notna()
        X_test = X_test[valid]
        y_test = y_test[valid]

        # apply x_transforms if present
        if x_transforms is not None:
            if x_transform_kwargs is not None:
                X_test = x_transforms(X_test, **x_transform_kwargs)
            else:
                X_test = x_transforms(X_test)

        # get predictions
        if predict_call_kwargs is not None:
            y_test_pred = predict_call(X_test, **predict_call_kwargs)
        else:
            y_test_pred = predict_call(X_test)

        model_metrics = self._compute_metrics_dict(
            y_true=y_test,
            y_pred=y_test_pred,
            eval_metric_names=eval_metric_names,
            detailed_metrics=detailed_metrics,
            compute_ci=compute_ci,
            n_bootstrap=n_bootstrap,
            ci=ci,
            bootstrap_block_size=bootstrap_block_size,
        )

        if include_as_benchmark:
            self._add_experiment_to_benchmarks(model_name=model_name, model_metrics=model_metrics, overwrite=overwrite)
        if return_predictions:
            return model_metrics, y_test_pred
        return model_metrics


class ForecastingTask(BaseTask):

    def __init__(self, task_type: str, task_name: str, task: dict, benchmark_fp: str = BENCHMARK_FP) -> None:
        super().__init__(task_type=task_type, task_name=task_name, task=task, benchmark_fp=benchmark_fp)
        self.forecast_horizon = self.task["forecast_horizon"]
        self.window_size = self.task["window_size"]

    def prepare_forecasting_data(self,
                                 X: pd.DataFrame,
                                 y: Union[pd.DataFrame, pd.Series],
                                 window_size: Union[int, None] = None,
                                 forecast_horizon: Union[int, None] = None,
                                 lag_features: Optional[List[str]] = None):
        """
        Prepare data for forecasting, respecting session boundaries.

        When a session_col is set (e.g. night_id), lags are built *within each
        session* rather than across the full split.  This bounds peak memory to
        O(session_length × n_features × window_size) instead of
        O(N_total × n_features × window_size), and prevents lag values from
        bleeding across observing-night boundaries.

        Args:
            X: Input feature DataFrame.
            y: Target Series or DataFrame.
            window_size: Historical lookback window length (defaults to task window_size).
            forecast_horizon: Lead time in steps ahead (defaults to task forecast_horizon).
            lag_features: Optional subset of column names in ``X`` to lag across
                ``1 .. window_size - 1``. Columns not in ``lag_features`` remain
                contemporaneous only (t-0). Defaults to ``self.task.get("lag_features")``
                (None = lag all columns).
        """
        window_size = window_size if window_size is not None else self.window_size
        forecast_horizon = forecast_horizon if forecast_horizon is not None else self.forecast_horizon
        if lag_features is None:
            lag_features = self.task.get("lag_features")
        session_col = self.task.get("session_col")

        if session_col and session_col not in X.columns and len(X) > 0:
            try:
                ctx = self.get_dataset().get_context(X.index, session_col)
                X = X.join(ctx)
            except (KeyError, ValueError, NotImplementedError):
                pass

        # Fast path: when all features are lagged uniformly and all columns are numeric,
        # use the O(N*F) strided-window engine from prepare_sequence_data (~80x faster).
        feature_cols = [c for c in X.columns if c != session_col]
        y_df_check = y.to_frame() if isinstance(y, pd.Series) else y
        if (lag_features is None and window_size >= 1 and forecast_horizon >= 0 and len(feature_cols) > 0
                and all(pd.api.types.is_numeric_dtype(X[c].dtype) for c in feature_cols)
                and all(pd.api.types.is_numeric_dtype(y_df_check[c].dtype) for c in y_df_check.columns)):
            X_seq, y_seq, idx_seq = self.prepare_sequence_data(X,
                                                               y,
                                                               window_size=window_size,
                                                               forecast_horizon=forecast_horizon,
                                                               dropna=True)
            if len(X_seq) == 0:
                return pd.DataFrame(), pd.DataFrame()
            lag_col_names = list(feature_cols) + [
                f"{col} (t-{lag})" for lag in range(1, window_size) for col in feature_cols
            ]
            X_flat = X_seq[:, ::-1, :].reshape(len(X_seq), window_size * len(feature_cols))
            y_2d = y_seq[:, np.newaxis] if y_seq.ndim == 1 else y_seq
            X_out_df = pd.DataFrame(X_flat, index=idx_seq, columns=lag_col_names)
            y_out_df = pd.DataFrame(y_2d, index=idx_seq, columns=list(y_df_check.columns))
            return X_out_df, y_out_df

        if session_col and session_col in X.columns:
            # Build lags independently within each contiguous session block
            sessions = X[session_col]
            X = X.drop(columns=[session_col])

            X_parts: list = []
            y_parts: list = []

            for pos in sessions.groupby(sessions, sort=False).indices.values():
                X_sess = X.iloc[pos]
                y_sess = y.iloc[pos]

                if window_size > 1:
                    X_sess = self._add_lags(X_sess, window_size - 1, lag_features=lag_features)
                y_sess = self._shift_target(y_sess, forecast_horizon)

                combined = self._join_and_clean(X_sess, y_sess)
                if combined is None:
                    continue

                target_cols, original_y_cols = combined.attrs["_target_cols"], combined.attrs["_orig_y_cols"]
                y_out = combined[target_cols].copy()
                y_out.columns = original_y_cols
                X_parts.append(combined.drop(columns=target_cols))
                y_parts.append(y_out)

            if not X_parts:
                return pd.DataFrame(), pd.DataFrame()

            return pd.concat(X_parts), pd.concat(y_parts)

        else:
            # Contiguous split with no session boundary
            if window_size > 1:
                X = self._add_lags(X, window_size - 1, lag_features=lag_features)
            y = self._shift_target(y, forecast_horizon)

            combined = self._join_and_clean(X, y)
            if combined is None:
                return pd.DataFrame(), pd.DataFrame()

            target_cols, original_y_cols = combined.attrs["_target_cols"], combined.attrs["_orig_y_cols"]
            y_out = combined[target_cols].copy()
            y_out.columns = original_y_cols
            return combined.drop(columns=target_cols), y_out

    def prepare_sequence_data(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
        window_size: Optional[int] = None,
        forecast_horizon: Optional[int] = None,
        dropna: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
        """Construct 3D chronological sequence windows (N_valid, window_size, n_features) via zero-copy stride views.

        Respects ``session_col`` (e.g. ``night_id``) boundaries identically to
        :meth:`prepare_forecasting_data`, but avoids materializing an intermediate
        ``(N, n_features * window_size)`` 2D lagged DataFrame. Window step ``0`` corresponds
        to ``t - window_size + 1`` and step ``window_size - 1`` corresponds to ``t``.

        Returns:
            Tuple of ``(X_seq, y_seq, valid_index)`` where ``X_seq`` has shape
            ``(N_valid, window_size, n_features)``, ``y_seq`` has shape
            ``(N_valid, n_targets)`` (or ``(N_valid,)`` for 1D Series), and
            ``valid_index`` holds the anchor timestamps ``t`` of each valid window.
        """
        from numpy.lib.stride_tricks import sliding_window_view

        w = int(window_size if window_size is not None else self.window_size)
        h = int(forecast_horizon if forecast_horizon is not None else self.forecast_horizon)
        if w < 1:
            raise ValueError(f"window_size must be >= 1, got {w}.")
        if h < 0:
            raise ValueError(f"forecast_horizon must be >= 0, got {h}.")

        session_col = self.task.get("session_col")
        if session_col and session_col not in X.columns and len(X) > 0:
            try:
                ctx = self.get_dataset().get_context(X.index, session_col)
                X = X.join(ctx)
            except (KeyError, ValueError, NotImplementedError):
                pass

        is_series = isinstance(y, pd.Series)
        y_df = y.to_frame() if is_series else y

        if session_col and session_col in X.columns:
            sessions = X[session_col]
            X_clean = X.drop(columns=[session_col])
            group_positions = list(sessions.groupby(sessions, sort=False).indices.values())
        else:
            X_clean = X
            group_positions = [np.arange(len(X_clean))] if len(X_clean) > 0 else []

        n_features = X_clean.shape[1]
        n_targets = y_df.shape[1]
        if len(X_clean) == 0:
            empty_y = np.empty((0,), dtype=float) if is_series else np.empty((0, n_targets), dtype=float)
            return np.empty((0, w, n_features), dtype=float), empty_y, pd.Index([])

        X_all = X_clean.to_numpy(dtype=float)
        y_all = y_df.to_numpy(dtype=float)
        all_index = X_clean.index
        if dropna:
            x_row_valid_all = ~np.isnan(X_all).any(axis=1)
            y_row_valid_all = ~np.isnan(y_all).any(axis=1)

        X_chunks: list = []
        y_chunks: list = []
        idx_chunks: list = []

        for pos in group_positions:
            m = len(pos)
            if m < w + h:
                continue
            X_sess = X_all[pos]
            y_sess = y_all[pos]
            sess_index = all_index[pos]

            # sliding_window_view along axis=0 yields shape (n_win, n_features, w) -> swap to (n_win, w, n_features)
            X_win = np.swapaxes(sliding_window_view(X_sess[:m - h], window_shape=w, axis=0), 1, 2)
            y_win = y_sess[w - 1 + h:]
            idx_win = sess_index[w - 1:m - h]

            if dropna:
                x_win_valid = sliding_window_view(x_row_valid_all[pos][:m - h], window_shape=w, axis=0).all(axis=-1)
                y_win_valid = y_row_valid_all[pos][w - 1 + h:]
                valid = x_win_valid & y_win_valid
                if not np.any(valid):
                    continue
                X_win = X_win[valid]
                y_win = y_win[valid]
                idx_win = idx_win[valid]

            X_chunks.append(np.ascontiguousarray(X_win))
            y_chunks.append(y_win)
            idx_chunks.append(idx_win)

        if not X_chunks:
            empty_y = np.empty((0,), dtype=float) if is_series else np.empty((0, n_targets), dtype=float)
            return np.empty((0, w, n_features), dtype=float), empty_y, pd.Index([])

        X_out = np.concatenate(X_chunks, axis=0)
        y_out = np.concatenate(y_chunks, axis=0)
        if is_series:
            y_out = y_out.ravel()
        idx_out = idx_chunks[0].append(idx_chunks[1:]) if len(idx_chunks) > 1 else idx_chunks[0]
        return X_out, y_out, idx_out

    def _join_and_clean(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, None]:
        """
        Collision-safe join of X and shifted y, followed by dropna.
        Returns the combined DataFrame with target column names stored in
        .attrs so the caller can split them back out, or None if empty.
        """
        if isinstance(y, pd.Series):
            y_temp = y.to_frame()
        else:
            y_temp = y.copy()

        temp_suffix = "_TEMP_TARGET_XYZ"
        original_y_cols = y_temp.columns.tolist()
        y_temp.columns = [str(c) + temp_suffix for c in y_temp.columns]

        combined = X.join(y_temp).dropna()
        if combined.empty:
            return None

        combined.attrs["_target_cols"] = y_temp.columns.tolist()
        combined.attrs["_orig_y_cols"] = original_y_cols
        return combined

    def evaluate_model(
        self,
        predict_call: Callable,
        predict_call_kwargs: Union[dict, None] = None,
        window_size: Union[int, None] = None,
        forecast_horizon: Union[int, None] = None,
        data_type: str = "pd",
        x_transforms: Union[Callable, None] = None,
        x_transform_kwargs: Union[dict, None] = None,
        forecast_transforms: Union[Callable, None] = None,
        forecast_transform_kwargs: Union[dict, None] = None,
        eval_metric_names: Union[List[str], None] = None,
        return_predictions: bool = False,
        include_as_benchmark: bool = False,
        model_name: Union[str, None] = None,
        detailed_metrics: bool = False,
        overwrite: bool = True,
        compute_ci: bool = False,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        bootstrap_block_size: int = 1,
        use_sequence_data: bool = False,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[Union[pd.DataFrame, pd.Series]] = None,
    ) -> Union[dict, Tuple[dict, 'np.ndarray']]:
        """Evaluate a model against this task's transformed test set, default against all metrics."""
        window_size = window_size if window_size is not None else self.window_size
        forecast_horizon = forecast_horizon if forecast_horizon is not None else self.forecast_horizon

        # obtain evaluation data
        if X_test is None or y_test is None:
            X_test, y_test = self.get_test_data(data_type=data_type)

        assert forecast_horizon + window_size < len(
            X_test
        ), f"window_size and forecast_horizon must be less than the length of the evaluation set ({len(X_test)})."

        # apply x_transforms if present
        if x_transforms is not None:
            if x_transform_kwargs is not None:
                X_test = x_transforms(X_test, **x_transform_kwargs)
            else:
                X_test = x_transforms(X_test)

        # apply window and forecasting horizon transforms
        if forecast_transforms is not None:
            if forecast_transform_kwargs is not None:
                y_test = forecast_transforms(y_test, **forecast_transform_kwargs)
            else:
                y_test = forecast_transforms(y_test)

        # obtain forecasting data (automatically recovers session_col if needed)
        if use_sequence_data:
            X_test, y_test, _ = self.prepare_sequence_data(X_test, y_test, window_size, forecast_horizon)
        else:
            X_test, y_test = self.prepare_forecasting_data(X_test, y_test, window_size, forecast_horizon)

        # get predictions
        if predict_call_kwargs is not None:
            y_test_pred = predict_call(X_test, **predict_call_kwargs)
        else:
            y_test_pred = predict_call(X_test)

        model_metrics = self._compute_metrics_dict(
            y_true=y_test,
            y_pred=y_test_pred,
            eval_metric_names=eval_metric_names,
            detailed_metrics=detailed_metrics,
            compute_ci=compute_ci,
            n_bootstrap=n_bootstrap,
            ci=ci,
            bootstrap_block_size=bootstrap_block_size,
        )

        if include_as_benchmark:
            self._add_experiment_to_benchmarks(model_name=model_name, model_metrics=model_metrics, overwrite=overwrite)
        if return_predictions:
            return model_metrics, y_test_pred
        return model_metrics

    def evaluate_horizon_curve(
        self,
        models: Dict[str, Callable],
        horizons: Sequence[int] = (1, 2, 5, 10, 15, 30),
        window_size: Optional[int] = None,
        eval_metric_names: Optional[List[str]] = None,
        reference_model: Optional[str] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[Union[pd.DataFrame, pd.Series]] = None,
    ) -> pd.DataFrame:
        """Evaluate one or more forecasting models across multiple lead times H to map predictability decay.

        Args:
            models: Mapping of ``model_name -> callable_or_model``. Each entry may be:
                - A ``predict_call(X_windowed)`` callable (for horizon-invariant or direct predictors), or
                - A ``(train_and_predict_fn(horizon, X_windowed, y_windowed) -> y_pred)`` callable, or
                - An object exposing ``.predict(X_windowed)``.
            horizons: Sequence of forecast lead times (in steps) to evaluate.
            window_size: Lookback window size (defaults to ``self.window_size``).
            eval_metric_names: Metrics to compute at each horizon (defaults to ``self.task["eval_metrics"]``).
            reference_model: Optional key in ``models`` (e.g. ``"Persistence"``) used to compute
                ``skill_score = 1 - (rmse_model / rmse_ref)^2`` at each horizon.
            X_test: Optional pre-loaded test features (defaults to ``self.get_test_data()``).
            y_test: Optional pre-loaded test targets.

        Returns:
            Tidy ``pd.DataFrame`` with one row per ``(horizon, model)`` containing scalar metric values
            and ``skill_score`` (when ``reference_model`` is present in ``models``).
        """
        w = window_size if window_size is not None else self.window_size
        if eval_metric_names is None:
            eval_metric_names = self.task["eval_metrics"]
        if X_test is None or y_test is None:
            X_test, y_test = self.get_test_data(data_type="pd")

        rows: list = []
        for h in horizons:
            X_h, y_h = self.prepare_forecasting_data(X_test.copy(), y_test.copy(), window_size=w, forecast_horizon=h)
            if len(X_h) == 0:
                continue
            for model_name, mdl in models.items():
                predict_fn = mdl.predict if hasattr(mdl, "predict") else mdl
                y_pred = predict_fn(X_h)
                m_dict = self._compute_metrics_dict(
                    y_true=y_h,
                    y_pred=y_pred,
                    eval_metric_names=eval_metric_names,
                    detailed_metrics=False,
                )
                row: dict = {
                    "horizon":
                        int(h),
                    "model":
                        model_name,
                    "valid_predictions":
                        int(next(
                            (v["valid_predictions"] for v in m_dict.values() if isinstance(v, dict)),
                            len(y_h),
                        )),
                }
                for m_name, m_val in m_dict.items():
                    if isinstance(m_val, dict):
                        row[m_name] = float(m_val["metric_value"])
                        if "per_layer" in m_val:
                            for layer_k, layer_v in m_val["per_layer"].items():
                                row[f"rmse_{layer_k}"] = float(layer_v)
                rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty and reference_model and "root_mean_square_error" in df.columns:
            ref_df = df[df["model"] == reference_model].set_index("horizon")["root_mean_square_error"]
            ref_rmse = df["horizon"].map(ref_df)
            df["skill_score"] = 1.0 - (df["root_mean_square_error"]**2) / np.clip(ref_rmse**2, 1e-30, None)
        return df

    def _add_lags(
        self,
        X: pd.DataFrame,
        n_lags: int,
        lag_features: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Add lag columns (1 … n_lags inclusive) for all features or a selective subset."""
        if n_lags < 1:
            return X

        cols_to_lag = ([c for c in X.columns if c in lag_features] if lag_features is not None else list(X.columns))
        if not cols_to_lag:
            return X

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            lag_dict = {f'{col} (t-{lag})': X[col].shift(lag) for lag in range(1, n_lags + 1) for col in cols_to_lag}
            X = X.assign(**lag_dict)

        return X

    def _shift_target(self, y, forecast_horizon):
        """Shift the target by the forecast horizon."""
        if forecast_horizon > 0:
            y = y.shift(-forecast_horizon)

        return y


class TaskApi(object):
    """A factory for creating optical turbulence modeling tasks."""

    def __init__(self, root_dir: Union[str, None] = None) -> None:
        """Read the currently-supported benchmarking task for loaders and evaluators."""
        if root_dir is None:
            root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))  # obtain root path
        tasks_path = os.path.join(root_dir, 'config', 'tasks.json')
        with open(tasks_path, 'r') as f:
            tasks = json.load(f)

        self.tasks = tasks

        if settings.USE_SYNTHETIC_DATA:
            self._patch_tasks_for_synthetic_data(self.tasks)

        self._build_task_names()

    def _patch_tasks_for_synthetic_data(self, d: dict) -> None:
        """Recursively patch task indices to fit within the small synthetic dataset."""
        if isinstance(d, dict):
            if "train_idx" in d and "val_idx" in d and "test_idx" in d:
                # Found a task definition, patch it
                # Ensure these indices fit within the 2000-step synthetic dataset
                d["train_idx"] = ["0:1000"]
                d["val_idx"] = ["1000:1500"]
                d["test_idx"] = ["1500:2000"]
            else:
                # Recurse
                for k, v in d.items():
                    self._patch_tasks_for_synthetic_data(v)

    def get_task(self, task_name: str, benchmark_fp: str = BENCHMARK_FP) -> Union[RegressionTask, ForecastingTask]:
        """Get a task by name."""
        if self._is_supported_task(task_name=task_name):
            if task_name.split(".")[0] == TaskTypes.REGRESSION.value:
                return RegressionTask(task_type=TaskTypes.REGRESSION,
                                      task_name=task_name,
                                      task=self._get_task(key=task_name),
                                      benchmark_fp=benchmark_fp)
            elif task_name.split(".")[0] == TaskTypes.FORECASTING.value:
                return ForecastingTask(task_type=TaskTypes.FORECASTING,
                                       task_name=task_name,
                                       task=self._get_task(key=task_name),
                                       benchmark_fp=benchmark_fp)
            else:
                raise NotImplementedError(f"task type {task_name.split('.')[0]} not supported.")
        else:
            raise NotImplementedError(f"task {task_name} not supported.")

    def list_tasks(self) -> List[str]:
        """List all currently supported tasks."""
        return sorted(list(self.task_names))

    def _get_task(self, key: str) -> dict:
        """Traverse a task key, returning bottom-level data."""
        path = key.split(".")
        d = self.tasks.copy()
        for i in range(len(path)):
            d = d[path[i]]
        return d

    def _is_supported_task(self, task_name: str) -> bool:
        """Check whether a task is supported by otbench."""
        return task_name in self.task_names

    def _build_task_names(self) -> None:
        """Gather `.` separated task names for easier inclusion checking."""
        task_names = set()
        task_paths = [(t, t, v) for t, v in self.tasks.items()]
        next_paths = []

        while task_paths or next_paths:
            if not task_paths:
                task_paths = next_paths
                next_paths = []

            _, full_k, v = task_paths.pop()

            if 'description' in v:
                task_names.add(full_k)
            else:
                next_paths.extend([(nk, f'{full_k}.{nk}', nv) for nk, nv in v.items()])

        self.task_names = task_names

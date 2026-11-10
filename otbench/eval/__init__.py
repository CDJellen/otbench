"""Evaluation metrics for optical turbulence benchmarks.

Standard regression metrics:
    coefficient_of_determination, root_mean_square_error,
    mean_absolute_error, mean_absolute_percentage_error

Profile-aware (AO-derived) metrics and utilities:
    integrated_seeing          — Fried parameter -> FWHM (arcsec)
    isoplanatic_angle          — AO correction field of view theta_0 (arcsec)
    coherence_time             — AO loop timing tau_0 (ms)
    greenwood_frequency        — AO bandwidth requirement f_G (Hz)
    per_layer_rmse             — Per-layer error decomposition
    strehl_ratio               — Closed-loop Marechal Strehl ratio & OPD error
    ao_wavefront_error_budget  — Fitting, anisoplanatic, and servo-lag variances
    profile_to_fried_parameter — Shared J_total and r_0 helper
    bootstrap_ci               — Moving-block bootstrap confidence interval
    paired_bootstrap_test      — Paired moving-block statistical significance test

Profile metrics accept additional kwargs (heights, layer_names, wind_speed)
which are injected automatically by ``BaseTask._build_metric_kwargs`` when
the metric name appears in a task's ``eval_metrics`` list.
"""
from otbench.eval.metrics import (
    MetricSpec,
    get_metric_spec,
    is_implemented_metric,
    is_lower_better,
    coefficient_of_determination,
    root_mean_square_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    integrated_seeing,
    isoplanatic_angle,
    coherence_time,
    greenwood_frequency,
    per_layer_rmse,
    profile_to_fried_parameter,
    ao_wavefront_error_budget,
    strehl_ratio,
    ground_layer_fraction,
    bootstrap_ci,
    paired_bootstrap_test,
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

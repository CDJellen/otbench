"""Evaluation metrics for optical turbulence benchmarks.

Standard regression metrics:
    coefficient_of_determination, root_mean_square_error,
    mean_absolute_error, mean_absolute_percentage_error

Profile-aware (AO-derived) metrics:
    integrated_seeing     — Fried parameter → FWHM (arcsec)
    isoplanatic_angle     — AO correction field of view θ₀ (arcsec)
    coherence_time        — AO loop timing τ₀ (ms)
    greenwood_frequency   — AO bandwidth requirement f_G (Hz)
    per_layer_rmse        — Per-layer error decomposition

Profile metrics accept additional kwargs (heights, layer_names, wind_speed)
which are injected automatically by ``BaseTask._build_metric_kwargs`` when
the metric name appears in a task's ``eval_metrics`` list.

See :mod:`otbench.eval.integrated_metrics` for the physics implementations.
"""
from .metrics import *

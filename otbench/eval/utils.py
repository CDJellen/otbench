from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def _format_metric(metric_value: float, valid_predictions: int) -> dict:
    """Format the metric value and valid predictions into a dict."""
    return {"metric_value": metric_value, "valid_predictions": valid_predictions}


def _get_valid_indices(y_true: Sequence, y_pred: Sequence) -> Tuple[Sequence, Sequence]:
    """
    Get the valid indices for the supplied sequences, handling scalar and vector targets robustly.
    
    This function:
    - Converts inputs to numpy arrays
    - Normalizes (N, 1) shapes to (N,) for scalar targets
    - Ensures y_true and y_pred have compatible shapes
    - Creates a mask excluding rows where any value is NaN (for both scalar and vector targets)
    - Returns masked arrays of identical shape
    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Fast path for empty inputs
    if len(y_true) == 0:
        return np.array([]), np.array([])

    # Normalise common (N, 1) -> (N,) cases
    if y_true.ndim == 2 and y_true.shape[1] == 1:
        y_true = y_true.ravel()
    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.ravel()

    # After normalisation, shapes must match
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have compatible shapes after normalisation. "
                         f"Got y_true.shape = {y_true.shape} and y_pred.shape = {y_pred.shape}")

    # Determine masking strategy based on ground truth dimensionality
    if y_true.ndim == 1:
        # Scalar target: drop samples where either value is NaN
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    elif y_true.ndim == 2:
        # Vector target: drop samples where any value in the row is NaN
        mask = ~np.isnan(y_true).any(axis=1) & ~np.isnan(y_pred).any(axis=1)
    else:
        raise ValueError(f"Unsupported dimensionality for y_true: ndim={y_true.ndim}. "
                         f"Expected 1 (scalar) or 2 (vector) dimensions.")

    return y_true[mask], y_pred[mask]


def _sample_bootstrap_indices(
    rng: np.random.Generator,
    n: int,
    batch_b: int,
    block_size: int = 1,
) -> np.ndarray:
    """Generate a (batch_b, n) matrix of i.i.d. or moving-block bootstrap indices without Python loops."""
    b = max(int(block_size), 1)
    if b <= 1 or n <= b:
        return rng.integers(0, n, size=(batch_b, n))
    n_blocks = int(np.ceil(n / b))
    starts = rng.integers(0, n - b + 1, size=(batch_b, n_blocks, 1))
    offsets = np.arange(b, dtype=int)[np.newaxis, np.newaxis, :]
    return (starts + offsets).reshape(batch_b, n_blocks * b)[:, :n]


def _reduce_sample_losses(losses_batch: np.ndarray, reduction: str) -> np.ndarray:
    """Reduce a batch of resampled per-sample losses of shape (B, N) or (B, N, L) to (B,) metric values."""
    if reduction in ("rmse", "sqrt_mean"):
        # losses_batch holds squared errors: (B, N) -> sqrt(mean_N); (B, N, L) -> mean_L(sqrt(mean_N))
        mse = np.mean(losses_batch, axis=1)
        rmse = np.sqrt(mse)
        return np.mean(rmse, axis=-1) if rmse.ndim > 1 else rmse
    elif reduction in ("mae", "mean"):
        mean_val = np.mean(losses_batch, axis=1)
        return np.mean(mean_val, axis=-1) if mean_val.ndim > 1 else mean_val
    raise ValueError(f"Unsupported bootstrap loss reduction '{reduction}'.")


def _bootstrap_from_sample_losses(
    sample_losses: np.ndarray,
    reduction: str = "rmse",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    block_size: int = 1,
    chunk_size: int = 128,
) -> Tuple[float, float]:
    """Vectorized bootstrap CI over pre-computed per-sample losses (100x faster than re-evaluating metric_fn)."""
    n = len(sample_losses)
    if n == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    scores = np.empty(n_bootstrap, dtype=float)
    for start in range(0, n_bootstrap, chunk_size):
        end = min(start + chunk_size, n_bootstrap)
        idx = _sample_bootstrap_indices(rng, n, end - start, block_size=block_size)
        scores[start:end] = _reduce_sample_losses(sample_losses[idx], reduction=reduction)

    alpha = (1.0 - ci) / 2.0
    return float(np.nanpercentile(scores, 100 * alpha)), float(np.nanpercentile(scores, 100 * (1.0 - alpha)))


def _bootstrap_ci(
    y_true: Sequence,
    y_pred: Sequence,
    metric_fn: Optional[Callable] = None,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    block_size: int = 1,
    sample_losses: Optional[np.ndarray] = None,
    reduction: str = "rmse",
) -> Tuple[float, float]:
    """Compute non-parametric (i.i.d. or moving-block) bootstrap confidence interval for a metric.

    When ``sample_losses`` is provided (e.g. per-sample squared errors for RMSE or AO metrics),
    uses chunked vectorized NumPy indexing (~100x faster on large test splits). Otherwise falls
    back to evaluating ``metric_fn(yt[idx], yp[idx])`` on each resample.

    Args:
        y_true: Ground truth array or sequence.
        y_pred: Prediction array or sequence.
        metric_fn: Callable(y_true, y_pred) -> float or dict containing 'metric_value'.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (default 0.95 for 95% CI).
        seed: Random seed for reproducibility.
        block_size: Contiguous block length for moving-block bootstrap on
            autocorrelated time series (default 1 = standard i.i.d. bootstrap).
        sample_losses: Optional pre-computed per-sample loss array of length N_valid.
        reduction: Reduction mode when ``sample_losses`` is supplied (``'rmse'`` or ``'mean'``).

    Returns:
        (ci_lower, ci_upper) as floats.
    """
    if sample_losses is not None:
        return _bootstrap_from_sample_losses(
            np.asarray(sample_losses, dtype=float),
            reduction=reduction,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
            block_size=block_size,
        )

    if metric_fn is None:
        raise ValueError("Either metric_fn or sample_losses must be provided to bootstrap_ci.")

    yt, yp = _get_valid_indices(y_true, y_pred)
    n = len(yt)
    if n == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    scores = np.empty(n_bootstrap, dtype=float)
    idx_all = _sample_bootstrap_indices(rng, n, n_bootstrap, block_size=block_size)

    for i in range(n_bootstrap):
        idx = idx_all[i]
        val = metric_fn(yt[idx], yp[idx])
        scores[i] = float(val["metric_value"] if isinstance(val, dict) else val)

    alpha = (1.0 - ci) / 2.0
    return float(np.nanpercentile(scores, 100 * alpha)), float(np.nanpercentile(scores, 100 * (1.0 - alpha)))


bootstrap_ci = _bootstrap_ci


def paired_bootstrap_test(
    y_true: Sequence,
    y_pred_a: Sequence,
    y_pred_b: Sequence,
    metric_fn: Optional[Callable] = None,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    block_size: int = 1,
    lower_is_better: bool = True,
    sample_losses_a: Optional[np.ndarray] = None,
    sample_losses_b: Optional[np.ndarray] = None,
    reduction: str = "rmse",
) -> dict:
    """Paired (i.i.d. or moving-block) bootstrap hypothesis test comparing Model A vs. Model B.

    Evaluates both models on identical resampled timestamps ``I_b`` to preserve
    cross-model covariance across shared atmospheric conditions.

    Args:
        y_true: Ground truth array or sequence.
        y_pred_a: Predictions from Model A.
        y_pred_b: Predictions from Model B (reference/baseline).
        metric_fn: Callable(y_true, y_pred) -> float or dict containing 'metric_value'.
            Defaults to root_mean_square_error if None.
        n_bootstrap: Number of paired bootstrap resamples (default 1000).
        ci: Confidence level (default 0.95).
        seed: Random seed for reproducibility.
        block_size: Moving-block length for autocorrelated residuals (default 1).
        lower_is_better: If True (default for error metrics), negative ``delta``
            indicates Model A outperforms Model B.

    Returns:
        Dictionary with keys: ``score_a``, ``score_b``, ``delta`` (score_a - score_b),
        ``ci_lower``, ``ci_upper``, ``p_value`` (two-sided), ``significant`` (bool),
        ``a_is_better`` (bool), and ``valid_predictions`` (int).
    """
    yt = np.asarray(y_true, dtype=float)
    ya = np.asarray(y_pred_a, dtype=float)
    yb = np.asarray(y_pred_b, dtype=float)
    if yt.ndim == 2 and yt.shape[1] == 1:
        yt = yt.ravel()
    if ya.ndim == 2 and ya.shape[1] == 1:
        ya = ya.ravel()
    if yb.ndim == 2 and yb.shape[1] == 1:
        yb = yb.ravel()
    if yt.shape != ya.shape or yt.shape != yb.shape:
        raise ValueError(f"Incompatible shapes for paired test: {yt.shape}, {ya.shape}, {yb.shape}.")

    if yt.ndim == 1:
        mask = ~np.isnan(yt) & ~np.isnan(ya) & ~np.isnan(yb)
    else:
        mask = ~np.isnan(yt).any(axis=1) & ~np.isnan(ya).any(axis=1) & ~np.isnan(yb).any(axis=1)

    yt_m, ya_m, yb_m = yt[mask], ya[mask], yb[mask]
    n = len(yt_m)
    if n == 0:
        return {
            "score_a": float("nan"),
            "score_b": float("nan"),
            "delta": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "p_value": float("nan"),
            "significant": False,
            "a_is_better": False,
            "valid_predictions": 0,
        }

    rng = np.random.default_rng(seed)
    if sample_losses_a is not None and sample_losses_b is not None:
        la = np.asarray(sample_losses_a, dtype=float)
        lb = np.asarray(sample_losses_b, dtype=float)
        n = len(la)
        score_a = float(_reduce_sample_losses(la[np.newaxis, ...], reduction=reduction)[0])
        score_b = float(_reduce_sample_losses(lb[np.newaxis, ...], reduction=reduction)[0])
        deltas = np.empty(n_bootstrap, dtype=float)
        chunk_size = 128
        for start in range(0, n_bootstrap, chunk_size):
            end = min(start + chunk_size, n_bootstrap)
            idx = _sample_bootstrap_indices(rng, n, end - start, block_size=block_size)
            deltas[start:end] = _reduce_sample_losses(la[idx], reduction=reduction) - _reduce_sample_losses(
                lb[idx], reduction=reduction)
    elif metric_fn is None:
        # Vectorized fast path for RMSE
        sq_a = (yt_m - ya_m)**2
        sq_b = (yt_m - yb_m)**2
        score_a = float(np.mean(np.sqrt(np.mean(sq_a, axis=0))))
        score_b = float(np.mean(np.sqrt(np.mean(sq_b, axis=0))))
        deltas = np.empty(n_bootstrap, dtype=float)
        chunk_size = 128
        for start in range(0, n_bootstrap, chunk_size):
            end = min(start + chunk_size, n_bootstrap)
            idx = _sample_bootstrap_indices(rng, n, end - start, block_size=block_size)
            deltas[start:end] = _reduce_sample_losses(sq_a[idx], "rmse") - _reduce_sample_losses(sq_b[idx], "rmse")
    else:

        def _eval_scalar(fn, y_t, y_p):
            v = fn(y_t, y_p)
            return float(v["metric_value"] if isinstance(v, dict) else v)

        score_a = _eval_scalar(metric_fn, yt_m, ya_m)
        score_b = _eval_scalar(metric_fn, yt_m, yb_m)
        idx_all = _sample_bootstrap_indices(rng, n, n_bootstrap, block_size=block_size)
        deltas = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            idx = idx_all[i]
            deltas[i] = _eval_scalar(metric_fn, yt_m[idx], ya_m[idx]) - _eval_scalar(metric_fn, yt_m[idx], yb_m[idx])

    delta_point = float(score_a - score_b)
    alpha = (1.0 - ci) / 2.0
    ci_low = float(np.nanpercentile(deltas, 100 * alpha))
    ci_high = float(np.nanpercentile(deltas, 100 * (1.0 - alpha)))
    p_neg = float(np.mean(deltas <= 0.0))
    p_pos = float(np.mean(deltas >= 0.0))
    p_val = float(min(1.0, 2.0 * min(p_neg, p_pos)))
    significant = bool(ci_high < 0.0 or ci_low > 0.0)
    a_is_better = bool(delta_point < 0.0 if lower_is_better else delta_point > 0.0)

    return {
        "score_a": score_a,
        "score_b": score_b,
        "delta": delta_point,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "p_value": p_val,
        "significant": significant,
        "a_is_better": a_is_better,
        "valid_predictions": int(n),
    }

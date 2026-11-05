import json

import pytest

from otbench.benchmark.bench_runner import run_benchmarks
from tests import TESTS_BENCHMARK_FP


@pytest.mark.slow
def test_run_benchmark_single():
    """Test running benchmarks."""
    # save the current expiraments.json
    print(TESTS_BENCHMARK_FP)
    with open(TESTS_BENCHMARK_FP, "r") as fp:
        experiments = json.load(fp)
    got = run_benchmarks(benchmark_tasks="regression.mlo_cn2.dropna.Cn2_15m",
                         verbose=True,
                         write_metrics=True,
                         metrics_fp=TESTS_BENCHMARK_FP,
                         include_pytorch_models=True,
                         n_epochs_override=10)
    # restore the original expiraments.json
    with open(TESTS_BENCHMARK_FP, "w") as fp:
        json.dump(experiments, fp, indent=4)
    assert isinstance(got, dict)


@pytest.mark.slow
def test_run_benchmarks():
    """Test running benchmarks."""
    got = run_benchmarks(benchmark_tasks=["regression.mlo_cn2.dropna.Cn2_15m", "regression.mlo_cn2.full.Cn2_15m"],
                         verbose=True,
                         write_metrics=False,
                         include_pytorch_models=False)

    assert isinstance(got, dict)
    assert len(got) == 2
    assert "regression.mlo_cn2.dropna.Cn2_15m" in got
    assert "regression.mlo_cn2.full.Cn2_15m" in got


@pytest.mark.slow
def test_run_benchmarks_all_single_model():
    """Test running benchmarks."""
    got = run_benchmarks(benchmark_tasks=None,
                         verbose=True,
                         benchmark_regression_models="PersistenceRegressionModel",
                         benchmark_forecasting_models="PersistenceForecastingModel",
                         write_metrics=False,
                         include_pytorch_models=True)

    assert isinstance(got, dict)


@pytest.mark.slow
def test_run_benchmarks_all_multiple_models():
    """Test running benchmarks."""
    got = run_benchmarks(benchmark_tasks=None,
                         verbose=True,
                         benchmark_regression_models=["PersistenceRegressionModel"],
                         benchmark_forecasting_models=["PersistenceForecastingModel"],
                         write_metrics=False,
                         include_pytorch_models=True)

    assert isinstance(got, dict)


@pytest.mark.slow
def test_run_benchmark_vector():
    """Test running benchmarks for vector task (Paranal Profile)."""
    # This task produces vector outputs.
    # We expect bench_runner to filter scalar-only models and handle JSON serialization of vector results.
    got = run_benchmarks(benchmark_tasks="regression.paranal_tomography.full.cn2_profile_reconstruction",
                         verbose=True,
                         write_metrics=False,
                         include_pytorch_models=False)

    assert isinstance(got, dict)
    assert "regression.paranal_tomography.full.cn2_profile_reconstruction" in got
    results = got["regression.paranal_tomography.full.cn2_profile_reconstruction"]

    # Check that compatible models ran
    assert "ClimatologyRegressionModel" in results
    assert "PersistenceRegressionModel" in results
    # Check that incompatible models were filtered (not present)
    assert "MacroMeteorologicalModel" not in results


def test_run_benchmarks_write_metrics_merges_existing_tasks(tmp_path):
    """write_metrics=True must preserve existing tasks in metrics_fp when running a subset of tasks."""
    metrics_fp = tmp_path / "experiments.json"
    initial = {
        "existing.sentinel.task": {
            "possible_predictions": 100,
            "SentinelModel": {
                "root_mean_square_error": {
                    "metric_value": 0.42,
                    "valid_predictions": 100
                }
            },
        }
    }
    metrics_fp.write_text(json.dumps(initial))

    got = run_benchmarks(
        benchmark_tasks="regression.mlo_cn2.dropna.Cn2_15m",
        benchmark_regression_models="PersistenceRegressionModel",
        verbose=False,
        write_metrics=True,
        metrics_fp=metrics_fp,
        include_pytorch_models=False,
    )
    assert "regression.mlo_cn2.dropna.Cn2_15m" in got
    persisted = json.loads(metrics_fp.read_text())
    assert "existing.sentinel.task" in persisted
    assert "regression.mlo_cn2.dropna.Cn2_15m" in persisted

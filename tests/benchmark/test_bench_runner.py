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
    got = run_benchmarks(benchmark_tasks="regression.mlo_cn2.dropna.Cn2_15m", verbose=True, write_metrics=True, metrics_fp=TESTS_BENCHMARK_FP, include_pytorch_models=True, n_epochs_override=10)
    # restore the original expiraments.json
    with open(TESTS_BENCHMARK_FP, "w") as fp:
        json.dump(experiments, fp, indent=4)
    assert isinstance(got, dict)


@pytest.mark.slow
def test_run_benchmarks():
    """Test running benchmarks."""
    got = run_benchmarks(benchmark_tasks=["regression.mlo_cn2.dropna.Cn2_15m", "regression.mlo_cn2.full.Cn2_15m"], verbose=True, write_metrics=False, include_pytorch_models=False)

    assert isinstance(got, dict)
    assert len(got) == 2
    assert "regression.mlo_cn2.dropna.Cn2_15m" in got
    assert "regression.mlo_cn2.full.Cn2_15m" in got


@pytest.mark.slow
def test_run_benchmarks_all_single_model():
    """Test running benchmarks."""
    got = run_benchmarks(benchmark_tasks=None, verbose=True, benchmark_regression_models="PersistenceRegressionModel", benchmark_forecasting_models="PersistenceForecastingModel", write_metrics=False, include_pytorch_models=True)

    assert isinstance(got, dict)

@pytest.mark.slow
def test_run_benchmarks_all_multiple_models():
    """Test running benchmarks."""
    got = run_benchmarks(benchmark_tasks=None, verbose=True, benchmark_regression_models=["PersistenceRegressionModel"], benchmark_forecasting_models=["PersistenceForecastingModel"], write_metrics=False, include_pytorch_models=True)

    assert isinstance(got, dict)

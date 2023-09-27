import json

import pytest

from otb.benchmark.bench_runner import run_benchmarks
from tests import TESTS_BENCHMARK_FP


@pytest.mark.slow
def test_run_benchmarks():
    """Test running benchmarks."""
    # save the current expiraments.json
    print(TESTS_BENCHMARK_FP)
    with open(TESTS_BENCHMARK_FP, "r") as fp:
        experiments = json.load(fp)
    got = run_benchmarks(verbose=True, write_metrics=True, metrics_fp=TESTS_BENCHMARK_FP)
    # restore the original expiraments.json
    with open(TESTS_BENCHMARK_FP, "w") as fp:
        json.dump(experiments, fp, indent=4)
    assert isinstance(got, dict)

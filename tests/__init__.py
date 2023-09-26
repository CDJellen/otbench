from os import path

from otb.config import ROOT_DIR


TESTS_DIR = path.abspath(path.dirname(__file__))
TESTS_BENCHMARK_FP = path.join(TESTS_DIR, "benchmark", "experiments.json")

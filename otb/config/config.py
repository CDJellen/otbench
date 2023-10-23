import os

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_FP = os.path.join(CONFIG_DIR, "tasks.json")
DATASETS_FP = os.path.join(CONFIG_DIR, "datasets.json")
ROOT_DIR = os.path.dirname(CONFIG_DIR)
CACHE_DIR = os.path.join(ROOT_DIR, "cache", "processed")
DATA_DIR = os.path.join(ROOT_DIR, "data")
BENCHMARK_FP = os.path.join(ROOT_DIR, "benchmark", "experiments.json")
RETURN_TYPES = ["pd", "np", "xr", "nc"]

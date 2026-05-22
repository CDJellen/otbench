from .settings import settings

# Re-exports for backward compatibility.  Internal and external code that
# imports ``from otbench.config import CACHE_DIR`` etc. continues to work.
# New code should import ``settings`` directly.
CONFIG_DIR = str(settings.CONFIG_DIR)
TASKS_FP = str(settings.TASKS_FP)
DATASETS_FP = str(settings.DATASETS_FP)
ROOT_DIR = str(settings.ROOT_DIR)
CACHE_DIR = str(settings.CACHE_DIR)
DATA_DIR = str(settings.DATA_DIR)
BENCHMARK_FP = str(settings.BENCHMARK_FP)
RETURN_TYPES = settings.RETURN_TYPES

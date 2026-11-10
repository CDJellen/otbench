from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Global configuration for otbench using Pydantic.
    """
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent
    CONFIG_DIR: Path = Path(__file__).resolve().parent

    # Deriving paths relative to ROOT_DIR or CONFIG_DIR
    # Note: Pydantic will populate fields from env vars if available, e.g. OTBENCH_DATA_DIR
    DATA_DIR: Path = ROOT_DIR / "data"
    CACHE_DIR: Path = ROOT_DIR / "cache" / "processed"
    BENCHMARK_FP: Path = ROOT_DIR / "benchmark" / "experiments.json"

    TASKS_FP: Path = CONFIG_DIR / "tasks.json"
    DATASETS_FP: Path = CONFIG_DIR / "datasets.json"

    RETURN_TYPES: list[str] = ["pd", "np", "xr", "nc"]
    USE_SYNTHETIC_DATA: bool = False

    model_config = SettingsConfigDict(env_prefix="OTBENCH_")


settings = Settings()

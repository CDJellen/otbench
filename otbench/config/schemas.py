from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class Task(BaseModel):
    """Represents a declarative benchmarking task configuration."""
    description: Optional[str] = None
    description_long: Optional[str] = None
    ds_name: Optional[str] = None
    obs_lat: Optional[float] = None
    obs_lon: Optional[float] = None
    obs_tz: Optional[str] = None
    train_idx: List[str]
    test_idx: List[str]
    val_idx: List[str]
    target: Union[str, List[str]]
    remove: List[str] = Field(default_factory=list)
    eval_metrics: List[str] = Field(default_factory=list)
    dropna: bool = True
    log_transform: bool = False
    window_size: Optional[int] = None
    forecast_horizon: Optional[int] = None
    session_col: Optional[str] = None
    effective_wind_speed: Optional[float] = None
    lag_features: Optional[List[str]] = None


class DatasetConfig(BaseModel):
    """Represents the metadata and schema configuration for a supported dataset."""
    local_data_path: str
    citation_path: Optional[str] = None
    remote_data_path: Optional[str] = None
    description: Optional[str] = None
    lazy: bool = False
    flatten: bool = True
    feature_map: Dict[str, Any] = Field(default_factory=dict)

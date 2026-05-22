from pydantic import BaseModel, Field
from typing import List, Optional, Any, Union


class Task(BaseModel):
    """
    Represents a benchmarking task configuration.
    """
    train_idx: List[str]
    test_idx: List[str]
    val_idx: List[str]
    target: Union[str, List[str]]
    remove: List[str] = Field(default_factory=list)
    dropna: bool = True
    log_transform: bool = False


class DatasetConfig(BaseModel):
    """
    Represents the metadata for a supported dataset.
    """
    local_data_path: str
    description: Optional[str] = None
    # Add other fields as discovered in datasets.json

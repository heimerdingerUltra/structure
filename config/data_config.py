from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class ScalerType(Enum):
    ROBUST = "robust"
    STANDARD = "standard"
    MINMAX = "minmax"
    QUANTILE = "quantile"


@dataclass
class DataConfig:
    test_size: float = 0.2
    val_size: float = 0.1
    seed: int = 42
    
    scaler_type: ScalerType = ScalerType.ROBUST
    
    outlier_removal: bool = True
    outlier_threshold: float = 3.0
    
    missing_strategy: str = "median"
    
    feature_selection: bool = False
    feature_selection_k: Optional[int] = None
    
    augmentation: bool = False
    augmentation_factor: float = 0.1
    
    iv_min: float = 0.0
    iv_max: float = 500.0
    
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    
    cache_preprocessed: bool = True
    cache_dir: str = ".cache"

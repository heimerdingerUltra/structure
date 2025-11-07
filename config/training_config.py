from dataclasses import dataclass, field
from typing import Optional, Dict, Literal
from enum import Enum


class OptimizerType(Enum):
    ADAMW = "adamw"
    ADAM = "adam"
    LAMB = "lamb"
    LION = "lion"
    

class SchedulerType(Enum):
    COSINE = "cosine"
    COSINE_WARMUP = "cosine_warmup"
    ONECYCLE = "onecycle"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    

@dataclass
class TrainingConfig:
    batch_size: int = 512
    epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    
    optimizer: OptimizerType = OptimizerType.ADAMW
    optimizer_kwargs: Dict = field(default_factory=dict)
    
    scheduler: SchedulerType = SchedulerType.COSINE_WARMUP
    scheduler_kwargs: Dict = field(default_factory=dict)
    
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    
    validation_frequency: int = 1
    checkpoint_frequency: int = 10
    
    accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    warmup_epochs: int = 10
    warmup_start_lr: float = 1e-6
    
    label_smoothing: float = 0.0
    
    def __post_init__(self):
        if not self.scheduler_kwargs:
            if self.scheduler == SchedulerType.COSINE_WARMUP:
                self.scheduler_kwargs = {
                    'T_0': 10,
                    'T_mult': 2,
                    'eta_min': 1e-6
                }
            elif self.scheduler == SchedulerType.ONECYCLE:
                self.scheduler_kwargs = {
                    'max_lr': self.learning_rate * 10,
                    'pct_start': 0.3,
                    'div_factor': 25,
                    'final_div_factor': 1e4
                }


def create_training_config(
    strategy: Literal["fast", "accurate", "balanced"] = "balanced",
    **overrides
) -> TrainingConfig:
    
    strategies = {
        "fast": TrainingConfig(
            batch_size=1024,
            epochs=100,
            learning_rate=2e-3,
            early_stopping_patience=10,
            scheduler=SchedulerType.ONECYCLE
        ),
        "accurate": TrainingConfig(
            batch_size=256,
            epochs=400,
            learning_rate=5e-4,
            weight_decay=1e-4,
            early_stopping_patience=40,
            warmup_epochs=20,
            scheduler=SchedulerType.COSINE_WARMUP
        ),
        "balanced": TrainingConfig(
            batch_size=512,
            epochs=200,
            learning_rate=1e-3,
            early_stopping_patience=20,
            warmup_epochs=10,
            scheduler=SchedulerType.COSINE_WARMUP
        )
    }
    
    config = strategies[strategy]
    
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import yaml
from enum import Enum


class Device(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class Precision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


@dataclass(frozen=True)
class Runtime:
    device: Device = Device.CUDA
    precision: Precision = Precision.FP16
    compile: bool = True
    channels_last: bool = True
    deterministic: bool = False
    benchmark: bool = True
    
    num_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass(frozen=True)
class Training:
    epochs: int = 200
    batch_size: int = 512
    accumulation_steps: int = 1
    
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    
    gradient_clip: float = 1.0
    ema_decay: float = 0.9999
    
    warmup_steps: int = 1000
    scheduler_cycles: int = 3
    min_lr_ratio: float = 1e-2
    
    early_stopping_patience: int = 25
    early_stopping_delta: float = 1e-5
    
    validation_frequency: int = 1
    checkpoint_frequency: int = 10


@dataclass(frozen=True)
class Data:
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    stratify_bins: int = 10
    min_samples_per_bin: int = 5
    
    outlier_removal: bool = True
    outlier_std_threshold: float = 4.0
    
    cache_enabled: bool = True
    cache_dir: Path = field(default_factory=lambda: Path(".cache"))
    
    augmentation_probability: float = 0.3
    mixup_alpha: float = 0.4
    cutmix_alpha: float = 1.0
    noise_std: float = 0.02


@dataclass(frozen=True)
class Model:
    architecture: str = "tabpfn"
    
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    
    dropout: float = 0.1
    attention_dropout: float = 0.0
    path_dropout: float = 0.1
    
    activation: str = "swiglu"
    norm_type: str = "rmsnorm"
    
    use_flash_attention: bool = True
    use_rotary_embeddings: bool = True
    
    label_smoothing: float = 0.0


@dataclass
class Configuration:
    runtime: Runtime = field(default_factory=Runtime)
    training: Training = field(default_factory=Training)
    data: Data = field(default_factory=Data)
    model: Model = field(default_factory=Model)
    
    experiment_name: str = "volatility_forge"
    seed: int = 42
    
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    
    def __post_init__(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Configuration':
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            runtime=Runtime(**config_dict.get('runtime', {})),
            training=Training(**config_dict.get('training', {})),
            data=Data(**config_dict.get('data', {})),
            model=Model(**config_dict.get('model', {})),
            **{k: v for k, v in config_dict.items() 
               if k not in ['runtime', 'training', 'data', 'model']}
        )
    
    def to_yaml(self, path: str):
        config_dict = {
            'runtime': self.runtime.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'experiment_name': self.experiment_name,
            'seed': self.seed
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'runtime': vars(self.runtime),
            'training': vars(self.training),
            'data': vars(self.data),
            'model': vars(self.model),
            'experiment_name': self.experiment_name,
            'seed': self.seed
        }


def get_optimal_config(
    gpu_memory_gb: int = 16,
    strategy: str = "balanced"
) -> Configuration:
    
    if strategy == "speed":
        return Configuration(
            training=Training(
                epochs=100,
                batch_size=min(1024, 2048 if gpu_memory_gb >= 24 else 1024),
                learning_rate=2e-3
            ),
            model=Model(
                d_model=256,
                n_layers=8,
                n_heads=8
            )
        )
    
    elif strategy == "quality":
        return Configuration(
            training=Training(
                epochs=500,
                batch_size=min(256, 512 if gpu_memory_gb >= 24 else 256),
                learning_rate=5e-4,
                ema_decay=0.9999
            ),
            model=Model(
                d_model=768,
                n_layers=16,
                n_heads=12,
                dropout=0.15
            )
        )
    
    else:
        return Configuration()

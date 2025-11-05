from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class ModelType(Enum):
    TABPFN = "tabpfn"
    MAMBA = "mamba"
    XLSTM = "xlstm"
    HYPERMIXER = "hypermixer"
    TTT = "ttt"
    MODERN_TCN = "modern_tcn"


@dataclass
class ModelConfig:
    model_type: ModelType
    n_features: int
    hyperparameters: Dict
    
    @property
    def name(self) -> str:
        return self.model_type.value


@dataclass
class TabPFNConfig:
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    max_samples: int = 1000
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'mlp_ratio': self.mlp_ratio,
            'dropout': self.dropout,
            'max_samples': self.max_samples
        }


@dataclass
class MambaConfig:
    d_model: int = 256
    n_layers: int = 8
    d_state: int = 16
    expand: int = 2
    dropout: float = 0.1
    dt_rank: str = "auto"
    conv_kernel: int = 3
    
    def to_dict(self) -> Dict:
        return {
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'd_state': self.d_state,
            'expand': self.expand,
            'dropout': self.dropout
        }


@dataclass
class xLSTMConfig:
    hidden_size: int = 256
    n_layers: int = 4
    use_mlstm: bool = True
    dropout: float = 0.1
    stabilization: str = "group_norm"
    
    def to_dict(self) -> Dict:
        return {
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers,
            'use_mlstm': self.use_mlstm,
            'dropout': self.dropout
        }


@dataclass
class HyperMixerConfig:
    dim: int = 256
    n_blocks: int = 8
    patch_size: int = 1
    expansion_factor: int = 4
    dropout: float = 0.1
    hyper_dim: int = 32
    
    def to_dict(self) -> Dict:
        return {
            'dim': self.dim,
            'n_blocks': self.n_blocks,
            'patch_size': self.patch_size,
            'expansion_factor': self.expansion_factor,
            'dropout': self.dropout
        }


@dataclass
class TTTConfig:
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    mini_batch_size: int = 16
    n_inner_steps: int = 1
    lr_inner: float = 0.01
    
    def to_dict(self) -> Dict:
        return {
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'dropout': self.dropout
        }


@dataclass
class ModernTCNConfig:
    channels: List[int] = field(default_factory=lambda: [256, 256, 256, 256])
    kernel_size: int = 3
    dropout: float = 0.1
    use_se: bool = True
    dilation_base: int = 2
    
    def to_dict(self) -> Dict:
        return {
            'channels': self.channels,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout,
            'use_se': self.use_se
        }


_MODEL_CONFIGS = {
    ModelType.TABPFN: TabPFNConfig,
    ModelType.MAMBA: MambaConfig,
    ModelType.XLSTM: xLSTMConfig,
    ModelType.HYPERMIXER: HyperMixerConfig,
    ModelType.TTT: TTTConfig,
    ModelType.MODERN_TCN: ModernTCNConfig
}


def create_model_config(
    model_type: ModelType,
    n_features: int,
    **kwargs
) -> ModelConfig:
    config_class = _MODEL_CONFIGS[model_type]
    hyperparams = config_class(**kwargs)
    
    return ModelConfig(
        model_type=model_type,
        n_features=n_features,
        hyperparameters=hyperparams.to_dict()
    )

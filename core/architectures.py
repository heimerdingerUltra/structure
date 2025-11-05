import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable
from dataclasses import dataclass
from core.attention import AttentionConfig, GroupedQueryAttention
from core.tensor_ops import rms_norm, fused_swiglu


@dataclass(frozen=True)
class TransformerConfig:
    dim: int
    depth: int
    n_heads: int
    n_kv_heads: int
    ffn_dim_multiplier: float
    norm_eps: float
    max_seq_len: int
    dropout: float
    
    @property
    def ffn_dim(self) -> int:
        hidden_dim = int(2 * self.dim * 4 / 3)
        return int(self.ffn_dim_multiplier * hidden_dim)


class RMSNorm(nn.Module):
    
    __constants__ = ['dim', 'eps']
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps)


class SwiGLUFFN(nn.Module):
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w2(nn.functional.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    
    def __init__(self, layer_id: int, config: TransformerConfig):
        super().__init__()
        
        self.layer_id = layer_id
        
        attn_config = AttentionConfig.from_dim(
            config.dim,
            config.n_heads,
            config.n_kv_heads,
            config.dropout,
            bias=False
        )
        
        self.attention = GroupedQueryAttention(attn_config)
        self.feed_forward = SwiGLUFFN(config.dim, config.ffn_dim, config.dropout)
        
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        
        self.layer_scale = nn.Parameter(torch.ones(config.dim) * 1e-5)
    
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        cache: Optional[tuple] = None
    ) -> tuple[Tensor, Optional[tuple]]:
        
        h, new_cache = self.attention(self.attention_norm(x), mask, cache)
        x = x + self.layer_scale * h
        
        h = self.feed_forward(self.ffn_norm(x))
        x = x + self.layer_scale * h
        
        return x, new_cache


class Transformer(nn.Module):
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.config = config
        
        self.layers = nn.ModuleList([
            TransformerBlock(i, config)
            for i in range(config.depth)
        ])
        
        self.norm = RMSNorm(config.dim, config.norm_eps)
    
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        cache: Optional[list] = None
    ) -> tuple[Tensor, Optional[list]]:
        
        new_cache = [] if cache is not None else None
        
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, layer_new_cache = layer(x, mask, layer_cache)
            
            if new_cache is not None:
                new_cache.append(layer_new_cache)
        
        x = self.norm(x)
        
        return x, new_cache


class VolatilityTransformer(nn.Module):
    
    def __init__(
        self,
        n_features: int,
        config: TransformerConfig,
        pooling: str = 'attention'
    ):
        super().__init__()
        
        self.token_embedding = nn.Linear(n_features, config.dim)
        self.transformer = Transformer(config)
        
        self.pooling = pooling
        if pooling == 'attention':
            from core.attention import AttentionPooling
            self.pool = AttentionPooling(config.dim)
        
        self.head = nn.Sequential(
            nn.Linear(config.dim, config.dim // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim // 2, 1)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.token_embedding(x)
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x, _ = self.transformer(x)
        
        if self.pooling == 'mean':
            x = x.mean(dim=1)
        elif self.pooling == 'max':
            x = x.max(dim=1)[0]
        elif self.pooling == 'attention':
            x = self.pool(x)
        elif self.pooling == 'cls':
            x = x[:, 0]
        
        return self.head(x).squeeze(-1)


class ResidualBlock(nn.Module):
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: Callable = nn.SiLU
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.layer_scale = nn.Parameter(torch.ones(dim) * 1e-5)
    
    def forward(self, x: Tensor) -> Tensor:
        return x + self.layer_scale * self.net(x)


class ModernMLP(nn.Module):
    
    def __init__(
        self,
        n_features: int,
        hidden_dims: list[int],
        dropout: float = 0.1,
        activation: Callable = nn.SiLU
    ):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Linear(n_features, hidden_dims[0]),
            RMSNorm(hidden_dims[0]),
            activation()
        )
        
        self.blocks = nn.ModuleList([
            ResidualBlock(dim, dim * 4, dropout, activation)
            for dim in hidden_dims
        ])
        
        self.transitions = nn.ModuleList([
            nn.Linear(hidden_dims[i], hidden_dims[i+1])
            for i in range(len(hidden_dims) - 1)
        ])
        
        self.norm = RMSNorm(hidden_dims[-1])
        
        self.head = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        
        x = self.norm(x)
        
        return self.head(x).squeeze(-1)


class DenseBlock(nn.Module):
    
    def __init__(
        self,
        in_dim: int,
        growth_rate: int,
        n_layers: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(n_layers):
            layer = nn.Sequential(
                RMSNorm(in_dim + i * growth_rate),
                nn.Linear(in_dim + i * growth_rate, growth_rate),
                nn.SiLU(),
                nn.Dropout(dropout)
            )
            self.layers.append(layer)
    
    def forward(self, x: Tensor) -> Tensor:
        features = [x]
        
        for layer in self.layers:
            out = layer(torch.cat(features, dim=-1))
            features.append(out)
        
        return torch.cat(features, dim=-1)


class DenseNet(nn.Module):
    
    def __init__(
        self,
        n_features: int,
        init_dim: int = 256,
        growth_rate: int = 64,
        block_config: tuple = (4, 4, 4),
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Linear(n_features, init_dim),
            RMSNorm(init_dim),
            nn.SiLU()
        )
        
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        
        num_features = init_dim
        
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_features, growth_rate, num_layers, dropout)
            self.blocks.append(block)
            
            num_features += num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = nn.Sequential(
                    RMSNorm(num_features),
                    nn.Linear(num_features, num_features // 2),
                    nn.Dropout(dropout)
                )
                self.transitions.append(trans)
                num_features = num_features // 2
        
        self.norm = RMSNorm(num_features)
        self.head = nn.Linear(num_features, 1)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        
        x = self.norm(x)
        
        return self.head(x).squeeze(-1)

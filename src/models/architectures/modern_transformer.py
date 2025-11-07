import torch
import torch.nn as nn
from .attention import MultiQueryAttention, LinearAttention
from .normalization import RMSNorm, LayerScale
from .activations import SwiGLU


class TransformerBlock(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_type: str = 'multi_query',
        use_layer_scale: bool = True,
        layer_scale_init: float = 1e-5
    ):
        super().__init__()
        
        self.norm1 = RMSNorm(d_model)
        
        if attention_type == 'multi_query':
            self.attn = MultiQueryAttention(d_model, n_heads, n_kv_heads=max(1, n_heads // 4))
        elif attention_type == 'linear':
            self.attn = LinearAttention(d_model, n_heads)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, int(d_model * mlp_ratio))
        
        self.dropout = nn.Dropout(dropout)
        
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.ls1 = LayerScale(d_model, layer_scale_init)
            self.ls2 = LayerScale(d_model, layer_scale_init)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_layer_scale:
            x = x + self.dropout(self.ls1(self.attn(self.norm1(x))))
            x = x + self.dropout(self.ls2(self.mlp(self.norm2(x))))
        else:
            x = x + self.dropout(self.attn(self.norm1(x)))
            x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class ModernTransformer(nn.Module):
    
    def __init__(
        self,
        n_features: int,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_type: str = 'multi_query',
        use_layer_scale: bool = True,
        pooling: str = 'mean'
    ):
        super().__init__()
        
        self.embed = nn.Linear(n_features, d_model)
        self.norm_in = RMSNorm(d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model,
                n_heads,
                mlp_ratio,
                dropout,
                attention_type,
                use_layer_scale
            )
            for _ in range(n_layers)
        ])
        
        self.norm_out = RMSNorm(d_model)
        
        self.pooling = pooling
        if pooling == 'attention':
            self.pool = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.norm_in(x)
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm_out(x)
        
        if self.pooling == 'mean':
            x = x.mean(dim=1)
        elif self.pooling == 'max':
            x = x.max(dim=1)[0]
        elif self.pooling == 'attention':
            query = self.pool_query.expand(x.size(0), -1, -1)
            x, _ = self.pool(query, x, x)
            x = x.squeeze(1)
        elif self.pooling == 'cls':
            x = x[:, 0]
        
        return self.head(x).squeeze(-1)


class HybridTransformer(nn.Module):
    
    def __init__(
        self,
        n_features: int,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed = nn.Linear(n_features, d_model)
        self.norm_in = RMSNorm(d_model)
        
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            attention_type = 'multi_query' if i % 2 == 0 else 'linear'
            self.blocks.append(
                TransformerBlock(d_model, n_heads, mlp_ratio, dropout, attention_type)
            )
        
        self.norm_out = RMSNorm(d_model)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.norm_in(x)
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm_out(x)
        x = x.mean(dim=1)
        
        return self.head(x).squeeze(-1)


class ParallelTransformer(nn.Module):
    
    def __init__(
        self,
        n_features: int,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed = nn.Linear(n_features, d_model)
        self.norm_in = RMSNorm(d_model)
        
        self.attn_blocks = nn.ModuleList([
            MultiQueryAttention(d_model, n_heads, n_kv_heads=max(1, n_heads // 4))
            for _ in range(n_layers)
        ])
        
        self.mlp_blocks = nn.ModuleList([
            SwiGLU(d_model, int(d_model * mlp_ratio))
            for _ in range(n_layers)
        ])
        
        self.attn_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.mlp_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        
        self.norm_out = RMSNorm(d_model)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.norm_in(x)
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        for attn, mlp, attn_norm, mlp_norm in zip(
            self.attn_blocks, self.mlp_blocks, self.attn_norms, self.mlp_norms
        ):
            x = x + attn(attn_norm(x)) + mlp(mlp_norm(x))
        
        x = self.norm_out(x)
        x = x.mean(dim=1)
        
        return self.head(x).squeeze(-1)

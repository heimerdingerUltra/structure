import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import math
from typing import Optional


class RotaryEmbedding(nn.Module):
    
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb
    
    @staticmethod
    def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated


class MultiHeadAttention(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        use_rotary: bool = True,
        use_flash: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash = use_flash
        
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        self.rotary = RotaryEmbedding(self.head_dim) if use_rotary else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.rotary is not None:
            emb = self.rotary(N, x.device)
            cos = emb.cos()[None, None, :, :].expand(B, self.n_heads, -1, -1)
            sin = emb.sin()[None, None, :, :].expand(B, self.n_heads, -1, -1)
            
            q = self.rotary.apply_rotary_emb(q, cos, sin)
            k = self.rotary.apply_rotary_emb(k, cos, sin)
        
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=self.scale
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            out = attn @ v
        
        out = out.transpose(1, 2).contiguous().reshape(B, N, C)
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return out


class SwiGLU(nn.Module):
    
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(d_model * expansion * 2 / 3)
        
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class RMSNorm(nn.Module):
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class TransformerBlock(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        expansion: int = 4,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        path_dropout: float = 0.0,
        use_rotary: bool = True,
        use_flash: bool = True
    ):
        super().__init__()
        
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model, n_heads,
            dropout=attention_dropout,
            use_rotary=use_rotary,
            use_flash=use_flash
        )
        
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, expansion, dropout)
        
        self.path_dropout = path_dropout
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        
        if self.training and self.path_dropout > 0:
            attn_out = F.dropout(attn_out, p=self.path_dropout, training=True)
        
        x = x + attn_out
        
        ffn_out = self.ffn(self.norm2(x))
        
        if self.training and self.path_dropout > 0:
            ffn_out = F.dropout(ffn_out, p=self.path_dropout, training=True)
        
        x = x + ffn_out
        
        return x


class VisionTransformer(nn.Module):
    
    def __init__(
        self,
        n_features: int,
        d_model: int = 512,
        n_layers: int = 12,
        n_heads: int = 8,
        expansion: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        path_dropout: float = 0.1,
        patch_size: int = 1,
        use_rotary: bool = True,
        use_flash: bool = True,
        pooling: str = 'cls'
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.n_patches = (n_features + patch_size - 1) // patch_size
        self.pooling = pooling
        
        self.patch_embed = nn.Linear(patch_size, d_model)
        
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.n_patches + 1, d_model) * 0.02
            )
        else:
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.n_patches, d_model) * 0.02
            )
        
        path_drop_rates = [
            path_dropout * (i / (n_layers - 1))
            for i in range(n_layers)
        ]
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, expansion,
                dropout, attention_dropout,
                path_drop_rates[i],
                use_rotary, use_flash
            )
            for i in range(n_layers)
        ])
        
        self.norm = RMSNorm(d_model)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        padding = (self.patch_size - x.shape[1] % self.patch_size) % self.patch_size
        if padding > 0:
            x = F.pad(x, (0, padding))
        
        x = x.view(B, self.n_patches, self.patch_size)
        x = self.patch_embed(x)
        
        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        if self.pooling == 'cls':
            x = x[:, 0]
        elif self.pooling == 'mean':
            x = x.mean(dim=1)
        elif self.pooling == 'max':
            x = x.max(dim=1)[0]
        else:
            x = x[:, 0]
        
        return self.head(x).squeeze(-1)


class ExponentialMovingAverage(nn.Module):
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        super().__init__()
        self.module = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def create_model(config) -> nn.Module:
    model = VisionTransformer(
        n_features=config.model.n_features,
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        expansion=4,
        dropout=config.model.dropout,
        attention_dropout=config.model.attention_dropout,
        path_dropout=config.model.path_dropout,
        use_rotary=config.model.use_rotary_embeddings,
        use_flash=config.model.use_flash_attention
    )
    
    if config.runtime.compile:
        model = torch.compile(model, mode='max-autotune')
    
    if config.runtime.channels_last:
        model = model.to(memory_format=torch.channels_last)
    
    return model

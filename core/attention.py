import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AttentionConfig:
    dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    dropout: float
    bias: bool
    
    @classmethod
    def from_dim(
        cls,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False
    ) -> 'AttentionConfig':
        assert dim % n_heads == 0
        head_dim = dim // n_heads
        n_kv_heads = n_kv_heads or n_heads
        return cls(dim, n_heads, n_kv_heads, head_dim, dropout, bias)


class GroupedQueryAttention(nn.Module):
    
    __constants__ = ['n_heads', 'n_kv_heads', 'head_dim', 'scale']
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        
        self.q_proj = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * config.head_dim, bias=config.bias)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=config.bias)
        
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        
        self.register_buffer(
            'scale',
            torch.tensor(config.head_dim ** -0.5, dtype=torch.float32),
            persistent=False
        )
    
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        cache: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        
        B, N, C = x.shape
        
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        if cache is not None:
            k_cache, v_cache = cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        out = self.o_proj(out)
        
        new_cache = (k, v) if cache is not None else None
        
        return out, new_cache


class FlashAttention(nn.Module):
    
    def __init__(self, config: AttentionConfig, block_size: int = 128):
        super().__init__()
        
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.block_size = block_size
        
        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=config.bias)
        self.o_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        
        self.scale = config.head_dim ** -0.5
    
    @torch.jit.ignore
    def _flash_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return F.scaled_dot_product_attention(q, k, v, scale=self.scale)
    
    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if hasattr(F, 'scaled_dot_product_attention'):
            out = self._flash_attention(q, k, v)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = attn @ v
        
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.o_proj(out)
        
        return out


class LinearAttention(nn.Module):
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        
        self.q_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.k_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.v_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.o_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        
        self.eps = 1e-6
    
    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        kv = torch.einsum('bhnd,bhnm->bhdm', k, v)
        z = k.sum(dim=2)
        
        out = torch.einsum('bhnd,bhdm->bhnm', q, kv)
        out = out / (torch.einsum('bhnd,bhd->bhn', q, z).unsqueeze(-1) + self.eps)
        
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.o_proj(out)
        
        return out


class HybridAttention(nn.Module):
    
    def __init__(self, config: AttentionConfig, window_size: int = 256):
        super().__init__()
        
        self.window_size = window_size
        
        self.local_attn = GroupedQueryAttention(config)
        self.global_attn = LinearAttention(config)
        
        self.gate = nn.Linear(config.dim, 1)
    
    def forward(self, x: Tensor) -> Tensor:
        N = x.size(1)
        
        if N <= self.window_size:
            return self.local_attn(x)[0]
        
        local_out = self.local_attn(x)[0]
        global_out = self.global_attn(x)
        
        gate = torch.sigmoid(self.gate(x))
        
        return gate * local_out + (1 - gate) * global_out


class MultiHeadSelfAttention(nn.Module):
    
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        
        assert dim % n_heads == 0
        
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class AttentionPooling(nn.Module):
    
    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Parameter(torch.randn(1, 1, dim))
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        
        q = self.q.expand(B, -1, -1).reshape(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        
        kv = self.kv(x).reshape(B, N, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        out = self.proj(out)
        
        return out.squeeze(1)

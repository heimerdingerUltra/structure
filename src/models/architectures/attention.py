import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiQueryAttention(nn.Module):
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int = None):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0
        assert n_heads % self.n_kv_heads == 0
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape
        
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            attn = attn + attn_mask
        
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.o_proj(out)
        
        return out


class SlidingWindowAttention(nn.Module):
    
    def __init__(self, d_model: int, n_heads: int, window_size: int = 512):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        window_size = min(self.window_size, N)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        mask = torch.ones(N, N, device=x.device, dtype=torch.bool)
        for i in range(N):
            start = max(0, i - window_size // 2)
            end = min(N, i + window_size // 2 + 1)
            mask[i, start:end] = False
        
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        
        return out


class LinearAttention(nn.Module):
    
    def __init__(self, d_model: int, n_heads: int = 8, eps: float = 1e-6):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.eps = eps
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        kv = torch.einsum('bhnd,bhnm->bhdm', k, v)
        z = torch.einsum('bhnd,bhd->bhn', k, torch.ones(B, self.n_heads, self.head_dim, device=x.device))
        
        out = torch.einsum('bhnd,bhdm->bhnm', q, kv)
        out = out / (torch.einsum('bhnd,bhd->bhn', q, z) + self.eps).unsqueeze(-1)
        
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.o_proj(out)
        
        return out


class CrossAttention(nn.Module):
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        _, M, _ = context.shape
        
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.o_proj(out)
        
        return out

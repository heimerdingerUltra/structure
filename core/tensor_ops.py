import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    scale: Optional[float] = None
) -> Tuple[Tensor, Tensor]:
    
    if scale is None:
        scale = query.size(-1) ** -0.5
    
    attn_weight = torch.einsum('bhqd,bhkd->bhqk', query, key) * scale
    
    if mask is not None:
        attn_weight = attn_weight.masked_fill(mask == 0, float('-inf'))
    
    attn_weight = F.softmax(attn_weight, dim=-1)
    
    if dropout_p > 0.0:
        attn_weight = F.dropout(attn_weight, p=dropout_p, training=True)
    
    output = torch.einsum('bhqk,bhkd->bhqd', attn_weight, value)
    
    return output, attn_weight


def grouped_query_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    n_groups: int,
    mask: Optional[Tensor] = None
) -> Tensor:
    
    batch_size, n_heads, seq_len, head_dim = query.shape
    
    assert n_heads % n_groups == 0
    heads_per_group = n_heads // n_groups
    
    query = query.view(batch_size, n_groups, heads_per_group, seq_len, head_dim)
    key = key.view(batch_size, n_groups, 1, seq_len, head_dim)
    value = value.view(batch_size, n_groups, 1, seq_len, head_dim)
    
    scale = head_dim ** -0.5
    attn = torch.einsum('bghqd,bgkd->bghqk', query, key.squeeze(2)) * scale
    
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float('-inf'))
    
    attn = F.softmax(attn, dim=-1)
    
    output = torch.einsum('bghqk,bgkd->bghqd', attn, value.squeeze(2))
    output = output.reshape(batch_size, n_heads, seq_len, head_dim)
    
    return output


def apply_rotary_embeddings(
    x: Tensor,
    cos: Tensor,
    sin: Tensor
) -> Tensor:
    
    x1, x2 = x.chunk(2, dim=-1)
    
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)
    
    return rotated


def flash_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    block_size: int = 128
) -> Tensor:
    
    batch_size, n_heads, seq_len, head_dim = query.shape
    
    scale = head_dim ** -0.5
    output = torch.zeros_like(query)
    
    for i in range(0, seq_len, block_size):
        end_i = min(i + block_size, seq_len)
        query_block = query[:, :, i:end_i]
        
        for j in range(0, seq_len, block_size):
            end_j = min(j + block_size, seq_len)
            key_block = key[:, :, j:end_j]
            value_block = value[:, :, j:end_j]
            
            scores = torch.einsum('bhqd,bhkd->bhqk', query_block, key_block) * scale
            attn = F.softmax(scores, dim=-1)
            
            output[:, :, i:end_i] += torch.einsum('bhqk,bhkd->bhqd', attn, value_block)
    
    return output


def compute_rope_embeddings(
    dim: int,
    seq_len: int,
    device: torch.device,
    base: float = 10000.0
) -> Tuple[Tensor, Tensor]:
    
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    
    t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
    
    freqs = torch.einsum('i,j->ij', t, inv_freq)
    
    emb = torch.cat([freqs, freqs], dim=-1)
    
    return emb.cos(), emb.sin()


def fused_swiglu(x: Tensor, w_gate: Tensor, w_up: Tensor, w_down: Tensor) -> Tensor:
    
    gate = torch.einsum('bd,dh->bh', x, w_gate)
    up = torch.einsum('bd,dh->bh', x, w_up)
    
    activated = F.silu(gate) * up
    
    output = torch.einsum('bh,hd->bd', activated, w_down)
    
    return output


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    
    return weight * x_normed


def group_norm(x: Tensor, n_groups: int, weight: Tensor, bias: Tensor, eps: float = 1e-6) -> Tensor:
    
    batch_size, channels = x.shape
    
    x_grouped = x.view(batch_size, n_groups, -1)
    
    mean = x_grouped.mean(dim=-1, keepdim=True)
    var = x_grouped.var(dim=-1, keepdim=True, unbiased=False)
    
    x_normed = (x_grouped - mean) / torch.sqrt(var + eps)
    x_normed = x_normed.view(batch_size, channels)
    
    return weight * x_normed + bias


def gelu_approx(x: Tensor) -> Tensor:
    
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
    ))


def swish(x: Tensor, beta: float = 1.0) -> Tensor:
    
    return x * torch.sigmoid(beta * x)


def geglu(x: Tensor) -> Tensor:
    
    x, gate = x.chunk(2, dim=-1)
    return x * F.gelu(gate)


def apply_causal_mask(scores: Tensor, seq_len: int) -> Tensor:
    
    mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
    return scores.masked_fill(mask, float('-inf'))


def compute_relative_positions(seq_len: int, max_distance: int = 128) -> Tensor:
    
    positions = torch.arange(seq_len)
    relative = positions[:, None] - positions[None, :]
    
    return torch.clamp(relative, -max_distance, max_distance) + max_distance


def local_attention_mask(seq_len: int, window_size: int, device: torch.device) -> Tensor:
    
    mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
    
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = False
    
    return mask


def fused_layernorm_residual(
    x: Tensor,
    residual: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: float = 1e-6
) -> Tuple[Tensor, Tensor]:
    
    x = x + residual
    
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    
    x_normed = (x - mean) / torch.sqrt(var + eps)
    output = weight * x_normed + bias
    
    return output, x


def efficient_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    use_flash: bool = True,
    block_size: int = 128
) -> Tensor:
    
    if use_flash and query.size(2) > block_size:
        return flash_attention_forward(query, key, value, block_size)
    else:
        return scaled_dot_product_attention(query, key, value)[0]


def tensor_parallel_linear(
    x: Tensor,
    weight: Tensor,
    n_devices: int = 1
) -> Tensor:
    
    if n_devices == 1:
        return F.linear(x, weight)
    
    weight_splits = weight.chunk(n_devices, dim=0)
    
    outputs = [F.linear(x, w) for w in weight_splits]
    
    return torch.cat(outputs, dim=-1)


def gradient_checkpointing(fn, *args, use_reentrant: bool = False):
    
    return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=use_reentrant)

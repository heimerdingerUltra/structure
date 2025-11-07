import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class LayerScale(nn.Module):
    
    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class AdaptiveLayerNorm(nn.Module):
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        
        self.scale_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        scale = self.scale_net(x.mean(dim=1, keepdim=True))
        
        return self.weight * x_norm * scale + self.bias


class GroupNorm1d(nn.Module):
    
    def __init__(self, num_channels: int, num_groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape
        x = x.view(B, self.num_groups, -1)
        
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        x = x.view(B, C)
        return self.weight * x + self.bias


class BatchChannelNorm(nn.Module):
    
    def __init__(self, num_features: int, eps: float = 1e-6, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean = x.mean(0)
            var = x.var(0, unbiased=False)
            
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x + self.bias


class ConditionalLayerNorm(nn.Module):
    
    def __init__(self, dim: int, condition_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        
        self.scale_transform = nn.Linear(condition_dim, dim)
        self.shift_transform = nn.Linear(condition_dim, dim)
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        scale = self.scale_transform(condition)
        shift = self.shift_transform(condition)
        
        if scale.dim() == 2:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        
        return scale * x_norm + shift


class ScaleNorm(nn.Module):
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.tensor(dim ** 0.5))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True)
        return self.scale * x / (norm + self.eps)


class QKNorm(nn.Module):
    
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)
        
    def forward(self, q: torch.Tensor, k: torch.Tensor):
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q, k

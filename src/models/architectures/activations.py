import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SwiGLU(nn.Module):
    
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GeGLU(nn.Module):
    
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)) * self.w3(x))


class ReGLU(nn.Module):
    
    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.relu(self.w1(x)) * self.w3(x))


class SquaredReLU(nn.Module):
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pow(F.relu(x), 2)


class STAR(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * F.relu(x) ** 2 + self.bias


class Mish(nn.Module):
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class LiSHT(nn.Module):
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(x)


class GELU2(nn.Module):
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Swish(nn.Module):
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


class HardSwish(nn.Module):
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.hardtanh(x + 3, 0, 6) / 6


class ELU(nn.Module):
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x, self.alpha)


class SELU(nn.Module):
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.selu(x)


class PReLU(nn.Module):
    
    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super().__init__()
        self.num_parameters = num_parameters
        self.weight = nn.Parameter(torch.Tensor(num_parameters).fill_(init))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.prelu(x, self.weight)


class LeakyReLU(nn.Module):
    
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, self.negative_slope)


class SoftExponential(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha
        
        if alpha == 0:
            return x
        elif alpha < 0:
            return -torch.log(1 - alpha * (x + alpha)) / alpha
        else:
            return (torch.exp(alpha * x) - 1) / alpha + alpha


class ACON(nn.Module):
    
    def __init__(self, width: int):
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, width, 1))
        self.p2 = nn.Parameter(torch.randn(1, width, 1))
        self.beta = nn.Parameter(torch.ones(1, width, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.p1 - self.p2) * x * torch.sigmoid(self.beta * (self.p1 - self.p2) * x) + self.p2 * x


class FReLU(nn.Module):
    
    def __init__(self, in_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size, 
                             padding=kernel_size//2, groups=in_channels, bias=False)
        self.bn = nn.BatchNorm1d(in_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            squeeze = True
        else:
            squeeze = False
            
        tx = self.conv(x)
        tx = self.bn(tx)
        out = torch.max(x, tx)
        
        if squeeze:
            out = out.squeeze(-1)
            
        return out


def get_activation(name: str, **kwargs):
    activations = {
        'swiglu': lambda: SwiGLU(**kwargs),
        'geglu': lambda: GeGLU(**kwargs),
        'reglu': lambda: ReGLU(**kwargs),
        'squared_relu': lambda: SquaredReLU(),
        'star': lambda: STAR(),
        'mish': lambda: Mish(),
        'lisht': lambda: LiSHT(),
        'gelu': lambda: nn.GELU(),
        'gelu2': lambda: GELU2(),
        'silu': lambda: nn.SiLU(),
        'swish': lambda: Swish(**kwargs),
        'hardswish': lambda: HardSwish(),
        'relu': lambda: nn.ReLU(),
        'elu': lambda: ELU(**kwargs),
        'selu': lambda: SELU(),
        'prelu': lambda: PReLU(**kwargs),
        'leaky_relu': lambda: LeakyReLU(**kwargs),
        'tanh': lambda: nn.Tanh(),
        'softplus': lambda: nn.Softplus(),
        'softsign': lambda: nn.Softsign(),
    }
    
    return activations.get(name.lower(), lambda: nn.GELU())()

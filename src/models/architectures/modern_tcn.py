import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int = 1, padding: int = 0):
        super().__init__()
        
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=padding, dilation=dilation, groups=in_channels, bias=False
        )
        
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ModernTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, 
                 dropout: float = 0.1, use_se: bool = True):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = DepthwiseSeparableConv1d(
            channels, channels, kernel_size, dilation, padding
        )
        self.norm1 = nn.BatchNorm1d(channels)
        
        self.conv2 = DepthwiseSeparableConv1d(
            channels, channels, kernel_size, dilation, padding
        )
        self.norm2 = nn.BatchNorm1d(channels)
        
        self.se = SqueezeExcitation(channels) if use_se else nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)
        
        out = out + residual
        out = self.activation(out)
        
        return out


class ModernTCN(nn.Module):
    def __init__(
        self,
        n_features: int,
        channels: List[int] = [256, 256, 256, 256],
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_se: bool = True
    ):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, channels[0]),
            nn.BatchNorm1d(channels[0]),
            nn.GELU()
        )
        
        self.blocks = nn.ModuleList()
        
        for i, num_channels in enumerate(channels):
            dilation = 2 ** i
            
            if i > 0 and channels[i] != channels[i-1]:
                self.blocks.append(
                    nn.Conv1d(channels[i-1], channels[i], 1, bias=False)
                )
            
            self.blocks.append(
                ModernTCNBlock(num_channels, kernel_size, dilation, dropout, use_se)
            )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.head = nn.Sequential(
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1] // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        
        x = x.unsqueeze(-1)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.global_pool(x)
        x = x.squeeze(-1)
        
        return self.head(x).squeeze(-1)

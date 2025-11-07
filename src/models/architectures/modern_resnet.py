import torch
import torch.nn as nn
from .activations import get_activation
from .normalization import RMSNorm, LayerScale


class SqueezeExcite(nn.Module):
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class StochasticDepth(nn.Module):
    
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ResidualBlock(nn.Module):
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        use_se: bool = True,
        use_layer_scale: bool = True
    ):
        super().__init__()
        
        self.norm1 = RMSNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act1 = nn.SiLU()
        
        self.norm2 = RMSNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(drop_path)
        
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcite(dim)
        
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = LayerScale(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.norm1(x)
        out = self.fc1(out)
        out = self.act1(out)
        out = self.dropout(out)
        
        out = self.norm2(out)
        out = self.fc2(out)
        out = self.dropout(out)
        
        if self.use_layer_scale:
            out = self.layer_scale(out)
        
        out = self.drop_path(out)
        
        if self.use_se and out.dim() == 2:
            out = out.unsqueeze(-1)
            out = self.se(out)
            out = out.squeeze(-1)
        
        return identity + out


class BottleneckBlock(nn.Module):
    
    def __init__(
        self,
        dim: int,
        expansion: int = 4,
        dropout: float = 0.0,
        use_se: bool = True
    ):
        super().__init__()
        
        hidden_dim = dim * expansion
        
        self.norm1 = RMSNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act1 = nn.SiLU()
        
        self.norm2 = RMSNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.SiLU()
        
        self.norm3 = RMSNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcite(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.norm1(x)
        out = self.fc1(out)
        out = self.act1(out)
        out = self.dropout(out)
        
        out = self.norm2(out)
        out = self.fc2(out)
        out = self.act2(out)
        out = self.dropout(out)
        
        out = self.norm3(out)
        out = self.fc3(out)
        out = self.dropout(out)
        
        if self.use_se and out.dim() == 2:
            out = out.unsqueeze(-1)
            out = self.se(out)
            out = out.squeeze(-1)
        
        return identity + out


class ModernResNet(nn.Module):
    
    def __init__(
        self,
        n_features: int,
        dim: int = 512,
        n_blocks: int = 12,
        hidden_dim_ratio: float = 4.0,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        use_se: bool = True,
        use_layer_scale: bool = True
    ):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Linear(n_features, dim),
            RMSNorm(dim),
            nn.SiLU()
        )
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_blocks)]
        
        self.blocks = nn.ModuleList([
            ResidualBlock(
                dim,
                int(dim * hidden_dim_ratio),
                dropout,
                dpr[i],
                use_se,
                use_layer_scale
            )
            for i in range(n_blocks)
        ])
        
        self.norm = RMSNorm(dim)
        
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        return self.head(x).squeeze(-1)


class PyramidResNet(nn.Module):
    
    def __init__(
        self,
        n_features: int,
        dims: list = [256, 384, 512],
        depths: list = [3, 4, 5],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Linear(n_features, dims[0]),
            RMSNorm(dims[0]),
            nn.SiLU()
        )
        
        self.stages = nn.ModuleList()
        
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            stage = nn.ModuleList()
            
            if i > 0:
                stage.append(nn.Linear(dims[i-1], dim))
            
            for _ in range(depth):
                stage.append(ResidualBlock(dim, int(dim * 4), dropout))
            
            self.stages.append(stage)
        
        self.norm = RMSNorm(dims[-1])
        
        self.head = nn.Sequential(
            nn.Linear(dims[-1], dims[-1] // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dims[-1] // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        
        for stage in self.stages:
            for block in stage:
                x = block(x)
        
        x = self.norm(x)
        
        return self.head(x).squeeze(-1)


class DenseBlock(nn.Module):
    
    def __init__(self, in_dim: int, growth_rate: int, n_layers: int, dropout: float = 0.0):
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        
        self.head = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(num_features // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        
        x = self.norm(x)
        
        return self.head(x).squeeze(-1)

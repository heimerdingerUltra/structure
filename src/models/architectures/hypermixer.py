import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperNetwork(nn.Module):
    def __init__(self, latent_dim: int, target_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class AdaptiveTokenMixing(nn.Module):
    def __init__(self, n_patches: int, dim: int, hyper_dim: int = 32):
        super().__init__()

        self.n_patches = n_patches
        self.dim = dim

        self.hyper_z = nn.Parameter(torch.randn(1, hyper_dim))

        self.weight_generator = HyperNetwork(hyper_dim, n_patches * n_patches)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        weights = self.weight_generator(self.hyper_z)
        weights = weights.view(self.n_patches, self.n_patches)
        weights = F.softmax(weights, dim=-1)

        x = self.norm(x)

        x_transposed = x.transpose(1, 2)
        x_mixed = torch.matmul(weights, x_transposed)
        x_mixed = x_mixed.transpose(1, 2)

        return x_mixed


class AdaptiveChannelMixing(nn.Module):
    def __init__(self, dim: int, expansion_factor: int = 4, dropout: float = 0.0):
        super().__init__()

        hidden_dim = dim * expansion_factor

        self.norm = nn.LayerNorm(dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.gate = nn.Linear(dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        gate = self.gate(x)

        x = self.fc1(x)
        x = F.gelu(x) * torch.sigmoid(gate)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.dropout(x)

        return x


class HyperMixerBlock(nn.Module):
    def __init__(
        self,
        n_patches: int,
        dim: int,
        expansion_factor: int = 4,
        dropout: float = 0.0,
        hyper_dim: int = 32,
    ):
        super().__init__()

        self.token_mixing = AdaptiveTokenMixing(n_patches, dim, hyper_dim)
        self.channel_mixing = AdaptiveChannelMixing(dim, expansion_factor, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixing(x)
        x = x + self.channel_mixing(x)
        return x


class HyperMixer(nn.Module):
    def __init__(
        self,
        n_features: int,
        dim: int = 256,
        n_blocks: int = 8,
        patch_size: int = 1,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        hyper_dim: int = 32,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.n_patches = (n_features + patch_size - 1) // patch_size

        self.patch_embed = nn.Linear(patch_size, dim)

        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, dim) * 0.02)

        self.blocks = nn.ModuleList(
            [
                HyperMixerBlock(self.n_patches, dim, expansion_factor, dropout, hyper_dim)
                for _ in range(n_blocks)
            ]
        )

        self.norm = nn.LayerNorm(dim)

        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        padding = (self.patch_size - x.shape[1] % self.patch_size) % self.patch_size
        if padding > 0:
            x = F.pad(x, (0, padding))

        x = x.view(batch_size, self.n_patches, self.patch_size)

        x = self.patch_embed(x)

        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        x = x.mean(dim=1)

        return self.head(x).squeeze(-1)

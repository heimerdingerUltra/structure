import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class S6Block(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, dt_rank: str = "auto"):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            padding=2,
            groups=self.d_inner,
            bias=False,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape

        x_and_res = self.in_proj(x)
        x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)

        x = F.silu(x)

        x_dbl = self.x_proj(x)
        delta, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)

        delta = F.softplus(self.dt_proj(delta))

        A = -torch.exp(self.A_log.float())

        y = self.selective_scan(x, delta, A, B, C, self.D)

        y = y * F.silu(res)

        return self.out_proj(y)

    def selective_scan(self, u, delta, A, B, C, D):
        batch, seq_len, d_inner = u.shape
        n = A.shape[1]

        deltaA = torch.exp(torch.einsum("bld,dn->bldn", delta, A))
        deltaB_u = torch.einsum("bld,bld,bln->bldn", delta, u, B)

        x = torch.zeros((batch, d_inner, n), device=u.device)
        ys = []

        for i in range(seq_len):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = torch.einsum("bdn,bn->bd", x, C[:, i])
            ys.append(y)

        y = torch.stack(ys, dim=1)

        y = y + u * D.unsqueeze(0).unsqueeze(0)

        return y


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, dropout: float = 0.0):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.mixer = S6Block(d_model, d_state, expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mixer(x)
        x = self.dropout(x)
        return residual + x


class Mamba(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 256,
        n_layers: int = 8,
        d_state: int = 16,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed = nn.Linear(n_features, d_model)

        self.layers = nn.ModuleList(
            [MambaBlock(d_model, d_state, expand, dropout) for _ in range(n_layers)]
        )

        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)

        x = x.unsqueeze(1)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        x = x.mean(dim=1)

        return self.head(x).squeeze(-1)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class QuantumInspiredLinear(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, n_basis: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_basis = n_basis
        
        self.basis_weights = nn.Parameter(torch.randn(n_basis, in_features, out_features) * 0.02)
        self.mixing_coeffs = nn.Parameter(torch.ones(n_basis) / n_basis)
        self.phase_shifts = nn.Parameter(torch.zeros(n_basis))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        phases = torch.cos(self.phase_shifts).view(self.n_basis, 1, 1)
        modulated_weights = self.basis_weights * phases
        
        coeffs = F.softmax(self.mixing_coeffs, dim=0)
        
        superposed_weight = torch.sum(
            modulated_weights * coeffs.view(self.n_basis, 1, 1),
            dim=0
        )
        
        return F.linear(x, superposed_weight.t())


class AdaptiveComputationTime(nn.Module):
    
    def __init__(self, hidden_dim: int, threshold: float = 0.99):
        super().__init__()
        self.threshold = threshold
        self.ponder_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, core_fn, x: torch.Tensor, max_steps: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        device = x.device
        
        halting_prob = torch.zeros(batch_size, device=device)
        remainders = torch.zeros(batch_size, device=device)
        n_updates = torch.zeros(batch_size, device=device)
        
        accumulator = torch.zeros_like(x)
        
        for step in range(max_steps):
            p = torch.sigmoid(self.ponder_head(x))
            p = p.squeeze(-1)
            
            still_running = (halting_prob < self.threshold).float()
            
            new_halted = (halting_prob + p * still_running >= self.threshold).float() * still_running
            
            p_update = p * still_running - new_halted * remainders
            
            halting_prob += p_update
            remainders += new_halted * (1 - halting_prob)
            n_updates += still_running
            
            x_processed = core_fn(x)
            accumulator += x_processed * p_update.unsqueeze(-1)
            
            x = x_processed
            
            if (halting_prob >= self.threshold).all():
                break
        
        accumulator += x * (1 - halting_prob).unsqueeze(-1)
        ponder_cost = n_updates.mean() + remainders.mean()
        
        return accumulator, ponder_cost


class UniversalApproximationBlock(nn.Module):
    
    def __init__(self, dim: int, expansion_factor: int = 4, n_experts: int = 8):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        
        hidden_dim = dim * expansion_factor
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim)
            )
            for _ in range(n_experts)
        ])
        
        self.gate = nn.Linear(dim, n_experts)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        
        output = torch.sum(expert_outputs * gate_probs.unsqueeze(-2), dim=-1)
        
        return output


class HyperNetwork(nn.Module):
    
    def __init__(self, z_dim: int, target_dim: int, target_shape: Tuple[int, ...]):
        super().__init__()
        self.target_shape = target_shape
        self.target_numel = math.prod(target_shape)
        
        self.net = nn.Sequential(
            nn.Linear(z_dim, z_dim * 2),
            nn.LayerNorm(z_dim * 2),
            nn.GELU(),
            nn.Linear(z_dim * 2, z_dim * 4),
            nn.LayerNorm(z_dim * 4),
            nn.GELU(),
            nn.Linear(z_dim * 4, self.target_numel)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        weights = self.net(z)
        return weights.view(-1, *self.target_shape)


class NeuralODE(nn.Module):
    
    def __init__(self, dim: int, n_steps: int = 6):
        super().__init__()
        self.n_steps = n_steps
        
        self.dynamics = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.Tanh(),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, x: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        state = x
        
        for _ in range(self.n_steps):
            dx_dt = self.dynamics(state)
            state = state + dt * dx_dt
        
        return state


class SelfModulatingNeuron(nn.Module):
    
    def __init__(self, dim: int):
        super().__init__()
        
        self.meta_net = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 3)
        )
        
        self.base_weight = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.base_bias = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        meta_params = self.meta_net(x.mean(dim=0, keepdim=True))
        
        weight_scale = torch.sigmoid(meta_params[:, 0:1])
        bias_scale = torch.tanh(meta_params[:, 1:2])
        activation_temp = F.softplus(meta_params[:, 2:3])
        
        modulated_weight = self.base_weight * weight_scale
        modulated_bias = self.base_bias * bias_scale.squeeze()
        
        output = F.linear(x, modulated_weight, modulated_bias)
        output = torch.tanh(output / (activation_temp + 1e-8))
        
        return output


class RecursiveCorticalNetwork(nn.Module):
    
    def __init__(self, dim: int, depth: int = 3, share_weights: bool = True):
        super().__init__()
        self.depth = depth
        self.share_weights = share_weights
        
        if share_weights:
            self.recurrent_block = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.LayerNorm(dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            )
        else:
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, dim * 2),
                    nn.LayerNorm(dim * 2),
                    nn.GELU(),
                    nn.Linear(dim * 2, dim)
                )
                for _ in range(depth)
            ])
        
        self.memory = nn.Parameter(torch.zeros(1, dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        memory = self.memory.expand(batch_size, -1)
        
        for i in range(self.depth):
            if self.share_weights:
                block = self.recurrent_block
            else:
                block = self.blocks[i]
            
            combined = x + memory
            x = block(combined)
            memory = memory + x
        
        return x


class CapsuleNetwork(nn.Module):
    
    def __init__(self, in_dim: int, out_dim: int, n_capsules: int = 8, capsule_dim: int = 16):
        super().__init__()
        self.n_capsules = n_capsules
        self.capsule_dim = capsule_dim
        
        self.primary_capsules = nn.Linear(in_dim, n_capsules * capsule_dim)
        
        self.W = nn.Parameter(torch.randn(n_capsules, capsule_dim, out_dim) * 0.02)
        
    def squash(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * x / torch.sqrt(squared_norm + 1e-8)
    
    def forward(self, x: torch.Tensor, n_routing: int = 3) -> torch.Tensor:
        batch_size = x.shape[0]
        
        primary = self.primary_capsules(x)
        primary = primary.view(batch_size, self.n_capsules, self.capsule_dim)
        primary = self.squash(primary, dim=-1)
        
        u_hat = torch.einsum('bnd,ndo->bno', primary, self.W)
        
        b = torch.zeros(batch_size, self.n_capsules, 1, device=x.device)
        
        for _ in range(n_routing):
            c = F.softmax(b, dim=1)
            s = (c * u_hat.unsqueeze(-1)).sum(dim=1)
            v = self.squash(s.squeeze(-1))
            
            if _ < n_routing - 1:
                agreement = torch.einsum('bno,bo->bn', u_hat, v)
                b = b + agreement.unsqueeze(-1)
        
        return v


class NeuralTuringMachine(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim: int, memory_size: int = 128, memory_dim: int = 64):
        super().__init__()
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        
        self.controller = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        self.read_head = nn.Sequential(
            nn.Linear(hidden_dim, memory_dim + 1 + 1 + 3),
            nn.Sigmoid()
        )
        
        self.write_head = nn.Sequential(
            nn.Linear(hidden_dim, memory_dim + 1 + 1 + 3 + memory_dim + memory_dim),
            nn.Sigmoid()
        )
        
        self.output_layer = nn.Linear(hidden_dim + memory_dim, input_dim)
        
    def addressing(self, k: torch.Tensor, beta: torch.Tensor, g: torch.Tensor, 
                   s: torch.Tensor, gamma: torch.Tensor, memory: torch.Tensor, 
                   prev_w: torch.Tensor) -> torch.Tensor:
        
        similarity = F.cosine_similarity(k.unsqueeze(1), memory, dim=-1)
        content_w = F.softmax(beta * similarity, dim=-1)
        
        interpolated_w = g * content_w + (1 - g) * prev_w
        
        shifted_w = torch.zeros_like(interpolated_w)
        for i in range(3):
            shifted_w += s[:, i:i+1] * torch.roll(interpolated_w, shifts=i-1, dims=1)
        
        final_w = shifted_w ** gamma
        final_w = final_w / (final_w.sum(dim=-1, keepdim=True) + 1e-8)
        
        return final_w
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        device = x.device
        
        memory = torch.zeros(batch_size, self.memory_size, self.memory_dim, device=device)
        read_w = torch.zeros(batch_size, self.memory_size, device=device)
        read_w[:, 0] = 1.0
        
        x = x.unsqueeze(1)
        controller_out, _ = self.controller(x)
        controller_out = controller_out.squeeze(1)
        
        read_params = self.read_head(controller_out)
        k_r = read_params[:, :self.memory_dim]
        beta_r = read_params[:, self.memory_dim:self.memory_dim+1]
        g_r = read_params[:, self.memory_dim+1:self.memory_dim+2]
        s_r = read_params[:, self.memory_dim+2:self.memory_dim+5]
        gamma_r = read_params[:, self.memory_dim+5:self.memory_dim+6]
        
        read_w = self.addressing(k_r, beta_r, g_r, s_r, gamma_r, memory, read_w)
        read_vector = torch.sum(read_w.unsqueeze(-1) * memory, dim=1)
        
        output = self.output_layer(torch.cat([controller_out, read_vector], dim=-1))
        
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np


class DARTSCell(nn.Module):
    
    PRIMITIVES = [
        'none',
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5'
    ]
    
    def __init__(self, in_channels: int, out_channels: int, n_nodes: int = 4):
        super().__init__()
        self.n_nodes = n_nodes
        
        self.ops = nn.ModuleList()
        for i in range(n_nodes):
            for j in range(i + 2):
                op_list = nn.ModuleList([
                    self._get_op(primitive, in_channels, out_channels)
                    for primitive in self.PRIMITIVES
                ])
                self.ops.append(op_list)
        
        n_edges = sum(range(2, n_nodes + 2))
        self.alpha = nn.Parameter(torch.randn(n_edges, len(self.PRIMITIVES)))
        
    def _get_op(self, primitive: str, in_c: int, out_c: int) -> nn.Module:
        if primitive == 'none':
            return Zero(in_c)
        elif primitive == 'skip_connect':
            return Identity()
        elif primitive == 'max_pool_3x3':
            return nn.MaxPool1d(3, stride=1, padding=1)
        elif primitive == 'avg_pool_3x3':
            return nn.AvgPool1d(3, stride=1, padding=1)
        elif primitive == 'sep_conv_3x3':
            return SepConv(in_c, out_c, 3, 1, 1)
        elif primitive == 'sep_conv_5x5':
            return SepConv(in_c, out_c, 5, 1, 2)
        elif primitive == 'dil_conv_3x3':
            return DilConv(in_c, out_c, 3, 1, 2, 2)
        elif primitive == 'dil_conv_5x5':
            return DilConv(in_c, out_c, 5, 1, 4, 2)
        else:
            raise ValueError(f"Unknown primitive: {primitive}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states = [x, x]
        offset = 0
        
        for i in range(self.n_nodes):
            s = sum(
                sum(
                    F.softmax(self.alpha[offset + j], dim=-1)[k] * op(states[j])
                    for k, op in enumerate(self.ops[offset + j])
                )
                for j in range(i + 2)
            )
            offset += (i + 2)
            states.append(s)
        
        return torch.cat(states[2:], dim=1)
    
    def genotype(self) -> List[Tuple[str, int]]:
        gene = []
        offset = 0
        
        for i in range(self.n_nodes):
            edges = []
            for j in range(i + 2):
                weights = F.softmax(self.alpha[offset + j], dim=-1)
                k_best = torch.argmax(weights)
                edges.append((self.PRIMITIVES[k_best], j))
            offset += (i + 2)
            
            edges = sorted(edges, key=lambda x: x[1])[:2]
            gene.extend(edges)
        
        return gene


class Zero(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SepConv(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel: int, stride: int, padding: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv1d(in_c, in_c, kernel, stride, padding, groups=in_c, bias=False),
            nn.Conv1d(in_c, out_c, 1, bias=False),
            nn.BatchNorm1d(out_c),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        return self.op(x).squeeze(-1) if x.size(-1) == 1 else self.op(x)


class DilConv(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel: int, stride: int, padding: int, dilation: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_c),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        return self.op(x).squeeze(-1) if x.size(-1) == 1 else self.op(x)


class ProxylessNAS(nn.Module):
    
    def __init__(self, in_features: int, hidden_dims: List[int], n_ops: int = 5):
        super().__init__()
        self.n_ops = n_ops
        
        self.ops = nn.ModuleList()
        prev_dim = in_features
        
        for hidden_dim in hidden_dims:
            layer_ops = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU()
                )
                for _ in range(n_ops)
            ])
            self.ops.append(layer_ops)
            prev_dim = hidden_dim
        
        self.arch_params = nn.ParameterList([
            nn.Parameter(torch.randn(n_ops))
            for _ in range(len(hidden_dims))
        ])
        
    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        for layer_ops, arch_param in zip(self.ops, self.arch_params):
            weights = F.gumbel_softmax(arch_param, tau=temperature, hard=False)
            
            x = sum(w * op(x) for w, op in zip(weights, layer_ops))
        
        return x
    
    def derive_architecture(self) -> List[int]:
        return [torch.argmax(param).item() for param in self.arch_params]


class ENAS(nn.Module):
    
    def __init__(self, in_features: int, hidden_dim: int, n_layers: int, n_ops: int = 4):
        super().__init__()
        self.n_layers = n_layers
        self.n_ops = n_ops
        
        self.shared_weights = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(hidden_dim if i > 0 else in_features, hidden_dim)
                for _ in range(n_ops)
            ])
            for i in range(n_layers)
        ])
        
        self.controller = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, n_ops)
        
    def sample_architecture(self, batch_size: int = 1) -> List[int]:
        device = next(self.parameters()).device
        h = torch.zeros(1, batch_size, self.classifier.in_features, device=device)
        c = torch.zeros(1, batch_size, self.classifier.in_features, device=device)
        
        architecture = []
        
        for _ in range(self.n_layers):
            out, (h, c) = self.controller(h.transpose(0, 1), (h, c))
            logits = self.classifier(out.squeeze(1))
            
            sampled = torch.multinomial(F.softmax(logits, dim=-1), 1)
            architecture.append(sampled.item())
        
        return architecture
    
    def forward_with_arch(self, x: torch.Tensor, architecture: List[int]) -> torch.Tensor:
        for layer_idx, op_idx in enumerate(architecture):
            x = self.shared_weights[layer_idx][op_idx](x)
            x = F.relu(x)
        
        return x


class NASBench201Cell(nn.Module):
    
    OPS = {
        'none': lambda c: Zero(c),
        'skip_connect': lambda c: Identity(),
        'conv_1x1': lambda c: nn.Conv1d(c, c, 1),
        'conv_3x3': lambda c: nn.Conv1d(c, c, 3, padding=1),
        'avg_pool': lambda c: nn.AvgPool1d(3, stride=1, padding=1)
    }
    
    def __init__(self, channels: int):
        super().__init__()
        
        self.edges = nn.ModuleDict({
            '0->1': nn.ModuleList([op(channels) for op in self.OPS.values()]),
            '0->2': nn.ModuleList([op(channels) for op in self.OPS.values()]),
            '1->2': nn.ModuleList([op(channels) for op in self.OPS.values()]),
            '0->3': nn.ModuleList([op(channels) for op in self.OPS.values()]),
            '1->3': nn.ModuleList([op(channels) for op in self.OPS.values()]),
            '2->3': nn.ModuleList([op(channels) for op in self.OPS.values()])
        })
        
        self.arch_weights = nn.Parameter(torch.randn(6, len(self.OPS)))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        nodes = [x]
        
        edge_idx = 0
        for i in range(1, 4):
            node_input = []
            for j in range(i):
                edge_key = f'{j}->{i}'
                edge_ops = self.edges[edge_key]
                
                weights = F.softmax(self.arch_weights[edge_idx], dim=-1)
                edge_idx += 1
                
                node_input.append(
                    sum(w * op(nodes[j]) for w, op in zip(weights, edge_ops))
                )
            
            nodes.append(sum(node_input))
        
        return nodes[-1].squeeze(-1) if nodes[-1].size(-1) == 1 else nodes[-1]


class SuperNet(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, max_depth: int = 6, width_multiplier: float = 1.0):
        super().__init__()
        self.max_depth = max_depth
        
        base_channels = [64, 128, 256, 512, 512, 512]
        channels = [int(c * width_multiplier) for c in base_channels]
        
        self.cells = nn.ModuleList()
        prev_c = in_features
        
        for i in range(max_depth):
            cell = DARTSCell(prev_c, channels[i], n_nodes=4)
            self.cells.append(cell)
            prev_c = channels[i] * 4
        
        self.head = nn.Linear(prev_c, out_features)
        
    def forward(self, x: torch.Tensor, depth: Optional[int] = None) -> torch.Tensor:
        depth = depth or self.max_depth
        
        for i in range(depth):
            x = self.cells[i](x)
        
        return self.head(x.mean(dim=-1) if x.dim() > 2 else x)


class ArchitectureOptimizer:
    
    def __init__(self, model: nn.Module, model_lr: float = 0.001, arch_lr: float = 0.003):
        self.model = model
        
        model_params = []
        arch_params = []
        
        for name, param in model.named_parameters():
            if 'alpha' in name or 'arch' in name:
                arch_params.append(param)
            else:
                model_params.append(param)
        
        self.model_optimizer = torch.optim.Adam(model_params, lr=model_lr)
        self.arch_optimizer = torch.optim.Adam(arch_params, lr=arch_lr)
    
    def step(self, train_data: Tuple, val_data: Tuple, criterion):
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        self.arch_optimizer.zero_grad()
        pred_val = self.model(X_val)
        val_loss = criterion(pred_val, y_val)
        val_loss.backward()
        self.arch_optimizer.step()
        
        self.model_optimizer.zero_grad()
        pred_train = self.model(X_train)
        train_loss = criterion(pred_train, y_train)
        train_loss.backward()
        self.model_optimizer.step()
        
        return train_loss.item(), val_loss.item()

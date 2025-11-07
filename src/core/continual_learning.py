import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional
import copy


class ElasticWeightConsolidation:
    
    def __init__(self, model: nn.Module, lambda_: float = 0.4):
        self.model = model
        self.lambda_ = lambda_
        
        self.fisher_dict: Dict[str, Tensor] = {}
        self.optimal_params: Dict[str, Tensor] = {}
        
    def compute_fisher_information(self, dataloader, criterion):
        self.model.eval()
        
        fisher_dict = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        for X, y in dataloader:
            self.model.zero_grad()
            
            output = self.model(X)
            loss = criterion(output, y)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher_dict[n] += p.grad.pow(2)
        
        for n in fisher_dict:
            fisher_dict[n] /= len(dataloader)
        
        self.fisher_dict = fisher_dict
        self.optimal_params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
    
    def penalty(self) -> Tensor:
        loss = 0.0
        
        for n, p in self.model.named_parameters():
            if n in self.fisher_dict:
                loss += (self.fisher_dict[n] * (p - self.optimal_params[n]).pow(2)).sum()
        
        return self.lambda_ * loss


class ProgressiveNeuralNetwork(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        
        self.columns: List[nn.ModuleList] = []
        self.lateral_connections: List[nn.ModuleList] = []
        
        self._add_column()
    
    def _add_column(self):
        column = nn.ModuleList()
        
        column.append(nn.Linear(self.input_dim, self.hidden_dim))
        
        for _ in range(self.n_layers - 2):
            column.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        
        column.append(nn.Linear(self.hidden_dim, self.output_dim))
        
        self.columns.append(column)
        
        lateral = nn.ModuleList()
        n_prev_columns = len(self.columns) - 1
        
        for layer_idx in range(self.n_layers):
            layer_lateral = nn.ModuleList()
            for _ in range(n_prev_columns):
                layer_lateral.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            lateral.append(layer_lateral)
        
        self.lateral_connections.append(lateral)
    
    def forward(self, x: Tensor, column_idx: int = -1) -> Tensor:
        if column_idx < 0:
            column_idx = len(self.columns) + column_idx
        
        activations = []
        
        for col_idx in range(column_idx + 1):
            h = x
            col_activations = []
            
            for layer_idx, layer in enumerate(self.columns[col_idx]):
                if layer_idx > 0 and col_idx > 0:
                    lateral_input = sum(
                        self.lateral_connections[col_idx][layer_idx][prev_col_idx](activations[prev_col_idx][layer_idx - 1])
                        for prev_col_idx in range(col_idx)
                    )
                    h = layer(h) + lateral_input
                else:
                    h = layer(h)
                
                if layer_idx < len(self.columns[col_idx]) - 1:
                    h = F.relu(h)
                
                col_activations.append(h)
            
            activations.append(col_activations)
        
        return activations[-1][-1]
    
    def freeze_columns(self, except_last: bool = True):
        n_columns = len(self.columns) - (1 if except_last else 0)
        
        for col_idx in range(n_columns):
            for param in self.columns[col_idx].parameters():
                param.requires_grad = False
            
            for layer_lateral in self.lateral_connections[col_idx]:
                for lateral in layer_lateral:
                    for param in lateral.parameters():
                        param.requires_grad = False


class PackNet:
    
    def __init__(self, model: nn.Module, prune_ratio: float = 0.5):
        self.model = model
        self.prune_ratio = prune_ratio
        
        self.masks: Dict[str, Tensor] = {}
        self.task_masks: List[Dict[str, Tensor]] = []
    
    def compute_importance(self, dataloader, criterion) -> Dict[str, Tensor]:
        self.model.eval()
        
        importance = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        for X, y in dataloader:
            self.model.zero_grad()
            
            output = self.model(X)
            loss = criterion(output, y)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    importance[n] += p.grad.abs()
        
        for n in importance:
            importance[n] /= len(dataloader)
        
        return importance
    
    def prune_for_task(self, importance: Dict[str, Tensor]):
        task_mask = {}
        
        for n, imp in importance.items():
            if n in self.masks:
                available_weights = self.masks[n]
                imp_masked = imp * available_weights
            else:
                available_weights = torch.ones_like(imp)
                imp_masked = imp
            
            threshold = torch.quantile(imp_masked[available_weights.bool()], self.prune_ratio)
            
            new_mask = (imp_masked >= threshold).float()
            
            task_mask[n] = new_mask * available_weights
            
            if n in self.masks:
                self.masks[n] = self.masks[n] - new_mask
            else:
                self.masks[n] = torch.ones_like(imp) - new_mask
        
        self.task_masks.append(task_mask)
    
    def apply_mask(self, task_id: int):
        for n, p in self.model.named_parameters():
            if n in self.task_masks[task_id]:
                p.data *= self.task_masks[task_id][n]


class MemoryAwareReplay:
    
    def __init__(self, memory_size: int, input_dim: int):
        self.memory_size = memory_size
        self.memory_X = torch.zeros(memory_size, input_dim)
        self.memory_y = torch.zeros(memory_size)
        self.memory_ptr = 0
        self.memory_filled = False
        
    def add(self, X: Tensor, y: Tensor):
        batch_size = X.shape[0]
        
        if self.memory_ptr + batch_size <= self.memory_size:
            self.memory_X[self.memory_ptr:self.memory_ptr + batch_size] = X.detach()
            self.memory_y[self.memory_ptr:self.memory_ptr + batch_size] = y.detach()
            self.memory_ptr += batch_size
        else:
            self.memory_filled = True
            
            indices = torch.randperm(self.memory_size)[:batch_size]
            self.memory_X[indices] = X.detach()
            self.memory_y[indices] = y.detach()
    
    def sample(self, n_samples: int) -> Tuple[Tensor, Tensor]:
        if not self.memory_filled:
            max_idx = self.memory_ptr
        else:
            max_idx = self.memory_size
        
        indices = torch.randint(0, max_idx, (n_samples,))
        
        return self.memory_X[indices], self.memory_y[indices]


class DynamicArchitecture(nn.Module):
    
    def __init__(self, input_dim: int, initial_capacity: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.capacity = initial_capacity
        
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, initial_capacity)
        ])
        
        self.task_heads = nn.ModuleList()
        
    def add_capacity(self, additional_neurons: int):
        last_layer = self.layers[-1]
        old_out = last_layer.out_features
        new_out = old_out + additional_neurons
        
        new_layer = nn.Linear(last_layer.in_features, new_out)
        
        with torch.no_grad():
            new_layer.weight[:old_out] = last_layer.weight
            new_layer.bias[:old_out] = last_layer.bias
            
            nn.init.xavier_uniform_(new_layer.weight[old_out:])
            nn.init.zeros_(new_layer.bias[old_out:])
        
        self.layers[-1] = new_layer
        self.capacity = new_out
    
    def add_task_head(self, output_dim: int):
        head = nn.Linear(self.capacity, output_dim)
        self.task_heads.append(head)
        return len(self.task_heads) - 1
    
    def forward(self, x: Tensor, task_id: int) -> Tensor:
        for layer in self.layers:
            x = F.relu(layer(x))
        
        return self.task_heads[task_id](x)


class GradientEpisodicMemory:
    
    def __init__(self, memory_per_task: int, model: nn.Module):
        self.memory_per_task = memory_per_task
        self.model = model
        
        self.memory: Dict[int, List[Tuple[Tensor, Tensor]]] = {}
    
    def add_to_memory(self, task_id: int, X: Tensor, y: Tensor):
        if task_id not in self.memory:
            self.memory[task_id] = []
        
        if len(self.memory[task_id]) < self.memory_per_task:
            self.memory[task_id].append((X.detach(), y.detach()))
        else:
            idx = torch.randint(0, self.memory_per_task, (1,)).item()
            self.memory[task_id][idx] = (X.detach(), y.detach())
    
    def compute_reference_gradients(self, criterion) -> Dict[str, Tensor]:
        ref_grads = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        for task_id, examples in self.memory.items():
            for X, y in examples:
                self.model.zero_grad()
                
                output = self.model(X)
                loss = criterion(output, y)
                loss.backward()
                
                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        ref_grads[n] += p.grad.clone()
        
        return ref_grads
    
    def project_gradients(self, current_grads: Dict[str, Tensor], ref_grads: Dict[str, Tensor]):
        for n in current_grads:
            if n in ref_grads:
                dot_product = (current_grads[n] * ref_grads[n]).sum()
                
                if dot_product < 0:
                    ref_norm_sq = (ref_grads[n] ** 2).sum()
                    current_grads[n] = current_grads[n] - (dot_product / (ref_norm_sq + 1e-8)) * ref_grads[n]
        
        return current_grads


class ContinualLearner:
    
    def __init__(self, model: nn.Module, strategy: str = 'ewc'):
        self.model = model
        self.strategy = strategy
        
        if strategy == 'ewc':
            self.method = ElasticWeightConsolidation(model)
        elif strategy == 'packnet':
            self.method = PackNet(model)
        elif strategy == 'replay':
            input_dim = next(model.parameters()).shape[1] if hasattr(next(model.parameters()), 'shape') else 128
            self.method = MemoryAwareReplay(1000, input_dim)
        elif strategy == 'gem':
            self.method = GradientEpisodicMemory(50, model)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def before_task(self, task_id: int, dataloader, criterion):
        if self.strategy == 'ewc' and task_id > 0:
            self.method.compute_fisher_information(dataloader, criterion)
        elif self.strategy == 'packnet' and task_id > 0:
            importance = self.method.compute_importance(dataloader, criterion)
            self.method.prune_for_task(importance)
    
    def after_batch(self, X: Tensor, y: Tensor, task_id: int):
        if self.strategy == 'replay':
            self.method.add(X, y)
        elif self.strategy == 'gem':
            self.method.add_to_memory(task_id, X, y)
    
    def get_loss_adjustment(self) -> Tensor:
        if self.strategy == 'ewc':
            return self.method.penalty()
        return torch.tensor(0.0)

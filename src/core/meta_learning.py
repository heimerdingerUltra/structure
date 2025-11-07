import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Optional, Callable
from collections import OrderedDict
import copy


class MetaLearner:
    
    def __init__(self, model: nn.Module, inner_lr: float = 0.01, outer_lr: float = 0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
        
    def inner_loop(self, task_data: Tuple[Tensor, Tensor], n_steps: int = 5) -> OrderedDict:
        X_support, y_support = task_data
        
        fast_weights = OrderedDict(self.model.named_parameters())
        
        for _ in range(n_steps):
            logits = self._forward_with_params(X_support, fast_weights)
            loss = F.mse_loss(logits, y_support)
            
            grads = torch.autograd.grad(
                loss, 
                fast_weights.values(), 
                create_graph=True,
                allow_unused=True
            )
            
            fast_weights = OrderedDict(
                (name, param - self.inner_lr * grad if grad is not None else param)
                for ((name, param), grad) in zip(fast_weights.items(), grads)
            )
        
        return fast_weights
    
    def _forward_with_params(self, x: Tensor, params: OrderedDict) -> Tensor:
        x = x
        for name, param in params.items():
            if 'weight' in name:
                layer_name = name.split('.')[0]
                if hasattr(self.model, layer_name):
                    layer = getattr(self.model, layer_name)
                    if isinstance(layer, nn.Linear):
                        x = F.linear(x, param, params.get(name.replace('weight', 'bias')))
        return x
    
    def meta_update(self, tasks: List[Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]]):
        meta_loss = 0.0
        
        for support_set, query_set in tasks:
            fast_weights = self.inner_loop(support_set)
            
            X_query, y_query = query_set
            logits = self._forward_with_params(X_query, fast_weights)
            loss = F.mse_loss(logits, y_query)
            meta_loss += loss
        
        meta_loss = meta_loss / len(tasks)
        
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()


class ReptileLearner:
    
    def __init__(self, model: nn.Module, inner_lr: float = 0.01, outer_lr: float = 0.001):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        
    def adapt(self, task_data: Tuple[Tensor, Tensor], n_steps: int = 5) -> nn.Module:
        adapted_model = copy.deepcopy(self.model)
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        X, y = task_data
        
        for _ in range(n_steps):
            optimizer.zero_grad()
            logits = adapted_model(X)
            loss = F.mse_loss(logits, y)
            loss.backward()
            optimizer.step()
        
        return adapted_model
    
    def meta_update(self, tasks: List[Tuple[Tensor, Tensor]]):
        weights_before = [p.clone() for p in self.model.parameters()]
        
        adapted_weights = []
        for task_data in tasks:
            adapted_model = self.adapt(task_data)
            adapted_weights.append([p.clone() for p in adapted_model.parameters()])
        
        with torch.no_grad():
            for param, w_before in zip(self.model.parameters(), weights_before):
                avg_adapted = torch.stack([w[i] for w in adapted_weights for i, _ in enumerate(weights_before) if id(param) == id(w_before)]).mean(0)
                param.data = w_before + self.outer_lr * (avg_adapted - w_before)


class ProtoNet(nn.Module):
    
    def __init__(self, encoder: nn.Module, distance_metric: str = 'euclidean'):
        super().__init__()
        self.encoder = encoder
        self.distance_metric = distance_metric
        
    def compute_prototypes(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        unique_labels = torch.unique(labels)
        prototypes = torch.stack([
            embeddings[labels == label].mean(dim=0)
            for label in unique_labels
        ])
        return prototypes
    
    def compute_distances(self, query_embeddings: Tensor, prototypes: Tensor) -> Tensor:
        if self.distance_metric == 'euclidean':
            distances = torch.cdist(query_embeddings, prototypes, p=2)
        elif self.distance_metric == 'cosine':
            query_norm = F.normalize(query_embeddings, dim=-1)
            proto_norm = F.normalize(prototypes, dim=-1)
            distances = 1 - torch.mm(query_norm, proto_norm.t())
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distances
    
    def forward(self, support_x: Tensor, support_y: Tensor, query_x: Tensor) -> Tensor:
        support_embeddings = self.encoder(support_x)
        query_embeddings = self.encoder(query_x)
        
        prototypes = self.compute_prototypes(support_embeddings, support_y)
        
        distances = self.compute_distances(query_embeddings, prototypes)
        
        logits = -distances
        
        return logits


class MatchingNet(nn.Module):
    
    def __init__(self, encoder: nn.Module, hidden_dim: int):
        super().__init__()
        self.encoder = encoder
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
    def forward(self, support_x: Tensor, support_y: Tensor, query_x: Tensor) -> Tensor:
        support_embeddings = self.encoder(support_x)
        query_embeddings = self.encoder(query_x)
        
        attended_query, attention_weights = self.attention(
            query_embeddings.unsqueeze(1),
            support_embeddings.unsqueeze(0).expand(query_embeddings.size(0), -1, -1),
            support_embeddings.unsqueeze(0).expand(query_embeddings.size(0), -1, -1)
        )
        
        similarities = torch.mm(
            attended_query.squeeze(1),
            support_embeddings.t()
        )
        
        attention = F.softmax(similarities, dim=-1)
        
        predictions = torch.mm(attention, support_y.unsqueeze(-1) if support_y.dim() == 1 else support_y)
        
        return predictions.squeeze(-1)


class TaskEmbedding(nn.Module):
    
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        
        self.aggregator = nn.Sequential(
            nn.Linear(input_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, support_x: Tensor, support_y: Tensor) -> Tensor:
        support_stats = torch.cat([
            support_x.mean(dim=0),
            support_x.std(dim=0)
        ])
        
        task_embedding = self.aggregator(support_stats)
        
        return task_embedding


class ConditionalNeuralProcess(nn.Module):
    
    def __init__(self, x_dim: int, y_dim: int, r_dim: int, z_dim: int, hidden_dim: int):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, r_dim)
        )
        
        self.aggregator = lambda r: r.mean(dim=1)
        
        self.latent_encoder = nn.Sequential(
            nn.Linear(r_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim * 2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(x_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, y_dim * 2)
        )
        
    def encode(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        xy = torch.cat([x, y], dim=-1)
        r = self.encoder(xy)
        r_agg = self.aggregator(r)
        
        z_params = self.latent_encoder(r_agg)
        z_mu, z_logvar = torch.chunk(z_params, 2, dim=-1)
        
        return z_mu, z_logvar
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, x: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:
        xz = torch.cat([x, z.unsqueeze(1).expand(-1, x.size(1), -1)], dim=-1)
        y_params = self.decoder(xz)
        y_mu, y_logvar = torch.chunk(y_params, 2, dim=-1)
        
        return y_mu, y_logvar
    
    def forward(self, context_x: Tensor, context_y: Tensor, target_x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        z_mu_context, z_logvar_context = self.encode(context_x, context_y)
        z = self.reparameterize(z_mu_context, z_logvar_context)
        
        y_mu, y_logvar = self.decode(target_x, z)
        
        return y_mu, y_logvar, z_mu_context, z_logvar_context


class MetaSGD(nn.Module):
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        
        self.meta_lr = nn.ParameterDict({
            name: nn.Parameter(torch.ones_like(param) * 0.01)
            for name, param in model.named_parameters()
        })
        
    def adapt(self, support_x: Tensor, support_y: Tensor, n_steps: int = 5) -> OrderedDict:
        fast_weights = OrderedDict(self.model.named_parameters())
        
        for _ in range(n_steps):
            logits = self._forward_with_params(support_x, fast_weights)
            loss = F.mse_loss(logits, support_y)
            
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            
            fast_weights = OrderedDict(
                (name, param - self.meta_lr[name] * grad)
                for ((name, param), grad) in zip(fast_weights.items(), grads)
            )
        
        return fast_weights
    
    def _forward_with_params(self, x: Tensor, params: OrderedDict) -> Tensor:
        return self.model(x)


class ANILNetwork(nn.Module):
    
    def __init__(self, feature_extractor: nn.Module, head: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head = head
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
    def forward(self, x: Tensor) -> Tensor:
        features = self.feature_extractor(x)
        return self.head(features)
    
    def adapt_head(self, support_x: Tensor, support_y: Tensor, n_steps: int = 10, lr: float = 0.01):
        optimizer = torch.optim.SGD(self.head.parameters(), lr=lr)
        
        for _ in range(n_steps):
            optimizer.zero_grad()
            predictions = self(support_x)
            loss = F.mse_loss(predictions, support_y)
            loss.backward()
            optimizer.step()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from typing import Tuple, Optional
import math


class BayesianLinear(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features) * 0.02 - 5)
        
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.randn(out_features) * 0.02 - 5)
        
        self.prior_sigma = prior_sigma
        
        self.weight_sampler = Normal(0, prior_sigma)
        self.bias_sampler = Normal(0, prior_sigma)
        
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
            
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        kl_weight = self._kl_normal(self.weight_mu, weight_sigma, 0, self.prior_sigma)
        kl_bias = self._kl_normal(self.bias_mu, bias_sigma, 0, self.prior_sigma)
        
        return kl_weight + kl_bias
    
    def _kl_normal(self, mu1: torch.Tensor, sigma1: torch.Tensor, mu2: float, sigma2: float) -> torch.Tensor:
        kl = torch.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5
        return kl.sum()


class MonteCarloDropout(nn.Module):
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, self.p, training=True)


class DeepEnsemble(nn.Module):
    
    def __init__(self, base_model: nn.Module, n_models: int = 5):
        super().__init__()
        self.models = nn.ModuleList([
            self._copy_model(base_model) for _ in range(n_models)
        ])
        
    def _copy_model(self, model: nn.Module) -> nn.Module:
        import copy
        return copy.deepcopy(model)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions = torch.stack([model(x) for model in self.models])
        
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std
    
    def train_model(self, model_idx: int):
        for i, model in enumerate(self.models):
            if i == model_idx:
                model.train()
            else:
                model.eval()


class ConcreteDropout(nn.Module):
    
    def __init__(self, weight_regularizer: float = 1e-6, dropout_regularizer: float = 1e-5):
        super().__init__()
        
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        
        init_min = 0.1
        init_max = 0.1
        init_p = (init_min + init_max) / 2
        
        self.p_logit = nn.Parameter(torch.tensor(math.log(init_p / (1 - init_p))))
        
    def forward(self, x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        p = torch.sigmoid(self.p_logit)
        
        out = layer(self._concrete_dropout(x, p))
        
        return out
    
    def _concrete_dropout(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        eps = 1e-7
        temp = 0.1
        
        unif_noise = torch.rand_like(x)
        
        drop_prob = (
            torch.log(p + eps)
            - torch.log(1 - p + eps)
            + torch.log(unif_noise + eps)
            - torch.log(1 - unif_noise + eps)
        )
        
        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        
        return x * random_tensor
    
    def regularization(self, layer: nn.Module, input_shape: int) -> torch.Tensor:
        p = torch.sigmoid(self.p_logit)
        
        weight_reg = self.weight_regularizer * torch.sum(layer.weight ** 2) / (1 - p)
        dropout_reg = p * torch.log(p) + (1 - p) * torch.log(1 - p)
        dropout_reg = self.dropout_regularizer * input_shape * dropout_reg
        
        return weight_reg + dropout_reg


class SwagOptimizer:
    
    def __init__(self, base_optimizer, max_rank: int = 20, swa_start: int = 100):
        self.base_optimizer = base_optimizer
        self.max_rank = max_rank
        self.swa_start = swa_start
        
        self.n_averaged = 0
        
        self.mean = {}
        self.sq_mean = {}
        self.cov_mat_sqrt = {}
        
    def update(self, step: int):
        if step >= self.swa_start:
            if self.n_averaged == 0:
                for group in self.base_optimizer.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        
                        self.mean[p] = p.data.clone()
                        self.sq_mean[p] = p.data.clone() ** 2
                        self.cov_mat_sqrt[p] = []
            else:
                for group in self.base_optimizer.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        
                        self.mean[p] = (self.n_averaged * self.mean[p] + p.data) / (self.n_averaged + 1)
                        self.sq_mean[p] = (self.n_averaged * self.sq_mean[p] + p.data ** 2) / (self.n_averaged + 1)
                        
                        dev = p.data - self.mean[p]
                        self.cov_mat_sqrt[p].append(dev.view(-1))
                        
                        if len(self.cov_mat_sqrt[p]) > self.max_rank:
                            self.cov_mat_sqrt[p].pop(0)
            
            self.n_averaged += 1
    
    def sample(self, scale: float = 0.5, cov: bool = True) -> None:
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p not in self.mean:
                    continue
                
                var = torch.clamp(self.sq_mean[p] - self.mean[p] ** 2, min=1e-8)
                
                p.data = self.mean[p] + scale * torch.randn_like(p.data) * var.sqrt()
                
                if cov and len(self.cov_mat_sqrt[p]) > 0:
                    cov_mat = torch.stack(self.cov_mat_sqrt[p])
                    
                    z = torch.randn(len(self.cov_mat_sqrt[p]), device=p.device)
                    cov_sample = (z.unsqueeze(1) * cov_mat).sum(0).view_as(p.data)
                    
                    p.data += scale / ((self.max_rank + 1) ** 0.5) * cov_sample


class LaplaceBridge:
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.hessian = None
        
    def compute_hessian(self, dataloader, criterion):
        self.model.eval()
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in params)
        
        hessian = torch.zeros(n_params, n_params)
        
        for X, y in dataloader:
            self.model.zero_grad()
            
            output = self.model(X)
            loss = criterion(output, y)
            
            grads = torch.autograd.grad(loss, params, create_graph=True)
            grad_vec = torch.cat([g.view(-1) for g in grads])
            
            for i in range(n_params):
                grad2 = torch.autograd.grad(grad_vec[i], params, retain_graph=True)
                grad2_vec = torch.cat([g.contiguous().view(-1) for g in grad2])
                hessian[i] = grad2_vec
        
        self.hessian = hessian / len(dataloader)
    
    def sample_posterior(self, n_samples: int = 100) -> list:
        if self.hessian is None:
            raise ValueError("Must compute Hessian first")
        
        eigenvalues, eigenvectors = torch.linalg.eigh(self.hessian)
        eigenvalues = torch.clamp(eigenvalues, min=1e-8)
        
        samples = []
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        param_vec = torch.cat([p.view(-1) for p in params])
        
        for _ in range(n_samples):
            noise = torch.randn_like(param_vec)
            
            sample = param_vec + eigenvectors @ (noise / eigenvalues.sqrt())
            samples.append(sample)
        
        return samples


class EvidentialRegression(nn.Module):
    
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        
        last_dim = self._get_last_layer_dim(base_model)
        
        self.evidence_head = nn.Sequential(
            nn.Linear(last_dim, last_dim // 2),
            nn.ReLU(),
            nn.Linear(last_dim // 2, 4)
        )
        
    def _get_last_layer_dim(self, model: nn.Module) -> int:
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Linear):
                return module.out_features
        return 128
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.base_model(x)
        
        evidence = self.evidence_head(features)
        
        gamma, nu, alpha, beta = torch.split(evidence, 1, dim=-1)
        
        nu = F.softplus(nu) + 1
        alpha = F.softplus(alpha) + 1
        beta = F.softplus(beta)
        
        return gamma.squeeze(-1), nu.squeeze(-1), alpha.squeeze(-1), beta.squeeze(-1)
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma, nu, alpha, beta = self.forward(x)
        
        mean = gamma
        
        aleatoric = beta / (alpha - 1)
        
        epistemic = beta / (nu * (alpha - 1))
        
        total_uncertainty = aleatoric + epistemic
        
        return mean, aleatoric, epistemic

import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberLoss(nn.Module):
    
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(pred, target, reduction=self.reduction, delta=self.delta)


class QuantileLoss(nn.Module):
    
    def __init__(self, quantiles: list = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses = []
        
        for i, q in enumerate(self.quantiles):
            errors = target - pred[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors))
        
        return torch.mean(torch.sum(torch.stack(losses, dim=1), dim=1))


class WingLoss(nn.Module):
    
    def __init__(self, omega: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * torch.log(torch.tensor(1.0 + self.omega / self.epsilon))
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        
        loss = torch.where(
            diff < self.omega,
            self.omega * torch.log(1.0 + diff / self.epsilon),
            diff - self.C
        )
        
        return torch.mean(loss)


class AdaptiveWingLoss(nn.Module):
    
    def __init__(self, omega: float = 14.0, theta: float = 0.5, epsilon: float = 1.0, alpha: float = 2.1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target))) * \
            (self.alpha - target) * torch.pow(self.theta / self.epsilon, self.alpha - target - 1) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - target))
        
        loss = torch.where(
            diff < self.theta,
            self.omega * torch.log(1 + torch.pow(diff / self.epsilon, self.alpha - target)),
            A * diff - C
        )
        
        return torch.mean(loss)


class FocalLoss(nn.Module):
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = (pred - target) ** 2
        focal_weight = self.alpha * torch.pow(torch.abs(pred - target), self.gamma)
        loss = focal_weight * mse
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        return loss


class LogCoshLoss(nn.Module):
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.log(torch.cosh(diff)))


class XTanhLoss(nn.Module):
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(diff * torch.tanh(diff))


class CombinedLoss(nn.Module):
    
    def __init__(
        self,
        loss_types: list = ['huber', 'mse'],
        weights: list = [0.7, 0.3],
        delta: float = 1.0
    ):
        super().__init__()
        self.loss_types = loss_types
        self.weights = weights
        self.delta = delta
        
        assert len(loss_types) == len(weights)
        assert abs(sum(weights) - 1.0) < 1e-6
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        
        for loss_type, weight in zip(self.loss_types, self.weights):
            if loss_type == 'huber':
                loss = F.huber_loss(pred, target, delta=self.delta)
            elif loss_type == 'mse':
                loss = F.mse_loss(pred, target)
            elif loss_type == 'mae':
                loss = F.l1_loss(pred, target)
            elif loss_type == 'smooth_l1':
                loss = F.smooth_l1_loss(pred, target)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            total_loss += weight * loss
        
        return total_loss


class TukeyLoss(nn.Module):
    
    def __init__(self, c: float = 4.685):
        super().__init__()
        self.c = c
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(pred - target)
        
        loss = torch.where(
            diff <= self.c,
            (self.c ** 2 / 6.0) * (1.0 - (1.0 - (diff / self.c) ** 2) ** 3),
            self.c ** 2 / 6.0
        )
        
        return torch.mean(loss)


class WeightedMSELoss(nn.Module):
    
    def __init__(self, weight_fn=None):
        super().__init__()
        self.weight_fn = weight_fn or (lambda x: torch.ones_like(x))
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = self.weight_fn(target)
        mse = (pred - target) ** 2
        weighted_mse = weights * mse
        return torch.mean(weighted_mse)


class PercentileLoss(nn.Module):
    
    def __init__(self, percentile: float = 95.0):
        super().__init__()
        self.percentile = percentile
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        errors = torch.abs(pred - target)
        k = int(len(errors) * (self.percentile / 100.0))
        return torch.kthvalue(errors, k)[0]


class AsymmetricLoss(nn.Module):
    
    def __init__(self, over_penalty: float = 2.0, under_penalty: float = 1.0):
        super().__init__()
        self.over_penalty = over_penalty
        self.under_penalty = under_penalty
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        
        loss = torch.where(
            diff > 0,
            self.over_penalty * diff ** 2,
            self.under_penalty * diff ** 2
        )
        
        return torch.mean(loss)


class CVaRLoss(nn.Module):
    
    def __init__(self, alpha: float = 0.95, lambda_cvar: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.lambda_cvar = lambda_cvar
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        errors = torch.abs(pred - target)
        mse = torch.mean(errors ** 2)
        
        k = int(len(errors) * self.alpha)
        topk_errors = torch.topk(errors, k)[0]
        cvar = torch.mean(topk_errors)
        
        return (1 - self.lambda_cvar) * mse + self.lambda_cvar * cvar


class RMSELoss(nn.Module):
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((pred - target) ** 2)
        return torch.sqrt(mse + self.eps)


def get_loss_function(name: str, **kwargs):
    losses = {
        'huber': HuberLoss,
        'quantile': QuantileLoss,
        'wing': WingLoss,
        'adaptive_wing': AdaptiveWingLoss,
        'focal': FocalLoss,
        'logcosh': LogCoshLoss,
        'xtanh': XTanhLoss,
        'combined': CombinedLoss,
        'tukey': TukeyLoss,
        'weighted_mse': WeightedMSELoss,
        'percentile': PercentileLoss,
        'asymmetric': AsymmetricLoss,
        'cvar': CVaRLoss,
        'rmse': RMSELoss,
        'mse': lambda **k: nn.MSELoss(),
        'mae': lambda **k: nn.L1Loss(),
        'smooth_l1': lambda **k: nn.SmoothL1Loss(),
    }
    
    loss_class = losses.get(name.lower())
    if loss_class is None:
        raise ValueError(f"Unknown loss function: {name}")
    
    return loss_class(**kwargs)

import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW, Adam
from typing import Dict, Any


class LAMB(Optimizer):
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(LAMB, self).__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] * (bias_correction2 ** 0.5) / bias_correction1
                
                adam_step = exp_avg / (exp_avg_sq.sqrt() + group['eps'])
                
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])
                
                weight_norm = p.data.norm()
                adam_norm = adam_step.norm()
                
                if weight_norm > 0 and adam_norm > 0:
                    trust_ratio = weight_norm / adam_norm
                else:
                    trust_ratio = 1.0
                
                p.data.add_(adam_step, alpha=-step_size * trust_ratio)
        
        return loss


class OptimizerFactory:
    
    @staticmethod
    def create_optimizer(
        model: nn.Module,
        optimizer_name: str,
        learning_rate: float,
        weight_decay: float = 0.01,
        **kwargs
    ) -> Optimizer:
        
        params = model.parameters()
        
        if optimizer_name.lower() == 'adamw':
            return AdamW(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8)
            )
        
        elif optimizer_name.lower() == 'adam':
            return Adam(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8)
            )
        
        elif optimizer_name.lower() == 'lamb':
            return LAMB(
                params,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=kwargs.get('betas', (0.9, 0.999)),
                eps=kwargs.get('eps', 1e-8)
            )
        
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    @staticmethod
    def create_scheduler(
        optimizer: Optimizer,
        scheduler_name: str,
        epochs: int,
        **kwargs
    ):
        
        if scheduler_name.lower() == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        
        elif scheduler_name.lower() == 'cosine_warmup':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=kwargs.get('T_0', 10),
                T_mult=kwargs.get('T_mult', 2),
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        
        elif scheduler_name.lower() == 'onecycle':
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=kwargs.get('max_lr', optimizer.param_groups[0]['lr'] * 10),
                epochs=epochs,
                steps_per_epoch=kwargs.get('steps_per_epoch', 100),
                pct_start=kwargs.get('pct_start', 0.3),
                div_factor=kwargs.get('div_factor', 25),
                final_div_factor=kwargs.get('final_div_factor', 1e4)
            )
        
        elif scheduler_name.lower() == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 10),
                min_lr=kwargs.get('min_lr', 1e-6)
            )
        
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")


def configure_training(
    model: nn.Module,
    optimizer_name: str = 'adamw',
    scheduler_name: str = 'cosine_warmup',
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 200,
    **kwargs
):
    
    optimizer = OptimizerFactory.create_optimizer(
        model,
        optimizer_name,
        learning_rate,
        weight_decay,
        **kwargs.get('optimizer_kwargs', {})
    )
    
    scheduler = OptimizerFactory.create_scheduler(
        optimizer,
        scheduler_name,
        epochs,
        **kwargs.get('scheduler_kwargs', {})
    )
    
    return optimizer, scheduler

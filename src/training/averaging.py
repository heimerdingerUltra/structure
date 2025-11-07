import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional


class EMA:
    
    def __init__(self, model: nn.Module, decay: float = 0.9999, device: str = 'cuda'):
        self.model = model
        self.decay = decay
        self.device = device
        
        self.shadow = deepcopy(model)
        self.shadow.eval()
        
        for param in self.shadow.parameters():
            param.detach_()
        
        self.backup = {}
        self.num_updates = 0
        
    def update(self):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        with torch.no_grad():
            model_params = dict(self.model.named_parameters())
            shadow_params = dict(self.shadow.named_parameters())
            
            for name in model_params.keys():
                shadow_params[name].data.mul_(decay).add_(
                    model_params[name].data,
                    alpha=1 - decay
                )
    
    def apply_shadow(self):
        self.backup = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
        
        shadow_params = dict(self.shadow.named_parameters())
        model_params = dict(self.model.named_parameters())
        
        for name in model_params.keys():
            model_params[name].data.copy_(shadow_params[name].data)
    
    def restore(self):
        model_params = dict(self.model.named_parameters())
        
        for name in model_params.keys():
            model_params[name].data.copy_(self.backup[name])
        
        self.backup = {}
    
    def state_dict(self):
        return {
            'shadow': self.shadow.state_dict(),
            'num_updates': self.num_updates
        }
    
    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict['shadow'])
        self.num_updates = state_dict['num_updates']


class SWA:
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        
        self.swa_model = deepcopy(model)
        self.swa_model.eval()
        
        for param in self.swa_model.parameters():
            param.detach_()
        
        self.n_averaged = 0
    
    def update(self):
        self.n_averaged += 1
        
        with torch.no_grad():
            model_params = dict(self.model.named_parameters())
            swa_params = dict(self.swa_model.named_parameters())
            
            for name in model_params.keys():
                swa_params[name].data.mul_(self.n_averaged - 1).add_(
                    model_params[name].data
                ).div_(self.n_averaged)
    
    def update_bn(self, loader):
        torch.optim.swa_utils.update_bn(loader, self.swa_model)
    
    def state_dict(self):
        return {
            'swa_model': self.swa_model.state_dict(),
            'n_averaged': self.n_averaged
        }
    
    def load_state_dict(self, state_dict):
        self.swa_model.load_state_dict(state_dict['swa_model'])
        self.n_averaged = state_dict['n_averaged']


class ModelEnsembleAveraging:
    
    def __init__(self, models: list):
        self.models = models
        for model in self.models:
            model.eval()
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred = model(x)
                predictions.append(pred)
        
        return torch.stack(predictions).mean(dim=0)
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> tuple:
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        return mean, std


class PolyakAveraging:
    
    def __init__(self, model: nn.Module, alpha: float = 0.999):
        self.model = model
        self.alpha = alpha
        
        self.averaged_model = deepcopy(model)
        self.averaged_model.eval()
        
        for param in self.averaged_model.parameters():
            param.detach_()
    
    def update(self):
        with torch.no_grad():
            model_params = dict(self.model.named_parameters())
            avg_params = dict(self.averaged_model.named_parameters())
            
            for name in model_params.keys():
                avg_params[name].data.mul_(self.alpha).add_(
                    model_params[name].data,
                    alpha=1 - self.alpha
                )
    
    def state_dict(self):
        return self.averaged_model.state_dict()
    
    def load_state_dict(self, state_dict):
        self.averaged_model.load_state_dict(state_dict)

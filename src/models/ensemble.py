import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import numpy as np


class NeuralEnsemble(nn.Module):
    def __init__(self, n_models: int, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        
        layers = []
        input_dim = n_models
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        return self.network(predictions).squeeze(-1)


class EnsembleSystem:
    def __init__(self, models: Dict[str, nn.Module], device: str = 'cuda'):
        self.models = {name: model.to(device) for name, model in models.items()}
        self.device = device
        
        n_models = len(models)
        self.combiner = NeuralEnsemble(n_models).to(device)
        
    def get_base_predictions(self, X: torch.Tensor) -> torch.Tensor:
        predictions = []
        
        for model in self.models.values():
            model.eval()
            with torch.no_grad():
                pred = model(X)
                predictions.append(pred)
        
        return torch.stack(predictions, dim=1)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        base_preds = self.get_base_predictions(X)
        return self.combiner(base_preds)
    
    def train_combiner(self, train_loader, val_loader, optimizer, criterion, 
                      n_epochs: int = 50, patience: int = 10):
        for model in self.models.values():
            model.eval()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            self.combiner.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                base_preds = self.get_base_predictions(X_batch)
                
                optimizer.zero_grad()
                pred = self.combiner(base_preds)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            self.combiner.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    base_preds = self.get_base_predictions(X_batch)
                    pred = self.combiner(base_preds)
                    loss = criterion(pred, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.combiner.state_dict()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        self.combiner.load_state_dict(best_state)
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        X = X.to(self.device)
        with torch.no_grad():
            pred = self.forward(X)
        return pred.cpu().numpy()


class ModelFactory:
    @staticmethod
    def create_tabpfn(n_features: int, config: Dict) -> nn.Module:
        from src.models.architectures.tabpfn import TabPFN
        return TabPFN(
            n_features=n_features,
            d_model=config.get('d_model', 512),
            n_layers=config.get('n_layers', 12),
            n_heads=config.get('n_heads', 8),
            mlp_ratio=config.get('mlp_ratio', 4.0),
            dropout=config.get('dropout', 0.1)
        )
    
    @staticmethod
    def create_mamba(n_features: int, config: Dict) -> nn.Module:
        from src.models.architectures.mamba import Mamba
        return Mamba(
            n_features=n_features,
            d_model=config.get('d_model', 256),
            n_layers=config.get('n_layers', 8),
            d_state=config.get('d_state', 16),
            expand=config.get('expand', 2),
            dropout=config.get('dropout', 0.1)
        )
    
    @staticmethod
    def create_xlstm(n_features: int, config: Dict) -> nn.Module:
        from src.models.architectures.xlstm import xLSTM
        return xLSTM(
            n_features=n_features,
            hidden_size=config.get('hidden_size', 256),
            n_layers=config.get('n_layers', 4),
            use_mlstm=config.get('use_mlstm', True),
            dropout=config.get('dropout', 0.1)
        )
    
    @staticmethod
    def create_hypermixer(n_features: int, config: Dict) -> nn.Module:
        from src.models.architectures.hypermixer import HyperMixer
        return HyperMixer(
            n_features=n_features,
            dim=config.get('dim', 256),
            n_blocks=config.get('n_blocks', 8),
            patch_size=config.get('patch_size', 1),
            expansion_factor=config.get('expansion_factor', 4),
            dropout=config.get('dropout', 0.1)
        )
    
    @staticmethod
    def create_ttt(n_features: int, config: Dict) -> nn.Module:
        from src.models.architectures.ttt import TTT
        return TTT(
            n_features=n_features,
            d_model=config.get('d_model', 256),
            n_layers=config.get('n_layers', 6),
            n_heads=config.get('n_heads', 8),
            d_ff=config.get('d_ff', 1024),
            dropout=config.get('dropout', 0.1)
        )
    
    @staticmethod
    def create_modern_tcn(n_features: int, config: Dict) -> nn.Module:
        from src.models.architectures.modern_tcn import ModernTCN
        return ModernTCN(
            n_features=n_features,
            channels=config.get('channels', [256, 256, 256, 256]),
            kernel_size=config.get('kernel_size', 3),
            dropout=config.get('dropout', 0.1),
            use_se=config.get('use_se', True)
        )

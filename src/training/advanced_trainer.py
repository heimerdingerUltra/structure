import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Optional, Callable
import numpy as np
import time
from pathlib import Path
from abc import ABC, abstractmethod


class Callback(ABC):
    
    @abstractmethod
    def on_epoch_start(self, epoch: int, trainer: 'AdvancedTrainer') -> None:
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, trainer: 'AdvancedTrainer', metrics: Dict) -> None:
        pass
    
    @abstractmethod
    def on_train_start(self, trainer: 'AdvancedTrainer') -> None:
        pass
    
    @abstractmethod
    def on_train_end(self, trainer: 'AdvancedTrainer') -> None:
        pass


class EarlyStopping(Callback):
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        
    def on_train_start(self, trainer: 'AdvancedTrainer') -> None:
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def on_epoch_start(self, epoch: int, trainer: 'AdvancedTrainer') -> None:
        pass
    
    def on_epoch_end(self, epoch: int, trainer: 'AdvancedTrainer', metrics: Dict) -> None:
        score = metrics['val_loss']
        
        if self.best_score is None:
            self.best_score = score
            trainer.save_checkpoint('best')
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            trainer.save_checkpoint('best')
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
    
    def on_train_end(self, trainer: 'AdvancedTrainer') -> None:
        pass
    
    def _is_improvement(self, score: float) -> bool:
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


class ModelCheckpoint(Callback):
    
    def __init__(self, checkpoint_dir: Path, save_frequency: int = 10):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = save_frequency
    
    def on_train_start(self, trainer: 'AdvancedTrainer') -> None:
        pass
    
    def on_epoch_start(self, epoch: int, trainer: 'AdvancedTrainer') -> None:
        pass
    
    def on_epoch_end(self, epoch: int, trainer: 'AdvancedTrainer', metrics: Dict) -> None:
        if (epoch + 1) % self.save_frequency == 0:
            trainer.save_checkpoint(f'epoch_{epoch+1}')
    
    def on_train_end(self, trainer: 'AdvancedTrainer') -> None:
        pass


class LearningRateLogger(Callback):
    
    def __init__(self):
        self.lrs = []
    
    def on_train_start(self, trainer: 'AdvancedTrainer') -> None:
        self.lrs = []
    
    def on_epoch_start(self, epoch: int, trainer: 'AdvancedTrainer') -> None:
        pass
    
    def on_epoch_end(self, epoch: int, trainer: 'AdvancedTrainer', metrics: Dict) -> None:
        lr = trainer.optimizer.param_groups[0]['lr']
        self.lrs.append(lr)
    
    def on_train_end(self, trainer: 'AdvancedTrainer') -> None:
        pass


class AdvancedTrainer:
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cuda',
        gradient_clip: float = 1.0,
        mixed_precision: bool = True,
        accumulation_steps: int = 1,
        checkpoint_dir: Optional[Path] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.gradient_clip = gradient_clip
        self.mixed_precision = mixed_precision
        self.accumulation_steps = accumulation_steps
        self.checkpoint_dir = checkpoint_dir
        
        if mixed_precision:
            self.scaler = GradScaler()
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [],
            'val_mae': []
        }
        
        self.callbacks: List[Callback] = []
        self.scheduler = None
        
    def add_callback(self, callback: Callback) -> None:
        self.callbacks.append(callback)
    
    def set_scheduler(self, scheduler) -> None:
        self.scheduler = scheduler
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            if self.mixed_precision:
                with autocast():
                    predictions = self.model(X_batch)
                    loss = self.criterion(predictions, y_batch)
                    loss = loss / self.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    if self.gradient_clip:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip
                        )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                loss = loss / self.accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    if self.gradient_clip:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.gradient_clip
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                if self.mixed_precision:
                    with autocast():
                        predictions = self.model(X_batch)
                        loss = self.criterion(predictions, y_batch)
                else:
                    predictions = self.model(X_batch)
                    loss = self.criterion(predictions, y_batch)
                
                total_loss += loss.item()
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
        
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        
        mse = np.mean((preds - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds - targets))
        mape = np.mean(np.abs((targets - preds) / (targets + 1e-10))) * 100
        r2 = 1 - np.sum((targets - preds) ** 2) / np.sum((targets - targets.mean()) ** 2)
        
        return {
            'loss': total_loss / len(val_loader),
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        verbose: int = 10
    ) -> Dict:
        
        for callback in self.callbacks:
            callback.on_train_start(self)
        
        for epoch in range(n_epochs):
            for callback in self.callbacks:
                callback.on_epoch_start(epoch, self)
            
            start = time.time()
            
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_rmse'].append(val_metrics['rmse'])
            self.history['val_mae'].append(val_metrics['mae'])
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            if (epoch + 1) % verbose == 0:
                elapsed = time.time() - start
                lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch+1}/{n_epochs} - {elapsed:.1f}s - "
                    f"lr: {lr:.2e} - "
                    f"train: {train_loss:.4f} - "
                    f"val: {val_metrics['loss']:.4f} - "
                    f"rmse: {val_metrics['rmse']:.4f} - "
                    f"mae: {val_metrics['mae']:.4f}"
                )
            
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, self, {'val_loss': val_metrics['loss']})
            
            early_stopping = next(
                (cb for cb in self.callbacks if isinstance(cb, EarlyStopping)),
                None
            )
            if early_stopping and early_stopping.should_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        for callback in self.callbacks:
            callback.on_train_end(self)
        
        return self.history
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        self.model.eval()
        X = X.to(self.device)
        
        with torch.no_grad():
            if self.mixed_precision:
                with autocast():
                    pred = self.model(X)
            else:
                pred = self.model(X)
        
        return pred.cpu().numpy()
    
    def save_checkpoint(self, name: str = 'checkpoint') -> None:
        if self.checkpoint_dir is None:
            return
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = self.checkpoint_dir / f'{name}.pt'
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, name: str = 'checkpoint') -> None:
        if self.checkpoint_dir is None:
            return
        
        path = self.checkpoint_dir / f'{name}.pt'
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

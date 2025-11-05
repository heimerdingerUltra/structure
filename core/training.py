import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Protocol, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
from pathlib import Path
import time


class TrainingPhase(Enum):
    WARMUP = auto()
    TRAINING = auto()
    ANNEALING = auto()
    FINISHED = auto()


@dataclass
class TrainingMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float
    grad_norm: float
    elapsed_time: float
    
    def to_dict(self) -> dict:
        return {
            'epoch': self.epoch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'lr': self.learning_rate,
            'grad_norm': self.grad_norm,
            'time': self.elapsed_time
        }


@dataclass
class TrainingState:
    phase: TrainingPhase = TrainingPhase.WARMUP
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float('inf')
    patience_counter: int = 0
    metrics_history: list[TrainingMetrics] = field(default_factory=list)
    
    def update_best(self, val_loss: float) -> bool:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return True
        self.patience_counter += 1
        return False


class Callback(Protocol):
    def on_train_begin(self, state: TrainingState) -> None: ...
    def on_train_end(self, state: TrainingState) -> None: ...
    def on_epoch_begin(self, epoch: int, state: TrainingState) -> None: ...
    def on_epoch_end(self, epoch: int, metrics: TrainingMetrics, state: TrainingState) -> None: ...
    def on_batch_begin(self, batch_idx: int, state: TrainingState) -> None: ...
    def on_batch_end(self, batch_idx: int, loss: float, state: TrainingState) -> None: ...


class EarlyStoppingCallback:
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.should_stop = False
    
    def on_train_begin(self, state: TrainingState) -> None:
        self.should_stop = False
    
    def on_train_end(self, state: TrainingState) -> None:
        pass
    
    def on_epoch_begin(self, epoch: int, state: TrainingState) -> None:
        pass
    
    def on_epoch_end(self, epoch: int, metrics: TrainingMetrics, state: TrainingState) -> None:
        improved = state.update_best(metrics.val_loss)
        
        if not improved and state.patience_counter >= self.patience:
            self.should_stop = True
    
    def on_batch_begin(self, batch_idx: int, state: TrainingState) -> None:
        pass
    
    def on_batch_end(self, batch_idx: int, loss: float, state: TrainingState) -> None:
        pass


class CheckpointCallback:
    
    def __init__(self, checkpoint_dir: Path, save_frequency: int = 10):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = save_frequency
    
    def on_train_begin(self, state: TrainingState) -> None:
        pass
    
    def on_train_end(self, state: TrainingState) -> None:
        pass
    
    def on_epoch_begin(self, epoch: int, state: TrainingState) -> None:
        pass
    
    def on_epoch_end(self, epoch: int, metrics: TrainingMetrics, state: TrainingState) -> None:
        if (epoch + 1) % self.save_frequency == 0:
            self._save(epoch, state)
    
    def on_batch_begin(self, batch_idx: int, state: TrainingState) -> None:
        pass
    
    def on_batch_end(self, batch_idx: int, loss: float, state: TrainingState) -> None:
        pass
    
    def _save(self, epoch: int, state: TrainingState) -> None:
        pass


class Trainer:
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        mixed_precision: bool = True,
        gradient_clip: float = 1.0,
        accumulation_steps: int = 1
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.mixed_precision = mixed_precision
        self.gradient_clip = gradient_clip
        self.accumulation_steps = accumulation_steps
        
        self.scaler = GradScaler() if mixed_precision else None
        self.state = TrainingState()
        self.callbacks: list[Callback] = []
    
    def register_callback(self, callback: Callback) -> None:
        self.callbacks.append(callback)
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()
        
        for batch_idx, (x, y) in enumerate(dataloader):
            self._on_batch_begin(batch_idx)
            
            x, y = x.to(self.device), y.to(self.device)
            
            if self.mixed_precision:
                with autocast():
                    pred = self.model(x)
                    loss = self.criterion(pred, y) / self.accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self._optimizer_step()
            else:
                pred = self.model(x)
                loss = self.criterion(pred, y) / self.accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self._optimizer_step()
            
            total_loss += loss.item() * self.accumulation_steps
            self.state.global_step += 1
            
            self._on_batch_end(batch_idx, loss.item())
        
        return total_loss / len(dataloader)
    
    def _optimizer_step(self) -> None:
        if self.mixed_precision:
            self.scaler.unscale_(self.optimizer)
            
            if self.gradient_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.gradient_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            self.optimizer.step()
        
        self.optimizer.zero_grad()
    
    @torch.inference_mode()
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            
            if self.mixed_precision:
                with autocast():
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
            else:
                pred = self.model(x)
                loss = self.criterion(pred, y)
            
            total_loss += loss.item()
            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
        
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        return {
            'loss': total_loss / len(dataloader),
            'rmse': rmse,
            'mae': mae
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        verbose: int = 1
    ) -> TrainingState:
        
        self._on_train_begin()
        
        for epoch in range(n_epochs):
            self._on_epoch_begin(epoch)
            
            start_time = time.perf_counter()
            
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
            
            elapsed = time.perf_counter() - start_time
            lr = self.optimizer.param_groups[0]['lr']
            
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_metrics['loss'],
                learning_rate=lr,
                grad_norm=0.0,
                elapsed_time=elapsed
            )
            
            self.state.metrics_history.append(metrics)
            self.state.epoch = epoch
            
            if verbose > 0 and (epoch + 1) % verbose == 0:
                print(
                    f"Epoch {epoch+1}/{n_epochs} - "
                    f"{elapsed:.1f}s - "
                    f"lr: {lr:.2e} - "
                    f"train: {train_loss:.4f} - "
                    f"val: {val_metrics['loss']:.4f} - "
                    f"rmse: {val_metrics['rmse']:.4f}"
                )
            
            self._on_epoch_end(epoch, metrics)
            
            early_stopping = next(
                (cb for cb in self.callbacks if isinstance(cb, EarlyStoppingCallback)),
                None
            )
            
            if early_stopping and early_stopping.should_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        self._on_train_end()
        
        return self.state
    
    def _on_train_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_train_begin(self.state)
    
    def _on_train_end(self) -> None:
        for callback in self.callbacks:
            callback.on_train_end(self.state)
    
    def _on_epoch_begin(self, epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, self.state)
    
    def _on_epoch_end(self, epoch: int, metrics: TrainingMetrics) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, metrics, self.state)
    
    def _on_batch_begin(self, batch_idx: int) -> None:
        for callback in self.callbacks:
            callback.on_batch_begin(batch_idx, self.state)
    
    def _on_batch_end(self, batch_idx: int, loss: float) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(batch_idx, loss, self.state)

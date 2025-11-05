import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Optional, Callable
import numpy as np
from pathlib import Path
import time
from dataclasses import dataclass, field


@dataclass
class TrainingMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_rmse: float
    val_mae: float
    val_r2: float
    learning_rate: float
    elapsed_time: float


class WarmupCosineScheduler:
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        cycles: int = 1,
        min_lr_ratio: float = 0.01
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cycles = cycles
        self.min_lr_ratio = min_lr_ratio
        
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            lr_scale = self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lr_scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (
                1 + np.cos(np.pi * progress * self.cycles)
            )
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_scale
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


class TrainingEngine:
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        use_amp: bool = True,
        gradient_clip: float = 1.0,
        accumulation_steps: int = 1,
        ema_decay: Optional[float] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp
        self.gradient_clip = gradient_clip
        self.accumulation_steps = accumulation_steps
        
        self.scaler = GradScaler(enabled=use_amp)
        
        self.ema = None
        if ema_decay is not None:
            from models.transformer import ExponentialMovingAverage
            self.ema = ExponentialMovingAverage(model, decay=ema_decay)
        
        self.scheduler = None
        self.history = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
    
    def train_epoch(self, dataloader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            with autocast(enabled=self.use_amp):
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss = loss / self.accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                
                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                if self.ema is not None:
                    self.ema.update()
            
            total_loss += loss.item() * self.accumulation_steps
            n_batches += 1
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, float]:
        if self.ema is not None:
            self.ema.apply_shadow()
        
        self.model.eval()
        
        all_preds = []
        all_targets = []
        total_loss = 0.0
        
        for x, y in dataloader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            with autocast(enabled=self.use_amp):
                pred = self.model(x)
                loss = self.criterion(pred, y)
            
            total_loss += loss.item()
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
        
        if self.ema is not None:
            self.ema.restore()
        
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        
        mse = np.mean((preds - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(preds - targets))
        
        ss_res = np.sum((targets - preds) ** 2)
        ss_tot = np.sum((targets - targets.mean()) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        return {
            'loss': total_loss / len(dataloader),
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def fit(
        self,
        train_loader,
        val_loader,
        n_epochs: int,
        early_stopping_patience: int = 25,
        early_stopping_delta: float = 1e-5,
        checkpoint_dir: Optional[Path] = None,
        verbose: int = 1
    ):
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(n_epochs):
            epoch_start = time.time()
            
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            elapsed = time.time() - epoch_start
            
            lr = self.optimizer.param_groups[0]['lr']
            
            metrics = TrainingMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_metrics['loss'],
                val_rmse=val_metrics['rmse'],
                val_mae=val_metrics['mae'],
                val_r2=val_metrics['r2'],
                learning_rate=lr,
                elapsed_time=elapsed
            )
            
            self.history.append(metrics)
            
            if (epoch + 1) % verbose == 0:
                print(
                    f"Epoch {epoch+1:03d}/{n_epochs} | "
                    f"Train: {train_loss:.4f} | "
                    f"Val: {val_metrics['loss']:.4f} | "
                    f"RMSE: {val_metrics['rmse']:.4f} | "
                    f"MAE: {val_metrics['mae']:.4f} | "
                    f"R²: {val_metrics['r2']:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"{elapsed:.1f}s"
                )
            
            if val_metrics['loss'] < self.best_val_loss - early_stopping_delta:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                
                if checkpoint_dir is not None:
                    self.save_checkpoint(checkpoint_dir / 'best.pt')
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            if checkpoint_dir is not None and (epoch + 1) % 10 == 0:
                self.save_checkpoint(checkpoint_dir / f'epoch_{epoch+1}.pt')
        
        if checkpoint_dir is not None:
            self.load_checkpoint(checkpoint_dir / 'best.pt')
        
        return self.history
    
    def save_checkpoint(self, path: Path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        if self.ema is not None:
            checkpoint['ema_shadow'] = self.ema.shadow
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.__dict__
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        if self.ema is not None and 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']
    
    @torch.no_grad()
    def predict(self, dataloader) -> np.ndarray:
        if self.ema is not None:
            self.ema.apply_shadow()
        
        self.model.eval()
        
        predictions = []
        
        for x, _ in dataloader:
            x = x.to(self.device, non_blocking=True)
            
            with autocast(enabled=self.use_amp):
                pred = self.model(x)
            
            predictions.append(pred.cpu().numpy())
        
        if self.ema is not None:
            self.ema.restore()
        
        return np.concatenate(predictions)

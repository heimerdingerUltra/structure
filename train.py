import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

from core.config import Configuration, get_optimal_config
from core.features import FeatureEngineering
from core.data import DataModule, Augmentation
from core.engine import TrainingEngine, WarmupCosineScheduler
from models.transformer import create_model, VisionTransformer
from models.mamba import Mamba


def setup_environment(config: Configuration):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    if config.runtime.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif config.runtime.benchmark:
        torch.backends.cudnn.benchmark = True
    
    if config.runtime.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        config.runtime.device = 'cpu'
    
    device = torch.device(config.runtime.device.value)
    
    return device


def load_data(path: str, sheet: str = 'All Options') -> pd.DataFrame:
    if path.endswith('.xlsx'):
        df = pd.read_excel(path, sheet_name=sheet)
    elif path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='VolatilityForge Training')
    
    parser.add_argument('--data', type=str, required=True, help='Path to data file')
    parser.add_argument('--sheet', type=str, default='All Options', help='Sheet name for Excel files')
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML file')
    parser.add_argument('--strategy', type=str, default='balanced', choices=['speed', 'balanced', 'quality'])
    parser.add_argument('--gpu-memory', type=int, default=16, help='GPU memory in GB')
    parser.add_argument('--architecture', type=str, default='transformer', choices=['transformer', 'mamba'])
    parser.add_argument('--output-dir', type=str, default='outputs')
    
    args = parser.parse_args()
    
    if args.config is not None:
        config = Configuration.from_yaml(args.config)
    else:
        config = get_optimal_config(
            gpu_memory_gb=args.gpu_memory,
            strategy=args.strategy
        )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config.checkpoint_dir = output_dir / 'checkpoints'
    config.log_dir = output_dir / 'logs'
    config.output_dir = output_dir
    
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    
    device = setup_environment(config)
    
    print("="*80)
    print("VolatilityForge - Neural Implied Volatility Prediction")
    print("="*80)
    print(f"Strategy: {args.strategy}")
    print(f"Architecture: {args.architecture}")
    print(f"Device: {device}")
    print(f"Mixed Precision: {config.runtime.precision.value}")
    print(f"Torch Compile: {config.runtime.compile}")
    print("="*80)
    
    print("\nLoading data...")
    df = load_data(args.data, args.sheet)
    print(f"Loaded {len(df):,} samples")
    
    print("\nExtracting features...")
    feature_engine = FeatureEngineering(cache_dir=config.data.cache_dir)
    X, y = feature_engine.fit_transform(
        df,
        use_cache=config.data.cache_enabled
    )
    print(f"Features: {X.shape[1]}")
    print(f"Valid samples: {len(X):,}")
    
    print("\nPreparing data loaders...")
    data_module = DataModule(
        batch_size=config.training.batch_size,
        num_workers=config.runtime.num_workers,
        pin_memory=config.runtime.pin_memory,
        persistent_workers=config.runtime.persistent_workers,
        prefetch_factor=config.runtime.prefetch_factor
    )
    
    splits = data_module.split_data(
        X, y,
        train_ratio=config.data.train_split,
        val_ratio=config.data.val_split,
        test_ratio=config.data.test_split,
        seed=config.seed
    )
    
    train_data, val_data, test_data = splits
    
    train_loader, val_loader, test_loader = data_module.create_loaders(
        train_data,
        val_data,
        test_data,
        augment=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    print("\nBuilding model...")
    config.model.n_features = X.shape[1]
    
    if args.architecture == 'transformer':
        model = create_model(config)
    elif args.architecture == 'mamba':
        model = Mamba(
            n_features=X.shape[1],
            d_model=config.model.d_model,
            n_layers=config.model.n_layers,
            dropout=config.model.dropout
        )
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    
    print("\nInitializing optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
        eps=config.training.epsilon,
        weight_decay=config.training.weight_decay
    )
    
    total_steps = len(train_loader) * config.training.epochs // config.training.accumulation_steps
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=config.training.warmup_steps,
        total_steps=total_steps,
        cycles=config.training.scheduler_cycles,
        min_lr_ratio=config.training.min_lr_ratio
    )
    
    criterion = nn.HuberLoss(delta=1.0)
    
    print("\nCreating training engine...")
    engine = TrainingEngine(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        use_amp=(config.runtime.precision != 'fp32'),
        gradient_clip=config.training.gradient_clip,
        accumulation_steps=config.training.accumulation_steps,
        ema_decay=config.training.ema_decay
    )
    
    engine.set_scheduler(scheduler)
    
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    history = engine.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=config.training.epochs,
        early_stopping_patience=config.training.early_stopping_patience,
        early_stopping_delta=config.training.early_stopping_delta,
        checkpoint_dir=config.checkpoint_dir,
        verbose=1
    )
    
    print("\n" + "="*80)
    print("Evaluating on test set...")
    print("="*80 + "\n")
    
    test_metrics = engine.evaluate(test_loader)
    
    print("Test Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    print(f"  R²: {test_metrics['r2']:.4f}")
    
    results = {
        'test_loss': test_metrics['loss'],
        'test_rmse': test_metrics['rmse'],
        'test_mae': test_metrics['mae'],
        'test_r2': test_metrics['r2'],
        'n_parameters': n_params,
        'n_epochs': len(history),
        'best_val_loss': engine.best_val_loss
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(config.output_dir / 'results.csv', index=False)
    
    config.to_yaml(config.output_dir / 'config.yaml')
    
    print(f"\nResults saved to: {config.output_dir}")
    print("\nTraining complete!")


if __name__ == '__main__':
    main()

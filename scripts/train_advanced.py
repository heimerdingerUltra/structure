import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path
import json
from datetime import datetime
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    Environment,
    get_environment,
    ModelType,
    create_model_config,
    create_training_config
)
from src.data.advanced_features import AdvancedFeatures
from src.data.pipeline import DataPipeline, DataValidator
from src.models.ensemble import ModelFactory
from src.models.advanced_ensemble import AdvancedEnsembleSystem
from src.models.registry import ModelRegistry, ModelMetadata, create_version_string
from src.training.advanced_trainer import AdvancedTrainer, EarlyStopping, ModelCheckpoint, LearningRateLogger
from src.training.optimization import configure_training
from src.evaluation.metrics import MetricsCalculator, MetricsSummary
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--sheet', type=str, default='All Options')
    parser.add_argument('--output', type=str, default='outputs')
    
    parser.add_argument('--models', nargs='+', default=['tabpfn', 'mamba', 'xlstm'])
    parser.add_argument('--ensemble-type', type=str, default='attention',
                       choices=['attention', 'hierarchical', 'uncertainty'])
    
    parser.add_argument('--strategy', type=str, default='balanced',
                       choices=['fast', 'accurate', 'balanced'])
    
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-kfold', action='store_true')
    parser.add_argument('--n-folds', type=int, default=5)
    
    parser.add_argument('--env', type=str, default='production',
                       choices=['development', 'staging', 'production'])
    
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_environment(env_name: str):
    import os
    os.environ['VOLATILITY_ENV'] = env_name
    env = get_environment()
    
    if env.benchmark:
        torch.backends.cudnn.benchmark = True
    if env.deterministic:
        torch.backends.cudnn.deterministic = True
    
    return env


def train_single_model(
    model_name: str,
    n_features: int,
    train_loader,
    val_loader,
    test_loader,
    config,
    env,
    device: str
):
    print(f"\n{'='*80}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*80}\n")
    
    model_type = ModelType(model_name)
    model_config = create_model_config(model_type, n_features)
    
    if model_name == 'tabpfn':
        model = ModelFactory.create_tabpfn(n_features, model_config.hyperparameters)
    elif model_name == 'mamba':
        model = ModelFactory.create_mamba(n_features, model_config.hyperparameters)
    elif model_name == 'xlstm':
        model = ModelFactory.create_xlstm(n_features, model_config.hyperparameters)
    elif model_name == 'hypermixer':
        model = ModelFactory.create_hypermixer(n_features, model_config.hyperparameters)
    elif model_name == 'ttt':
        model = ModelFactory.create_ttt(n_features, model_config.hyperparameters)
    elif model_name == 'modern_tcn':
        model = ModelFactory.create_modern_tcn(n_features, model_config.hyperparameters)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    optimizer, scheduler = configure_training(
        model,
        optimizer_name='adamw',
        scheduler_name='cosine_warmup',
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        epochs=config.epochs
    )
    
    criterion = nn.HuberLoss(delta=1.0)
    
    trainer = AdvancedTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        gradient_clip=config.gradient_clip,
        mixed_precision=config.mixed_precision,
        accumulation_steps=config.accumulation_steps,
        checkpoint_dir=env.checkpoint_dir / model_name
    )
    
    trainer.set_scheduler(scheduler)
    
    trainer.add_callback(EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta
    ))
    
    trainer.add_callback(ModelCheckpoint(
        checkpoint_dir=env.checkpoint_dir / model_name,
        save_frequency=config.checkpoint_frequency
    ))
    
    trainer.add_callback(LearningRateLogger())
    
    trainer.fit(
        train_loader,
        val_loader,
        config.epochs,
        verbose=10
    )
    
    test_X = []
    test_y = []
    for X_batch, y_batch in test_loader:
        test_X.append(X_batch)
        test_y.append(y_batch)
    
    test_X = torch.cat(test_X, dim=0)
    test_y = torch.cat(test_y, dim=0).numpy()
    
    pred = trainer.predict(test_X)
    
    metrics = MetricsCalculator.compute_regression_metrics(test_y, pred)
    
    print(f"\nTest Metrics:")
    print(f"  RMSE: {metrics.rmse:.4f}")
    print(f"  MAE: {metrics.mae:.4f}")
    print(f"  R²: {metrics.r2:.4f}")
    print(f"  MAPE: {metrics.mape:.2f}%")
    
    trainer.load_checkpoint('best')
    
    return trainer.model, metrics


def main():
    args = parse_args()
    
    env = setup_environment(args.env)
    set_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Environment: {env.name.value}")
    print(f"Device: {device}\n")
    
    print("Loading data...")
    df = pd.read_excel(args.data, sheet_name=args.sheet)
    print(f"Loaded {len(df)} rows\n")
    
    print("Feature engineering...")
    fe = AdvancedFeatures(scaler_type='robust', cache_dir='.cache')
    X, y = fe.fit_transform(df, use_cache=True)
    print(f"Features: {X.shape[1]}, Samples: {len(X)}\n")
    
    DataValidator.validate_all(X, y)
    
    config = create_training_config(
        strategy=args.strategy,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr
    )
    
    print(f"Training Configuration:")
    print(f"  Strategy: {args.strategy}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Mixed Precision: {config.mixed_precision}\n")
    
    pipeline = DataPipeline(
        batch_size=config.batch_size,
        num_workers=env.num_workers,
        pin_memory=env.pin_memory,
        persistent_workers=env.persistent_workers
    )
    
    train_loader, val_loader, test_loader = pipeline.create_loaders(
        X, y,
        test_size=0.2,
        val_size=0.1,
        seed=args.seed
    )
    
    n_features = X.shape[1]
    trained_models = {}
    model_metrics = {}
    
    for model_name in args.models:
        model, metrics = train_single_model(
            model_name,
            n_features,
            train_loader,
            val_loader,
            test_loader,
            config,
            env,
            device
        )
        
        trained_models[model_name] = model
        model_metrics[model_name] = metrics
    
    if len(trained_models) > 1:
        print(f"\n{'='*80}")
        print(f"Training {args.ensemble_type.upper()} Ensemble")
        print(f"{'='*80}\n")
        
        ensemble = AdvancedEnsembleSystem(
            trained_models,
            device=device,
            ensemble_type=args.ensemble_type
        )
        
        optimizer = torch.optim.Adam(ensemble.combiner.parameters(), lr=1e-3)
        criterion = nn.HuberLoss(delta=1.0)
        
        ensemble.train_combiner(
            train_loader,
            val_loader,
            optimizer,
            criterion,
            n_epochs=100,
            patience=15,
            verbose=True
        )
        
        test_X = torch.cat([X_batch for X_batch, _ in test_loader], dim=0)
        test_y = torch.cat([y_batch for _, y_batch in test_loader], dim=0).numpy()
        
        ensemble_pred = ensemble.predict(test_X)
        
        metrics = MetricsCalculator.compute_regression_metrics(test_y, ensemble_pred)
        
        print(f"\nEnsemble Test Metrics:")
        print(f"  RMSE: {metrics.rmse:.4f}")
        print(f"  MAE: {metrics.mae:.4f}")
        print(f"  R²: {metrics.r2:.4f}")
        print(f"  MAPE: {metrics.mape:.2f}%")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    registry = ModelRegistry(registry_dir=str(output_dir / "registry"))
    
    version = create_version_string()
    
    for name, model in trained_models.items():
        metrics_dict = model_metrics[name].to_dict()
        
        metadata = ModelMetadata(
            model_name=name,
            model_type=name,
            version=version,
            timestamp=datetime.now().isoformat(),
            n_features=n_features,
            hyperparameters={},
            metrics=metrics_dict,
            tags={'strategy': args.strategy, 'ensemble': args.ensemble_type}
        )
        
        registry.register_model(
            model,
            metadata,
            scaler=fe.scaler,
            feature_names=fe.feature_names
        )
    
    summary = MetricsSummary.create_summary(test_y, ensemble_pred if len(trained_models) > 1 else pred)
    summary.to_csv(output_dir / f'metrics_{version}.csv', index=False)
    
    print(f"\nSaved to {output_dir}")
    print("\nTraining complete!")


if __name__ == '__main__':
    main()

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from core.abstractions import Registry
from core.functional import pipe, compose
from core.architectures import (
    VolatilityTransformer,
    ModernMLP,
    DenseNet,
    TransformerConfig
)
from core.training import Trainer, EarlyStoppingCallback, CheckpointCallback
from core.feature_engineering import (
    MarketMicrostructure,
    OptionCharacteristics,
    NonlinearFeatures,
    StatisticalFeatures,
    InteractionFeatures
)


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    n_features: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    n_epochs: int
    patience: int
    device: str
    mixed_precision: bool
    gradient_clip: float
    
    @classmethod
    def default(cls, n_features: int) -> 'ExperimentConfig':
        return cls(
            model_name='transformer',
            n_features=n_features,
            batch_size=512,
            learning_rate=1e-3,
            weight_decay=1e-5,
            n_epochs=200,
            patience=20,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            mixed_precision=True,
            gradient_clip=1.0
        )


class ModelRegistry:
    
    _registry = Registry()
    
    @classmethod
    def register(cls, name: str):
        return cls._registry.register(name)
    
    @classmethod
    def create(cls, name: str, *args, **kwargs):
        return cls._registry.create(name, *args, **kwargs)
    
    @classmethod
    def list_models(cls) -> list[str]:
        return cls._registry.list_available()


@ModelRegistry.register('transformer')
def create_transformer(n_features: int, **kwargs) -> torch.nn.Module:
    config = TransformerConfig(
        dim=kwargs.get('dim', 512),
        depth=kwargs.get('depth', 12),
        n_heads=kwargs.get('n_heads', 8),
        n_kv_heads=kwargs.get('n_kv_heads', 2),
        ffn_dim_multiplier=kwargs.get('ffn_dim_multiplier', 2.67),
        norm_eps=kwargs.get('norm_eps', 1e-6),
        max_seq_len=kwargs.get('max_seq_len', 2048),
        dropout=kwargs.get('dropout', 0.1)
    )
    return VolatilityTransformer(n_features, config)


@ModelRegistry.register('mlp')
def create_mlp(n_features: int, **kwargs) -> torch.nn.Module:
    hidden_dims = kwargs.get('hidden_dims', [512, 384, 256])
    return ModernMLP(n_features, hidden_dims, dropout=kwargs.get('dropout', 0.1))


@ModelRegistry.register('densenet')
def create_densenet(n_features: int, **kwargs) -> torch.nn.Module:
    return DenseNet(
        n_features,
        init_dim=kwargs.get('init_dim', 256),
        growth_rate=kwargs.get('growth_rate', 64),
        block_config=kwargs.get('block_config', (4, 4, 4)),
        dropout=kwargs.get('dropout', 0.1)
    )


class FeaturePipeline:
    
    def __init__(self):
        self.transforms = []
        self.fitted = False
    
    def add_transform(self, transform) -> 'FeaturePipeline':
        self.transforms.append(transform)
        return self
    
    def fit(self, df: pd.DataFrame) -> 'FeaturePipeline':
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
        
        micro = MarketMicrostructure(
            bid=df['BID'].values,
            ask=df['ASK'].values,
            bid_size=df['BIDSIZE'].values,
            ask_size=df['ASKSIZE'].values
        )
        
        option = OptionCharacteristics(
            spot=micro.mid_price,
            strike=df['STRIKE_PRC'].values,
            time_to_expiry=df['DAYS_TO_EXPIRY_CALC'].values / 365.25,
            is_call=(df['OPTION_TYPE'] == 'CALL').astype(float).values
        )
        
        features = []
        
        features.extend([
            micro.mid_price,
            micro.spread,
            micro.relative_spread,
            micro.order_flow_imbalance,
            micro.liquidity_score,
            option.moneyness,
            option.log_moneyness,
            option.standardized_moneyness,
            option.intrinsic_value
        ])
        
        moneyness = option.log_moneyness
        tte = option.time_to_expiry
        
        poly_features = NonlinearFeatures.polynomial_expansion(moneyness, degree=3)
        features.append(poly_features)
        
        interactions = np.column_stack([
            InteractionFeatures.multiplicative(moneyness, tte),
            InteractionFeatures.multiplicative(moneyness, np.sqrt(tte)),
            InteractionFeatures.ratio(micro.spread, tte)
        ])
        features.append(interactions)
        
        X = np.column_stack([f.reshape(-1, 1) if f.ndim == 1 else f for f in features])
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


class ExperimentRunner:
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> tuple:
        
        pipeline = FeaturePipeline()
        X = pipeline.fit_transform(df)
        y = df['IMP_VOLT'].values
        
        valid_mask = (y > 0) & (y < 500) & ~np.isnan(y)
        X, y = X[valid_mask], y[valid_mask]
        
        n = len(X)
        indices = np.random.permutation(n)
        
        test_split = int(n * test_size)
        val_split = int(n * (test_size + val_size))
        
        test_idx = indices[:test_split]
        val_idx = indices[test_split:val_split]
        train_idx = indices[val_split:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler
    
    def create_loaders(
        self,
        train_data: tuple,
        val_data: tuple,
        test_data: tuple
    ) -> tuple:
        
        from torch.utils.data import TensorDataset, DataLoader
        
        train_dataset = TensorDataset(
            torch.FloatTensor(train_data[0]),
            torch.FloatTensor(train_data[1])
        )
        
        val_dataset = TensorDataset(
            torch.FloatTensor(val_data[0]),
            torch.FloatTensor(val_data[1])
        )
        
        test_dataset = TensorDataset(
            torch.FloatTensor(test_data[0]),
            torch.FloatTensor(test_data[1])
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def run(
        self,
        train_loader,
        val_loader,
        test_loader,
        checkpoint_dir: Optional[Path] = None
    ) -> dict:
        
        model = ModelRegistry.create(self.config.model_name, self.config.n_features)
        model = model.to(self.device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        criterion = torch.nn.HuberLoss(delta=1.0)
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            mixed_precision=self.config.mixed_precision,
            gradient_clip=self.config.gradient_clip
        )
        
        trainer.register_callback(EarlyStoppingCallback(patience=self.config.patience))
        
        if checkpoint_dir is not None:
            trainer.register_callback(CheckpointCallback(checkpoint_dir))
        
        state = trainer.fit(
            train_loader,
            val_loader,
            self.config.n_epochs,
            scheduler=scheduler,
            verbose=10
        )
        
        test_metrics = trainer.validate(test_loader)
        
        return {
            'training_state': state,
            'test_metrics': test_metrics,
            'model': model
        }


def main(
    data_path: str,
    model_name: str = 'transformer',
    output_dir: str = 'outputs'
) -> None:
    
    df = pd.read_excel(data_path, sheet_name='All Options')
    
    print(f"Loaded {len(df):,} samples")
    
    pipeline = FeaturePipeline()
    X = pipeline.fit_transform(df)
    
    print(f"Generated {X.shape[1]} features")
    
    config = ExperimentConfig.default(n_features=X.shape[1])
    config = dataclasses.replace(config, model_name=model_name)
    
    runner = ExperimentRunner(config)
    
    train_data, val_data, test_data, scaler = runner.prepare_data(df)
    train_loader, val_loader, test_loader = runner.create_loaders(
        train_data, val_data, test_data
    )
    
    checkpoint_dir = Path(output_dir) / 'checkpoints'
    results = runner.run(train_loader, val_loader, test_loader, checkpoint_dir)
    
    print(f"\nTest RMSE: {results['test_metrics']['rmse']:.4f}")
    print(f"Test MAE: {results['test_metrics']['mae']:.4f}")
    
    torch.save(
        results['model'].state_dict(),
        Path(output_dir) / f'{model_name}_final.pt'
    )


if __name__ == '__main__':
    import argparse
    import dataclasses
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model', type=str, default='transformer')
    parser.add_argument('--output', type=str, default='outputs')
    
    args = parser.parse_args()
    
    main(args.data, args.model, args.output)

# VolatilityForge - Production Edition

> Enterprise-grade deep learning framework for options implied volatility prediction

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

**VolatilityForge** implements state-of-the-art neural architectures (2024-2025) with production-grade engineering patterns for predicting options implied volatility.

### Key Innovations

- **6 Cutting-Edge Architectures**: TabPFN, Mamba, xLSTM, HyperMixer, TTT, ModernTCN
- **Advanced Ensemble Methods**: Attention-based, hierarchical, and uncertainty-aware stacking
- **Production-Ready Infrastructure**: Model registry, caching, monitoring, inference optimization
- **Sophisticated Feature Engineering**: 40+ quantitative features with zero model bias
- **Enterprise Patterns**: Builder pattern, factory pattern, strategy pattern, observer pattern

## Architecture

```
volatilityforge/
├── config/                  # Configuration management
│   ├── environment.py      # Environment-specific settings
│   ├── model_config.py     # Model hyperparameters
│   └── training_config.py  # Training strategies
├── src/
│   ├── data/
│   │   ├── advanced_features.py  # Feature engineering
│   │   └── pipeline.py           # Data pipeline
│   ├── models/
│   │   ├── architectures/        # Neural architectures
│   │   ├── advanced_ensemble.py  # Ensemble systems
│   │   └── registry.py           # Model versioning
│   ├── training/
│   │   ├── advanced_trainer.py   # Training loop
│   │   └── optimization.py       # Optimizers & schedulers
│   ├── evaluation/
│   │   └── metrics.py            # Comprehensive metrics
│   └── inference/
│       └── pipeline.py           # Production inference
└── scripts/
    └── train_advanced.py         # Training orchestration
```

## Installation

```bash
git clone https://github.com/yourusername/volatilityforge.git
cd volatilityforge
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
python scripts/train_advanced.py \
    --data data/options.xlsx \
    --sheet "All Options" \
    --models tabpfn mamba xlstm \
    --strategy balanced \
    --ensemble-type attention \
    --env production
```

### Training Strategies

```bash
# Fast training (100 epochs)
--strategy fast

# Balanced (200 epochs)  
--strategy balanced

# High accuracy (400 epochs)
--strategy accurate
```

### Ensemble Types

```bash
# Attention-based ensemble
--ensemble-type attention

# Hierarchical ensemble
--ensemble-type hierarchical

# Uncertainty-aware ensemble
--ensemble-type uncertainty
```

## Configuration

### Environment Profiles

```python
# Development
VOLATILITY_ENV=development python scripts/train_advanced.py ...

# Staging
VOLATILITY_ENV=staging python scripts/train_advanced.py ...

# Production
VOLATILITY_ENV=production python scripts/train_advanced.py ...
```

### Model Configuration

```python
from config import create_model_config, ModelType

config = create_model_config(
    ModelType.TABPFN,
    n_features=40,
    d_model=512,
    n_layers=12,
    dropout=0.1
)
```

### Training Configuration

```python
from config import create_training_config

config = create_training_config(
    strategy="balanced",
    batch_size=512,
    epochs=200,
    learning_rate=1e-3
)
```

## Advanced Features

### Feature Engineering

40+ quantitative features across 5 categories:

**Microstructure**
- Bid-ask spreads (absolute, relative, basis points)
- Order imbalance & flow intensity
- Weighted mid prices

**Moneyness**
- Log-moneyness & polynomials
- ATM distance & categorization
- ITM/OTM/ATM flags

**Time Decay**
- Time to expiry transformations
- Short/medium/long term buckets
- Decay factors

**Volume & Liquidity**
- Log-transformed volume/OI
- Turnover ratios
- Liquidity scores

**Advanced Interactions**
- Cross-feature products
- Effective spreads
- Delta proxies

### Model Registry

```python
from src.models.registry import ModelRegistry

registry = ModelRegistry("model_registry")

# Register model
registry.register_model(
    model=trained_model,
    metadata=metadata,
    scaler=scaler,
    feature_names=features
)

# Load model
model, metadata, scaler, features = registry.load_model(
    model_name="tabpfn",
    version="20250101_120000"
)
```

### Callbacks System

```python
from src.training.advanced_trainer import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateLogger
)

trainer.add_callback(EarlyStopping(patience=20))
trainer.add_callback(ModelCheckpoint(checkpoint_dir="checkpoints"))
trainer.add_callback(LearningRateLogger())
```

### Inference Pipeline

```python
from src.inference.pipeline import InferencePipeline, ModelServer

# Single prediction
pipeline = InferencePipeline(model, scaler, feature_extractor)
result = pipeline.predict(df)

# Model server
server = ModelServer("model_registry")
server.load_model("tabpfn", version="latest")
result = server.predict("tabpfn", data)

# Batch processing
BatchInference.process_file(
    input_path="data.xlsx",
    output_path="predictions.xlsx",
    pipeline=pipeline
)
```

## Optimizations

### Mixed Precision Training

Automatic mixed precision (AMP) for 2x speedup:

```python
trainer = AdvancedTrainer(
    model=model,
    mixed_precision=True  # FP16/FP32 mixed precision
)
```

### Gradient Accumulation

Simulate larger batch sizes:

```python
trainer = AdvancedTrainer(
    model=model,
    accumulation_steps=4  # Effective batch = 512 * 4
)
```

### Data Pipeline Optimization

```python
pipeline = DataPipeline(
    batch_size=512,
    num_workers=4,           # Parallel data loading
    pin_memory=True,         # Faster GPU transfer
    persistent_workers=True, # Reuse workers
    prefetch_factor=2        # Prefetch batches
)
```

### Feature Caching

```python
fe = AdvancedFeatures(cache_dir=".cache")
X, y = fe.fit_transform(df, use_cache=True)  # Cache computed features
```

## Metrics & Evaluation

### Comprehensive Metrics

```python
from src.evaluation.metrics import MetricsCalculator

metrics = MetricsCalculator.compute_regression_metrics(y_true, y_pred)
# Returns: RMSE, MAE, MAPE, R², MSE, max_error, median_AE
```

### Financial Metrics

```python
from src.evaluation.metrics import FinancialMetrics

iv_metrics = FinancialMetrics.compute_iv_metrics(y_true, y_pred)
pricing_impact = FinancialMetrics.compute_pricing_impact(
    y_true_iv, y_pred_iv, moneyness, time_to_expiry
)
```

### Ensemble Metrics

```python
from src.evaluation.metrics import EnsembleMetrics

diversity = EnsembleMetrics.compute_diversity(predictions_list)
disagreement = EnsembleMetrics.compute_disagreement(predictions_list)
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| RMSE | < 2.0 | Volatility points |
| MAE | < 1.5 | Mean absolute error |
| R² | > 0.90 | Coefficient of determination |
| MAPE | < 5% | Percentage error |
| Inference | < 10ms | Single sample (GPU) |
| Throughput | > 10K/s | Batch inference |

## Advanced Usage

### K-Fold Cross-Validation

```bash
python scripts/train_advanced.py \
    --data data/options.xlsx \
    --use-kfold \
    --n-folds 5
```

### Custom Optimizer

```python
from src.training.optimization import configure_training

optimizer, scheduler = configure_training(
    model,
    optimizer_name='lamb',  # LAMB optimizer
    scheduler_name='onecycle',
    learning_rate=1e-3
)
```

### Uncertainty Estimation

```python
ensemble = AdvancedEnsembleSystem(
    models,
    ensemble_type='uncertainty'
)

pred, uncertainty = ensemble.predict(X, return_uncertainty=True)
```

## Design Patterns

### Builder Pattern
Configuration objects with fluent interface

### Factory Pattern
Model and optimizer creation

### Strategy Pattern
Training strategies (fast/balanced/accurate)

### Observer Pattern
Training callbacks

### Registry Pattern
Model versioning and management

### Pipeline Pattern
Data preprocessing and inference

## Best Practices

### Training
- Use mixed precision on modern GPUs (20-series+)
- Enable gradient accumulation for limited VRAM
- Start with `balanced` strategy, then optimize
- Monitor learning rate with callbacks

### Production
- Use model registry for versioning
- Enable caching for repeated predictions
- Batch predictions for throughput
- Monitor uncertainty for risk management

### Development
- Use `development` environment for debugging
- Set deterministic=True for reproducibility
- Use `fast` strategy for experimentation
- Enable verbose logging

## Requirements

- Python 3.8+
- PyTorch 2.1+
- CUDA 11.8+ (optional, for GPU)
- 16GB+ VRAM (recommended)
- 32GB+ RAM

## Citation

```bibtex
@software{volatilityforge2025,
  title={VolatilityForge: Production-Grade Neural IV Prediction},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/volatilityforge}
}
```

## License

MIT License - see [LICENSE](LICENSE) file

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Follow PEP 8 style guide
5. Submit a pull request

## Acknowledgments

- Research papers for architecture inspiration
- PyTorch team for excellent framework
- Financial ML community for domain expertise

---

**VolatilityForge** - Enterprise-grade volatility prediction powered by state-of-the-art deep learning

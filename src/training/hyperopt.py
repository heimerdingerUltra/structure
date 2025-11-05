import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import numpy as np
from typing import Dict, Callable, Optional


class HyperparameterOptimizer:
    
    def __init__(
        self,
        objective_fn: Callable,
        direction: str = 'minimize',
        n_trials: int = 100,
        timeout: Optional[int] = None,
        n_jobs: int = 1
    ):
        self.objective_fn = objective_fn
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        
        self.study = None
        self.best_params = None
        self.best_value = None
        
    def suggest_hyperparameters(self, trial: optuna.Trial, model_type: str) -> Dict:
        if model_type == 'tabpfn':
            return {
                'd_model': trial.suggest_categorical('d_model', [256, 384, 512, 768]),
                'n_layers': trial.suggest_int('n_layers', 6, 16),
                'n_heads': trial.suggest_categorical('n_heads', [4, 8, 12, 16]),
                'mlp_ratio': trial.suggest_float('mlp_ratio', 2.0, 6.0),
                'dropout': trial.suggest_float('dropout', 0.0, 0.3)
            }
        
        elif model_type == 'mamba':
            return {
                'd_model': trial.suggest_categorical('d_model', [128, 192, 256, 384]),
                'n_layers': trial.suggest_int('n_layers', 4, 12),
                'd_state': trial.suggest_categorical('d_state', [8, 16, 32]),
                'expand': trial.suggest_int('expand', 2, 4),
                'dropout': trial.suggest_float('dropout', 0.0, 0.3)
            }
        
        elif model_type == 'xlstm':
            return {
                'hidden_size': trial.suggest_categorical('hidden_size', [128, 192, 256, 384]),
                'n_layers': trial.suggest_int('n_layers', 2, 6),
                'use_mlstm': trial.suggest_categorical('use_mlstm', [True, False]),
                'dropout': trial.suggest_float('dropout', 0.0, 0.3)
            }
        
        elif model_type == 'ft_transformer':
            return {
                'd_token': trial.suggest_categorical('d_token', [96, 128, 192, 256]),
                'n_blocks': trial.suggest_int('n_blocks', 2, 6),
                'n_heads': trial.suggest_categorical('n_heads', [4, 8, 12]),
                'd_ffn_factor': trial.suggest_float('d_ffn_factor', 1.0, 2.0),
                'attention_dropout': trial.suggest_float('attention_dropout', 0.0, 0.3),
                'ffn_dropout': trial.suggest_float('ffn_dropout', 0.0, 0.3)
            }
        
        elif model_type == 'modern_resnet':
            return {
                'dim': trial.suggest_categorical('dim', [256, 384, 512, 768]),
                'n_blocks': trial.suggest_int('n_blocks', 6, 16),
                'hidden_dim_ratio': trial.suggest_float('hidden_dim_ratio', 2.0, 6.0),
                'dropout': trial.suggest_float('dropout', 0.0, 0.3),
                'drop_path_rate': trial.suggest_float('drop_path_rate', 0.0, 0.2),
                'use_se': trial.suggest_categorical('use_se', [True, False]),
                'use_layer_scale': trial.suggest_categorical('use_layer_scale', [True, False])
            }
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def optimize(self, model_type: str, storage: Optional[str] = None):
        def objective(trial: optuna.Trial) -> float:
            hyperparams = self.suggest_hyperparameters(trial, model_type)
            
            training_params = {
                'batch_size': trial.suggest_categorical('batch_size', [256, 512, 1024]),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
            }
            
            score = self.objective_fn(trial, hyperparams, training_params)
            
            return score
        
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            study_name=f'{model_type}_optimization'
        )
        
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs
        )
        
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        return self.best_params, self.best_value
    
    def get_best_params(self) -> Dict:
        return self.best_params
    
    def get_optimization_history(self):
        return self.study.trials_dataframe()
    
    def get_param_importances(self):
        return optuna.importance.get_param_importances(self.study)
    
    def visualize_optimization_history(self):
        try:
            import plotly
            fig = optuna.visualization.plot_optimization_history(self.study)
            return fig
        except ImportError:
            print("Plotly not installed. Install with: pip install plotly")
            return None
    
    def visualize_param_importances(self):
        try:
            import plotly
            fig = optuna.visualization.plot_param_importances(self.study)
            return fig
        except ImportError:
            print("Plotly not installed. Install with: pip install plotly")
            return None


class AutoML:
    
    def __init__(
        self,
        model_types: list,
        n_trials_per_model: int = 50,
        direction: str = 'minimize'
    ):
        self.model_types = model_types
        self.n_trials_per_model = n_trials_per_model
        self.direction = direction
        
        self.results = {}
        self.best_model = None
        self.best_config = None
        self.best_score = float('inf') if direction == 'minimize' else float('-inf')
        
    def run(self, objective_fn: Callable, storage: Optional[str] = None):
        for model_type in self.model_types:
            print(f"\nOptimizing {model_type}...")
            
            optimizer = HyperparameterOptimizer(
                objective_fn=objective_fn,
                direction=self.direction,
                n_trials=self.n_trials_per_model,
                n_jobs=1
            )
            
            best_params, best_value = optimizer.optimize(model_type, storage)
            
            self.results[model_type] = {
                'best_params': best_params,
                'best_value': best_value,
                'optimizer': optimizer
            }
            
            is_better = (
                (self.direction == 'minimize' and best_value < self.best_score) or
                (self.direction == 'maximize' and best_value > self.best_score)
            )
            
            if is_better:
                self.best_model = model_type
                self.best_config = best_params
                self.best_score = best_value
        
        return self.best_model, self.best_config, self.best_score
    
    def get_results(self) -> Dict:
        return self.results
    
    def get_leaderboard(self):
        leaderboard = []
        
        for model_type, result in self.results.items():
            leaderboard.append({
                'model': model_type,
                'score': result['best_value'],
                'params': result['best_params']
            })
        
        leaderboard.sort(
            key=lambda x: x['score'],
            reverse=(self.direction == 'maximize')
        )
        
        return leaderboard

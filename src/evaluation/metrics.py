import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import pandas as pd


@dataclass
class RegressionMetrics:
    rmse: float
    mae: float
    mape: float
    r2: float
    mse: float
    max_error: float
    median_ae: float
    
    def to_dict(self) -> Dict:
        return {
            'rmse': self.rmse,
            'mae': self.mae,
            'mape': self.mape,
            'r2': self.r2,
            'mse': self.mse,
            'max_error': self.max_error,
            'median_ae': self.median_ae
        }


class MetricsCalculator:
    
    @staticmethod
    def compute_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> RegressionMetrics:
        
        errors = y_true - y_pred
        squared_errors = errors ** 2
        absolute_errors = np.abs(errors)
        
        mse = np.mean(squared_errors)
        rmse = np.sqrt(mse)
        mae = np.mean(absolute_errors)
        
        mape = np.mean(np.abs(errors / (y_true + 1e-10))) * 100
        
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        ss_res = np.sum(squared_errors)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        max_error = np.max(absolute_errors)
        median_ae = np.median(absolute_errors)
        
        return RegressionMetrics(
            rmse=rmse,
            mae=mae,
            mape=mape,
            r2=r2,
            mse=mse,
            max_error=max_error,
            median_ae=median_ae
        )
    
    @staticmethod
    def compute_quantile_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
    ) -> Dict:
        
        errors = np.abs(y_true - y_pred)
        
        quantile_metrics = {}
        for q in quantiles:
            quantile_metrics[f'error_q{int(q*100)}'] = np.quantile(errors, q)
        
        return quantile_metrics
    
    @staticmethod
    def compute_directional_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 1.0
    ) -> Dict:
        
        errors = y_true - y_pred
        
        accurate = np.abs(errors) <= threshold
        accuracy = np.mean(accurate)
        
        overestimate = np.mean(errors < -threshold)
        underestimate = np.mean(errors > threshold)
        
        return {
            'accuracy': accuracy,
            'overestimate_rate': overestimate,
            'underestimate_rate': underestimate
        }
    
    @staticmethod
    def compute_segment_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        segments: np.ndarray
    ) -> Dict:
        
        segment_metrics = {}
        
        for segment in np.unique(segments):
            mask = segments == segment
            
            if np.sum(mask) > 0:
                metrics = MetricsCalculator.compute_regression_metrics(
                    y_true[mask],
                    y_pred[mask]
                )
                
                segment_metrics[f'segment_{segment}'] = metrics.to_dict()
        
        return segment_metrics


class FinancialMetrics:
    
    @staticmethod
    def compute_iv_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        
        iv_diff = y_pred - y_true
        
        iv_bias = np.mean(iv_diff)
        iv_std = np.std(iv_diff)
        
        relative_error = np.abs(iv_diff / (y_true + 1e-10))
        mean_relative_error = np.mean(relative_error)
        
        large_errors = np.mean(np.abs(iv_diff) > 5.0)
        
        return {
            'iv_bias': iv_bias,
            'iv_std': iv_std,
            'mean_relative_error': mean_relative_error,
            'large_error_rate': large_errors
        }
    
    @staticmethod
    def compute_pricing_impact(
        y_true_iv: np.ndarray,
        y_pred_iv: np.ndarray,
        moneyness: np.ndarray,
        time_to_expiry: np.ndarray
    ) -> Dict:
        
        vega_proxy = np.sqrt(time_to_expiry) * np.exp(-0.5 * (np.log(moneyness) ** 2))
        
        pricing_error = np.abs(y_pred_iv - y_true_iv) * vega_proxy
        mean_pricing_error = np.mean(pricing_error)
        
        return {
            'mean_pricing_error': mean_pricing_error,
            'max_pricing_error': np.max(pricing_error),
            'pricing_error_90pct': np.percentile(pricing_error, 90)
        }


class EnsembleMetrics:
    
    @staticmethod
    def compute_diversity(predictions_list: List[np.ndarray]) -> Dict:
        
        predictions = np.stack(predictions_list, axis=0)
        
        mean_pred = np.mean(predictions, axis=0)
        
        individual_errors = np.mean(np.abs(predictions - mean_pred), axis=1)
        diversity = np.mean(individual_errors)
        
        correlations = []
        n_models = len(predictions_list)
        for i in range(n_models):
            for j in range(i + 1, n_models):
                corr = np.corrcoef(predictions_list[i], predictions_list[j])[0, 1]
                correlations.append(corr)
        
        mean_correlation = np.mean(correlations)
        
        return {
            'diversity': diversity,
            'mean_correlation': mean_correlation,
            'prediction_std': np.mean(np.std(predictions, axis=0))
        }
    
    @staticmethod
    def compute_disagreement(
        predictions_list: List[np.ndarray],
        threshold: float = 2.0
    ) -> Dict:
        
        predictions = np.stack(predictions_list, axis=0)
        pred_std = np.std(predictions, axis=0)
        
        high_disagreement = np.mean(pred_std > threshold)
        
        return {
            'mean_disagreement': np.mean(pred_std),
            'max_disagreement': np.max(pred_std),
            'high_disagreement_rate': high_disagreement
        }


class MetricsSummary:
    
    @staticmethod
    def create_summary(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        include_financial: bool = True,
        moneyness: np.ndarray = None,
        time_to_expiry: np.ndarray = None
    ) -> pd.DataFrame:
        
        basic_metrics = MetricsCalculator.compute_regression_metrics(y_true, y_pred)
        quantile_metrics = MetricsCalculator.compute_quantile_metrics(y_true, y_pred)
        directional = MetricsCalculator.compute_directional_accuracy(y_true, y_pred)
        
        summary = {
            **basic_metrics.to_dict(),
            **quantile_metrics,
            **directional
        }
        
        if include_financial:
            financial = FinancialMetrics.compute_iv_metrics(y_true, y_pred)
            summary.update(financial)
            
            if moneyness is not None and time_to_expiry is not None:
                pricing = FinancialMetrics.compute_pricing_impact(
                    y_true, y_pred, moneyness, time_to_expiry
                )
                summary.update(pricing)
        
        return pd.DataFrame([summary])

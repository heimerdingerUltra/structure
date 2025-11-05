import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from scipy import stats
from scipy.special import ndtr
from sklearn.preprocessing import RobustScaler
import joblib
from pathlib import Path
import hashlib


@dataclass
class FeatureMetadata:
    names: List[str]
    statistics: Dict[str, Dict[str, float]]
    version: str
    n_samples: int


class FeatureEngineering:
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.scaler = RobustScaler(quantile_range=(1, 99))
        self.metadata: Optional[FeatureMetadata] = None
        self.cache_dir = cache_dir or Path(".cache/features")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _compute_hash(self, df: pd.DataFrame) -> str:
        data_hash = hashlib.blake2b(
            df.values.tobytes(),
            digest_size=16
        ).hexdigest()
        return data_hash
    
    def _try_load_cache(self, cache_key: str) -> Optional[Tuple]:
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            try:
                return joblib.load(cache_path)
            except:
                cache_path.unlink()
        return None
    
    def _save_cache(self, cache_key: str, data: Tuple):
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        joblib.dump(data, cache_path, compress=3)
    
    def _microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        
        if not {'BID', 'ASK'}.issubset(df.columns):
            return features
        
        bid, ask = df['BID'].values, df['ASK'].values
        mid = 0.5 * (bid + ask)
        spread = ask - bid
        
        features['mid'] = mid
        features['spread'] = spread
        features['spread_rel'] = spread / (mid + 1e-10)
        features['spread_bps'] = 1e4 * features['spread_rel']
        
        features['log_mid'] = np.log1p(mid)
        features['log_spread'] = np.log1p(spread)
        
        if {'BIDSIZE', 'ASKSIZE'}.issubset(df.columns):
            bid_size = df['BIDSIZE'].values
            ask_size = df['ASKSIZE'].values
            total_size = bid_size + ask_size + 1e-10
            
            features['imbalance'] = (bid_size - ask_size) / total_size
            features['imbalance_abs'] = np.abs(features['imbalance'])
            features['bid_ratio'] = bid_size / total_size
            
            features['weighted_mid'] = (
                bid * ask_size + ask * bid_size
            ) / total_size
            
            features['effective_spread'] = (
                features['weighted_mid'] - mid
            ) / mid
            
            features['microprice'] = (
                bid * ask_size + ask * bid_size
            ) / (bid_size + ask_size + 1e-10)
        
        return features
    
    def _moneyness_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        strike_col = next((c for c in ['STRIKE_PRC', 'STRIKE_PRICE', 'STRIKE'] if c in df.columns), None)
        
        if strike_col is None or 'mid' not in features.columns:
            return features
        
        spot = features['mid'].values
        strike = df[strike_col].values
        
        moneyness = spot / (strike + 1e-10)
        log_moneyness = np.log(moneyness + 1e-10)
        
        features['moneyness'] = moneyness
        features['log_moneyness'] = log_moneyness
        features['moneyness_sq'] = moneyness ** 2
        features['moneyness_cube'] = moneyness ** 3
        
        features['abs_log_moneyness'] = np.abs(log_moneyness)
        features['moneyness_deviation'] = moneyness - 1.0
        
        atm_threshold = 0.02
        features['is_atm'] = (np.abs(log_moneyness) < atm_threshold).astype(np.float32)
        features['is_itm'] = (log_moneyness > atm_threshold).astype(np.float32)
        features['is_otm'] = (log_moneyness < -atm_threshold).astype(np.float32)
        
        features['atm_distance'] = np.abs(log_moneyness)
        features['atm_proximity'] = np.exp(-features['atm_distance'] ** 2)
        
        moneyness_levels = [0.8, 0.9, 0.95, 0.99, 1.01, 1.05, 1.1, 1.2]
        for level in moneyness_levels:
            features[f'moneyness_above_{int(level*100)}'] = (
                moneyness > level
            ).astype(np.float32)
        
        return features
    
    def _temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        
        if 'DAYS_TO_EXPIRY_CALC' not in df.columns:
            return features
        
        days = df['DAYS_TO_EXPIRY_CALC'].values
        years = days / 365.25
        years_safe = np.maximum(years, 1/365.25)
        
        features['tte_years'] = years
        features['tte_days'] = days
        features['sqrt_tte'] = np.sqrt(years_safe)
        features['log_tte'] = np.log(years_safe)
        features['inv_tte'] = 1.0 / years_safe
        features['tte_sq'] = years ** 2
        
        features['decay_factor'] = np.exp(-2 * years)
        features['gamma_factor'] = years * np.exp(-years)
        
        tte_levels = [7, 14, 30, 60, 90, 180, 365]
        for level in tte_levels:
            features[f'tte_above_{level}d'] = (days > level).astype(np.float32)
        
        features['is_weekly'] = (days <= 7).astype(np.float32)
        features['is_monthly'] = ((days > 7) & (days <= 45)).astype(np.float32)
        features['is_quarterly'] = ((days > 45) & (days <= 120)).astype(np.float32)
        features['is_long_dated'] = (days > 120).astype(np.float32)
        
        return features
    
    def _liquidity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        
        if 'ACVOL_1' in df.columns:
            volume = df['ACVOL_1'].values
            features['volume'] = volume
            features['log_volume'] = np.log1p(volume)
            features['sqrt_volume'] = np.sqrt(volume)
            features['volume_sq'] = volume ** 2
        
        if 'OPINT_1' in df.columns:
            oi = df['OPINT_1'].values
            features['open_interest'] = oi
            features['log_oi'] = np.log1p(oi)
            features['sqrt_oi'] = np.sqrt(oi)
        
        if {'ACVOL_1', 'OPINT_1'}.issubset(df.columns):
            volume = df['ACVOL_1'].values
            oi = df['OPINT_1'].values
            
            features['turnover'] = volume / (oi + 1.0)
            features['log_turnover'] = np.log1p(features['turnover'])
            
            features['liquidity_score'] = np.sqrt(volume * oi)
            features['log_liquidity'] = np.log1p(features['liquidity_score'])
            
            features['activity_ratio'] = (
                volume / (volume + oi + 1e-10)
            )
        
        return features
    
    def _interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if 'moneyness' in features.columns and 'tte_years' in features.columns:
            m = features['moneyness'].values
            t = features['tte_years'].values
            sqrt_t = features['sqrt_tte'].values
            
            features['m_t'] = m * t
            features['m_sqrt_t'] = m * sqrt_t
            features['m_inv_t'] = m * features['inv_tte']
            
            features['log_m_log_t'] = (
                features['log_moneyness'] * features['log_tte']
            )
            
            features['m_sq_t'] = features['moneyness_sq'] * t
            features['m_sq_sqrt_t'] = features['moneyness_sq'] * sqrt_t
            
            features['atm_distance_t'] = features['atm_distance'] * t
            features['atm_distance_sqrt_t'] = features['atm_distance'] * sqrt_t
        
        if 'spread_rel' in features.columns and 'sqrt_tte' in features.columns:
            features['spread_sqrt_t'] = (
                features['spread_rel'] * features['sqrt_tte']
            )
            features['spread_inv_t'] = (
                features['spread_rel'] * features['inv_tte']
            )
        
        if 'log_volume' in features.columns and 'moneyness' in features.columns:
            features['volume_moneyness'] = (
                features['log_volume'] * features['moneyness']
            )
            features['volume_atm_distance'] = (
                features['log_volume'] * features['atm_distance']
            )
        
        if 'imbalance' in features.columns and 'spread_rel' in features.columns:
            features['imbalance_spread'] = (
                features['imbalance'] * features['spread_rel']
            )
        
        if {'log_liquidity', 'atm_distance', 'sqrt_tte'}.issubset(features.columns):
            features['liquidity_atm_time'] = (
                features['log_liquidity'] * 
                features['atm_proximity'] * 
                features['sqrt_tte']
            )
        
        return features
    
    def _statistical_features(self, features: pd.DataFrame) -> pd.DataFrame:
        numerical_cols = features.select_dtypes(include=[np.number]).columns[:10]
        
        for col in numerical_cols:
            values = features[col].values
            if len(values) > 10:
                features[f'{col}_rank'] = stats.rankdata(values) / len(values)
        
        return features
    
    def _option_greeks_proxy(self, features: pd.DataFrame) -> pd.DataFrame:
        if not {'moneyness', 'sqrt_tte', 'log_moneyness'}.issubset(features.columns):
            return features
        
        log_m = features['log_moneyness'].values
        sqrt_t = features['sqrt_tte'].values
        t = features['tte_years'].values
        
        features['delta_proxy'] = ndtr(log_m / (sqrt_t + 1e-10))
        
        features['gamma_proxy'] = (
            np.exp(-0.5 * (log_m / (sqrt_t + 1e-10)) ** 2) / 
            (sqrt_t * np.sqrt(2 * np.pi) + 1e-10)
        )
        
        features['theta_proxy'] = -features['gamma_proxy'] / (2 * t + 1e-10)
        
        features['vega_proxy'] = np.exp(-0.5 * log_m ** 2) * sqrt_t
        
        return features
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str = 'IMP_VOLT',
        use_cache: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        cache_key = self._compute_hash(df) if use_cache else None
        
        if cache_key and use_cache:
            cached = self._try_load_cache(cache_key)
            if cached is not None:
                X, y, self.scaler, self.metadata = cached
                return X, y
        
        features = pd.DataFrame(index=df.index)
        
        micro = self._microstructure_features(df)
        features = pd.concat([features, micro], axis=1)
        
        features = self._moneyness_features(df, features)
        
        temporal = self._temporal_features(df)
        features = pd.concat([features, temporal], axis=1)
        
        liquidity = self._liquidity_features(df)
        features = pd.concat([features, liquidity], axis=1)
        
        features = self._interaction_features(features)
        
        features = self._option_greeks_proxy(features)
        
        if 'OPTION_TYPE' in df.columns:
            features['is_call'] = (
                df['OPTION_TYPE'].str.upper() == 'CALL'
            ).astype(np.float32)
        elif 'PUT_CALL' in df.columns:
            features['is_call'] = (
                df['PUT_CALL'].astype(str).str.upper().str.contains('CALL', na=False)
            ).astype(np.float32)
        
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(features.median())
        features = features.fillna(0)
        
        feature_names = list(features.columns)
        
        X = self.scaler.fit_transform(features.values.astype(np.float32))
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        y = df[target_col].values.astype(np.float32)
        
        valid_mask = (
            (y > 0) & 
            (y < 500) & 
            ~np.isnan(y) & 
            ~np.isinf(y)
        )
        
        X_valid = X[valid_mask]
        y_valid = y[valid_mask]
        
        feature_stats = {}
        for i, name in enumerate(feature_names):
            feature_stats[name] = {
                'mean': float(np.mean(X_valid[:, i])),
                'std': float(np.std(X_valid[:, i])),
                'min': float(np.min(X_valid[:, i])),
                'max': float(np.max(X_valid[:, i]))
            }
        
        self.metadata = FeatureMetadata(
            names=feature_names,
            statistics=feature_stats,
            version="1.0.0",
            n_samples=len(X_valid)
        )
        
        if cache_key and use_cache:
            self._save_cache(cache_key, (X_valid, y_valid, self.scaler, self.metadata))
        
        return X_valid, y_valid
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.metadata is None:
            raise RuntimeError("Must call fit_transform before transform")
        
        features = pd.DataFrame(index=df.index)
        
        micro = self._microstructure_features(df)
        features = pd.concat([features, micro], axis=1)
        
        features = self._moneyness_features(df, features)
        
        temporal = self._temporal_features(df)
        features = pd.concat([features, temporal], axis=1)
        
        liquidity = self._liquidity_features(df)
        features = pd.concat([features, liquidity], axis=1)
        
        features = self._interaction_features(features)
        features = self._option_greeks_proxy(features)
        
        if 'OPTION_TYPE' in df.columns:
            features['is_call'] = (
                df['OPTION_TYPE'].str.upper() == 'CALL'
            ).astype(np.float32)
        elif 'PUT_CALL' in df.columns:
            features['is_call'] = (
                df['PUT_CALL'].astype(str).str.upper().str.contains('CALL', na=False)
            ).astype(np.float32)
        
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(features.median())
        features = features.fillna(0)
        
        for name in self.metadata.names:
            if name not in features.columns:
                features[name] = 0.0
        
        features = features[self.metadata.names]
        
        return self.scaler.transform(features.values.astype(np.float32))

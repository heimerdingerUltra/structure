import numpy as np
import pandas as pd
from typing import Protocol, runtime_checkable
from dataclasses import dataclass
from scipy import stats
from scipy.special import erf


@runtime_checkable
class FeatureTransform(Protocol):
    def __call__(self, x: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class MarketMicrostructure:
    bid: np.ndarray
    ask: np.ndarray
    bid_size: np.ndarray
    ask_size: np.ndarray
    
    @property
    def mid_price(self) -> np.ndarray:
        return 0.5 * (self.bid + self.ask)
    
    @property
    def spread(self) -> np.ndarray:
        return self.ask - self.bid
    
    @property
    def relative_spread(self) -> np.ndarray:
        return self.spread / (self.mid_price + 1e-10)
    
    @property
    def effective_spread(self) -> np.ndarray:
        return 2.0 * np.abs(self.mid_price - self.volume_weighted_mid)
    
    @property
    def volume_weighted_mid(self) -> np.ndarray:
        total_size = self.bid_size + self.ask_size + 1e-10
        return (self.bid * self.ask_size + self.ask * self.bid_size) / total_size
    
    @property
    def order_flow_imbalance(self) -> np.ndarray:
        total = self.bid_size + self.ask_size + 1e-10
        return (self.bid_size - self.ask_size) / total
    
    @property
    def liquidity_score(self) -> np.ndarray:
        return np.log1p(self.bid_size * self.ask_size) / (self.relative_spread + 1e-10)


@dataclass(frozen=True)
class OptionCharacteristics:
    spot: np.ndarray
    strike: np.ndarray
    time_to_expiry: np.ndarray
    is_call: np.ndarray
    
    @property
    def moneyness(self) -> np.ndarray:
        return self.spot / (self.strike + 1e-10)
    
    @property
    def log_moneyness(self) -> np.ndarray:
        return np.log(self.moneyness + 1e-10)
    
    @property
    def standardized_moneyness(self) -> np.ndarray:
        return self.log_moneyness / np.sqrt(self.time_to_expiry + 1e-10)
    
    @property
    def intrinsic_value(self) -> np.ndarray:
        return np.where(
            self.is_call > 0.5,
            np.maximum(self.spot - self.strike, 0),
            np.maximum(self.strike - self.spot, 0)
        )
    
    @property
    def time_value_proxy(self) -> np.ndarray:
        return np.sqrt(self.time_to_expiry) * np.abs(self.log_moneyness)
    
    def vega_proxy(self, vol: float = 0.2) -> np.ndarray:
        d1 = (self.log_moneyness + 0.5 * vol**2 * self.time_to_expiry) / (vol * np.sqrt(self.time_to_expiry) + 1e-10)
        return self.spot * np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi) * np.sqrt(self.time_to_expiry)
    
    def gamma_proxy(self, vol: float = 0.2) -> np.ndarray:
        d1 = (self.log_moneyness + 0.5 * vol**2 * self.time_to_expiry) / (vol * np.sqrt(self.time_to_expiry) + 1e-10)
        return np.exp(-0.5 * d1**2) / (self.spot * vol * np.sqrt(2.0 * np.pi * self.time_to_expiry) + 1e-10)
    
    def delta_proxy(self, vol: float = 0.2) -> np.ndarray:
        d1 = (self.log_moneyness + 0.5 * vol**2 * self.time_to_expiry) / (vol * np.sqrt(self.time_to_expiry) + 1e-10)
        return stats.norm.cdf(d1) if self.is_call.mean() > 0.5 else stats.norm.cdf(d1) - 1.0


class NonlinearFeatures:
    
    @staticmethod
    def polynomial_expansion(x: np.ndarray, degree: int = 3) -> np.ndarray:
        return np.column_stack([x**i for i in range(1, degree + 1)])
    
    @staticmethod
    def fourier_features(x: np.ndarray, n_frequencies: int = 10) -> np.ndarray:
        frequencies = 2.0 * np.pi * np.arange(1, n_frequencies + 1)
        cos_features = np.cos(x[:, None] * frequencies)
        sin_features = np.sin(x[:, None] * frequencies)
        return np.column_stack([cos_features, sin_features])
    
    @staticmethod
    def radial_basis_features(x: np.ndarray, centers: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        distances = np.sum((x[:, None] - centers[None, :]) ** 2, axis=-1)
        return np.exp(-gamma * distances)
    
    @staticmethod
    def spline_features(x: np.ndarray, knots: np.ndarray, degree: int = 3) -> np.ndarray:
        basis = []
        for k in knots:
            basis.append(np.maximum(x - k, 0) ** degree)
        return np.column_stack(basis)


class StatisticalFeatures:
    
    @staticmethod
    def rolling_statistics(x: np.ndarray, window: int = 20) -> dict[str, np.ndarray]:
        return {
            'mean': pd.Series(x).rolling(window).mean().fillna(x.mean()).values,
            'std': pd.Series(x).rolling(window).std().fillna(x.std()).values,
            'skew': pd.Series(x).rolling(window).skew().fillna(0).values,
            'kurt': pd.Series(x).rolling(window).kurt().fillna(0).values
        }
    
    @staticmethod
    def ewma(x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
        return pd.Series(x).ewm(alpha=alpha).mean().values
    
    @staticmethod
    def z_score(x: np.ndarray) -> np.ndarray:
        return (x - np.mean(x)) / (np.std(x) + 1e-10)
    
    @staticmethod
    def rank_transform(x: np.ndarray) -> np.ndarray:
        return stats.rankdata(x) / len(x)


class InteractionFeatures:
    
    @staticmethod
    def multiplicative(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return x1 * x2
    
    @staticmethod
    def ratio(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return x1 / (x2 + 1e-10)
    
    @staticmethod
    def difference(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return x1 - x2
    
    @staticmethod
    def geometric_mean(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return np.sqrt(np.abs(x1 * x2)) * np.sign(x1 * x2)
    
    @staticmethod
    def harmonic_mean(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return 2.0 / (1.0 / (x1 + 1e-10) + 1.0 / (x2 + 1e-10))


class VolatilityFeatures:
    
    @staticmethod
    def parkinson_estimator(high: np.ndarray, low: np.ndarray) -> np.ndarray:
        return np.sqrt(np.log(high / low) ** 2 / (4.0 * np.log(2)))
    
    @staticmethod
    def garman_klass_estimator(
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> np.ndarray:
        hl = np.log(high / low) ** 2
        oc = np.log(close / open_) ** 2
        return np.sqrt(0.5 * hl - (2 * np.log(2) - 1) * oc)
    
    @staticmethod
    def realized_volatility(returns: np.ndarray, window: int = 20) -> np.ndarray:
        return pd.Series(returns).rolling(window).std().fillna(returns.std()).values * np.sqrt(252)


class ImpliedVolatilityFeatures:
    
    @staticmethod
    def vol_smile_slope(moneyness: np.ndarray, iv: np.ndarray) -> np.ndarray:
        coeffs = np.polyfit(moneyness, iv, deg=2)
        return coeffs[1] + 2 * coeffs[0] * moneyness
    
    @staticmethod
    def vol_smile_curvature(moneyness: np.ndarray, iv: np.ndarray) -> np.ndarray:
        coeffs = np.polyfit(moneyness, iv, deg=2)
        return 2 * coeffs[0] * np.ones_like(moneyness)
    
    @staticmethod
    def atm_vol_distance(moneyness: np.ndarray, iv: np.ndarray) -> np.ndarray:
        atm_mask = np.abs(moneyness - 1.0) < 0.05
        atm_vol = np.mean(iv[atm_mask]) if atm_mask.any() else np.mean(iv)
        return iv - atm_vol


class QuantumFeatures:
    
    @staticmethod
    def entropy(x: np.ndarray, bins: int = 50) -> float:
        hist, _ = np.histogram(x, bins=bins, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log(hist))
    
    @staticmethod
    def mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 50) -> float:
        hist_xy, _, _ = np.histogram2d(x, y, bins=bins, density=True)
        hist_x, _ = np.histogram(x, bins=bins, density=True)
        hist_y, _ = np.histogram(y, bins=bins, density=True)
        
        mi = 0.0
        for i in range(len(hist_x)):
            for j in range(len(hist_y)):
                if hist_xy[i, j] > 0 and hist_x[i] > 0 and hist_y[j] > 0:
                    mi += hist_xy[i, j] * np.log(hist_xy[i, j] / (hist_x[i] * hist_y[j]))
        
        return mi

"""
NoiseInject Core Module
Implements noise injection strategies for ML robustness testing
"""

import numpy as np
from typing import Optional, Dict, Any


class NoiseInjector:
    """
    Inject noise into target values using various strategies.
    
    Strategies:
        - legacy: Simple Gaussian σ * N(0,1)
        - quantile: More noise at distribution extremes
        - threshold: More noise above/below absolute thresholds
        - outlier: More noise for z-score outliers
        - hetero: Heteroscedastic noise (variance ∝ |y|)
        - valprop: Value-proportional noise
    """
    
    def __init__(self, strategy: str = 'legacy', random_state: Optional[int] = None):
        """
        Initialize NoiseInjector.
        
        Args:
            strategy: One of ['legacy', 'quantile', 'threshold', 'outlier', 'hetero', 'valprop']
            random_state: Random seed for reproducibility
        """
        valid_strategies = ['legacy', 'quantile', 'threshold', 'outlier', 'hetero', 'valprop']
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
        
        self.strategy = strategy
        self.rng = np.random.RandomState(random_state)
    
    def inject(self, y: np.ndarray, sigma: float, **strategy_params) -> np.ndarray:
        """
        Inject noise into target values.
        
        Args:
            y: Target values (1D numpy array)
            sigma: Base noise level
            **strategy_params: Strategy-specific parameters
        
        Returns:
            Noisy target values
        """
        y = np.asarray(y).flatten()
        
        if self.strategy == 'legacy':
            return self._legacy(y, sigma)
        elif self.strategy == 'quantile':
            return self._quantile(y, sigma, **strategy_params)
        elif self.strategy == 'threshold':
            return self._threshold(y, sigma, **strategy_params)
        elif self.strategy == 'outlier':
            return self._outlier(y, sigma, **strategy_params)
        elif self.strategy == 'hetero':
            return self._hetero(y, sigma, **strategy_params)
        elif self.strategy == 'valprop':
            return self._valprop(y, sigma, **strategy_params)
    
    def _legacy(self, y: np.ndarray, sigma: float) -> np.ndarray:
        """Simple Gaussian: y_noisy = y + σ * N(0,1)"""
        noise = self.rng.normal(0, sigma, size=len(y))
        return y + noise
    
    def _quantile(self, y: np.ndarray, sigma: float,
                  high_quantile: float = 0.9,
                  low_quantile: float = 0.1,
                  high_sigma_mult: float = 2.0,
                  low_sigma_mult: float = 2.0,
                  mid_sigma_mult: float = 0.1) -> np.ndarray:
        """
        More noise at distribution extremes.
        
        Args:
            high_quantile: Upper quantile threshold (default 0.9)
            low_quantile: Lower quantile threshold (default 0.1)
            high_sigma_mult: Multiplier for high values (default 2.0)
            low_sigma_mult: Multiplier for low values (default 2.0)
            mid_sigma_mult: Multiplier for middle values (default 0.1)
        """
        high_thresh = np.quantile(y, high_quantile)
        low_thresh = np.quantile(y, low_quantile)
        
        sigma_values = np.full(len(y), sigma * mid_sigma_mult)
        sigma_values[y >= high_thresh] = sigma * high_sigma_mult
        sigma_values[y <= low_thresh] = sigma * low_sigma_mult
        
        noise = self.rng.normal(0, 1, size=len(y)) * sigma_values
        return y + noise
    
    def _threshold(self, y: np.ndarray, sigma: float,
                   high_threshold: float = 1.0,
                   low_threshold: float = -1.0,
                   high_sigma_mult: float = 2.0,
                   low_sigma_mult: float = 2.0,
                   mid_sigma_mult: float = 0.1) -> np.ndarray:
        """
        More noise above/below absolute thresholds.
        
        Args:
            high_threshold: Upper absolute threshold (default 1.0)
            low_threshold: Lower absolute threshold (default -1.0)
            high_sigma_mult: Multiplier above high threshold (default 2.0)
            low_sigma_mult: Multiplier below low threshold (default 2.0)
            mid_sigma_mult: Multiplier in middle range (default 0.1)
        """
        sigma_values = np.full(len(y), sigma * mid_sigma_mult)
        sigma_values[y >= high_threshold] = sigma * high_sigma_mult
        sigma_values[y <= low_threshold] = sigma * low_sigma_mult
        
        noise = self.rng.normal(0, 1, size=len(y)) * sigma_values
        return y + noise
    
    def _outlier(self, y: np.ndarray, sigma: float,
                 outlier_z_threshold: float = 2.0,
                 outlier_sigma_mult: float = 3.0,
                 normal_sigma_mult: float = 0.1) -> np.ndarray:
        """
        More noise for z-score outliers.
        
        Args:
            outlier_z_threshold: Z-score threshold for outliers (default 2.0)
            outlier_sigma_mult: Multiplier for outliers (default 3.0)
            normal_sigma_mult: Multiplier for normal points (default 0.1)
        """
        z_scores = np.abs((y - np.mean(y)) / np.std(y))
        
        sigma_values = np.full(len(y), sigma * normal_sigma_mult)
        sigma_values[z_scores > outlier_z_threshold] = sigma * outlier_sigma_mult
        
        noise = self.rng.normal(0, 1, size=len(y)) * sigma_values
        return y + noise
    
    def _hetero(self, y: np.ndarray, sigma: float,
                alpha_mult: float = 0.1,
                beta_mult: float = 0.05) -> np.ndarray:
        """
        Heteroscedastic noise: σ_i = sqrt(alpha*σ² + beta*σ²*|y_i|)
        
        Args:
            alpha_mult: Base variance multiplier (default 0.1)
            beta_mult: Value-dependent variance multiplier (default 0.05)
        """
        alpha = sigma * sigma * alpha_mult
        beta = sigma * sigma * beta_mult
        
        sigma_values = np.sqrt(alpha + beta * np.abs(y))
        noise = self.rng.normal(0, 1, size=len(y)) * sigma_values
        return y + noise
    
    def _valprop(self, y: np.ndarray, sigma: float,
                 proportionality_factor: float = 0.1) -> np.ndarray:
        """
        Value-proportional: σ_i = base_sigma + prop_factor * |y_i|
        
        Args:
            proportionality_factor: Proportionality to absolute value (default 0.1)
        """
        sigma_values = sigma + proportionality_factor * np.abs(y)
        noise = self.rng.normal(0, 1, size=len(y)) * sigma_values
        return y + noise
    
    def get_effective_noise(self, y_clean: np.ndarray, y_noisy: np.ndarray, 
                           method: str = 'std_normalized') -> float:
        """
        Calculate effective noise level.
        
        Args:
            y_clean: Original clean values
            y_noisy: Noisy values
            method: One of ['std_normalized', 'range_normalized', 'absolute']
        
        Returns:
            Effective noise level
        """
        y_clean = np.asarray(y_clean).flatten()
        y_noisy = np.asarray(y_noisy).flatten()
        
        absolute_diff = np.mean(np.abs(y_noisy - y_clean))
        
        if method == 'absolute':
            return absolute_diff
        elif method == 'std_normalized':
            std = np.std(y_clean)
            return absolute_diff / std if std > 1e-10 else 0.0
        elif method == 'range_normalized':
            range_val = np.ptp(y_clean)
            return absolute_diff / range_val if range_val > 1e-10 else 0.0
        else:
            raise ValueError(f"Invalid method: {method}")

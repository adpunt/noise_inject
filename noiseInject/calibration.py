"""
NoiseInject Calibration Module
Calibrate noise parameters to achieve target effective noise levels
Supports both regression (sigma) and classification (flip_probability)
"""

import numpy as np
from typing import Optional, Dict
from .core import NoiseInjectorRegression, NoiseInjectorClassification


# ============================================================================
# REGRESSION CALIBRATION
# ============================================================================

def calibrate_sigma(
    y_train: np.ndarray,
    target_effective_noise: float = 0.1,
    strategy: str = 'legacy',
    method: str = 'std_normalized',
    random_state: Optional[int] = None,
    max_iterations: int = 20,
    tolerance: float = 0.01,
    **strategy_params
) -> float:
    """
    Find sigma value that produces target effective noise level for regression.
    
    Uses binary search to find σ such that effective_noise(y, inject(y, σ)) ≈ target.
    
    Args:
        y_train: Training target values
        target_effective_noise: Desired effective noise level
        strategy: Noise strategy ('legacy', 'quantile', etc.)
        method: Effective noise calculation method ('std_normalized', etc.)
        random_state: Random seed for reproducibility
        max_iterations: Maximum binary search iterations
        tolerance: Convergence tolerance
        **strategy_params: Strategy-specific parameters
    
    Returns:
        Calibrated sigma value
    
    Example:
        >>> sigma_cal = calibrate_sigma(y_train, target_effective_noise=0.1)
        >>> # Now inject noise with calibrated sigma
        >>> injector = NoiseInjectorRegression('legacy')
        >>> y_noisy = injector.inject(y_train, sigma_cal)
    """
    y_train = np.asarray(y_train).flatten()
    injector = NoiseInjectorRegression(strategy=strategy, random_state=random_state)
    
    # Binary search bounds
    sigma_low = 0.0
    sigma_high = 2.0
    
    # First check if upper bound is high enough
    y_noisy_test = injector.inject(y_train, sigma_high, **strategy_params)
    effective_noise_test = injector.get_effective_noise(y_train, y_noisy_test, method=method)
    
    if effective_noise_test < target_effective_noise:
        sigma_high = 5.0
    
    # Binary search
    for iteration in range(max_iterations):
        sigma_mid = (sigma_low + sigma_high) / 2
        
        y_noisy = injector.inject(y_train, sigma_mid, **strategy_params)
        effective_noise = injector.get_effective_noise(y_train, y_noisy, method=method)
        
        error = abs(effective_noise - target_effective_noise)
        if error < tolerance:
            return sigma_mid
        
        if effective_noise < target_effective_noise:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid
    
    return (sigma_low + sigma_high) / 2


def calibrate_multiple_sigmas(
    y_train: np.ndarray,
    target_noise_levels: list,
    strategy: str = 'legacy',
    method: str = 'std_normalized',
    random_state: Optional[int] = None,
    **strategy_params
) -> Dict[float, float]:
    """
    Calibrate sigma values for multiple target noise levels (regression).
    
    Args:
        y_train: Training target values
        target_noise_levels: List of target effective noise levels
        strategy: Noise strategy
        method: Effective noise calculation method
        random_state: Random seed
        **strategy_params: Strategy-specific parameters
    
    Returns:
        Dictionary mapping target_noise -> calibrated_sigma
    
    Example:
        >>> sigma_dict = calibrate_multiple_sigmas(y_train, [0.05, 0.1, 0.2, 0.3])
        >>> # sigma_dict = {0.05: 0.12, 0.1: 0.24, 0.2: 0.48, 0.3: 0.72}
    """
    result = {}
    
    for target in target_noise_levels:
        sigma = calibrate_sigma(
            y_train=y_train,
            target_effective_noise=target,
            strategy=strategy,
            method=method,
            random_state=random_state,
            **strategy_params
        )
        result[target] = sigma
    
    return result


# ============================================================================
# CLASSIFICATION CALIBRATION
# ============================================================================

def calibrate_flip_probability(
    y_train: np.ndarray,
    target_flip_rate: float = 0.1,
    strategy: str = 'uniform',
    random_state: Optional[int] = None,
    max_iterations: int = 20,
    tolerance: float = 0.01,
    **strategy_params
) -> float:
    """
    Find flip_probability value that produces target effective flip rate for classification.
    
    Uses binary search to find p such that effective_flip_rate(y, inject(y, p)) ≈ target.
    
    Args:
        y_train: Training class labels
        target_flip_rate: Desired effective flip rate (0.0 to 1.0)
        strategy: Noise strategy ('uniform', 'class_imbalance', etc.)
        random_state: Random seed for reproducibility
        max_iterations: Maximum binary search iterations
        tolerance: Convergence tolerance
        **strategy_params: Strategy-specific parameters
    
    Returns:
        Calibrated flip_probability value
    
    Example:
        >>> flip_prob_cal = calibrate_flip_probability(y_train, target_flip_rate=0.1)
        >>> # Now inject noise with calibrated flip probability
        >>> injector = NoiseInjectorClassification('uniform')
        >>> y_noisy = injector.inject(y_train, flip_prob_cal)
    """
    y_train = np.asarray(y_train).flatten()
    injector = NoiseInjectorClassification(strategy=strategy, random_state=random_state)
    
    # Binary search bounds
    flip_prob_low = 0.0
    flip_prob_high = 1.0
    
    # Binary search
    for iteration in range(max_iterations):
        flip_prob_mid = (flip_prob_low + flip_prob_high) / 2
        
        y_noisy = injector.inject(y_train, flip_prob_mid, **strategy_params)
        effective_flip_rate = injector.get_effective_flip_rate(y_train, y_noisy)
        
        error = abs(effective_flip_rate - target_flip_rate)
        if error < tolerance:
            return flip_prob_mid
        
        if effective_flip_rate < target_flip_rate:
            flip_prob_low = flip_prob_mid
        else:
            flip_prob_high = flip_prob_mid
    
    return (flip_prob_low + flip_prob_high) / 2


def calibrate_multiple_flip_probabilities(
    y_train: np.ndarray,
    target_flip_rates: list,
    strategy: str = 'uniform',
    random_state: Optional[int] = None,
    **strategy_params
) -> Dict[float, float]:
    """
    Calibrate flip_probability values for multiple target flip rates (classification).
    
    Args:
        y_train: Training class labels
        target_flip_rates: List of target effective flip rates
        strategy: Noise strategy
        random_state: Random seed
        **strategy_params: Strategy-specific parameters
    
    Returns:
        Dictionary mapping target_flip_rate -> calibrated_flip_probability
    
    Example:
        >>> flip_prob_dict = calibrate_multiple_flip_probabilities(
        ...     y_train, [0.05, 0.1, 0.2, 0.3]
        ... )
        >>> # flip_prob_dict = {0.05: 0.06, 0.1: 0.12, 0.2: 0.24, 0.3: 0.36}
    """
    result = {}
    
    for target in target_flip_rates:
        flip_prob = calibrate_flip_probability(
            y_train=y_train,
            target_flip_rate=target,
            strategy=strategy,
            random_state=random_state,
            **strategy_params
        )
        result[target] = flip_prob
    
    return result
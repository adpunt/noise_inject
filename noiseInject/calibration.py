"""
NoiseInject Calibration Module
Calibrate sigma values to achieve target effective noise levels
"""

import numpy as np
from typing import Optional
from .core import NoiseInjector


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
    Find sigma value that produces target effective noise level.
    
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
        >>> injector = NoiseInjector('legacy')
        >>> y_noisy = injector.inject(y_train, sigma_cal)
    """
    y_train = np.asarray(y_train).flatten()
    
    injector = NoiseInjector(strategy=strategy, random_state=random_state)
    
    # Binary search bounds
    sigma_low = 0.0
    sigma_high = 2.0  # Start with reasonable upper bound
    
    # First check if upper bound is high enough
    y_noisy_test = injector.inject(y_train, sigma_high, **strategy_params)
    effective_noise_test = injector.get_effective_noise(y_train, y_noisy_test, method=method)
    
    if effective_noise_test < target_effective_noise:
        # Need higher upper bound
        sigma_high = 5.0
    
    # Binary search
    for iteration in range(max_iterations):
        sigma_mid = (sigma_low + sigma_high) / 2
        
        # Generate noisy data with current sigma
        y_noisy = injector.inject(y_train, sigma_mid, **strategy_params)
        
        # Calculate effective noise
        effective_noise = injector.get_effective_noise(y_train, y_noisy, method=method)
        
        # Check convergence
        error = abs(effective_noise - target_effective_noise)
        if error < tolerance:
            return sigma_mid
        
        # Update bounds
        if effective_noise < target_effective_noise:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid
    
    # Return best approximation if not converged
    return (sigma_low + sigma_high) / 2


def calibrate_multiple_sigmas(
    y_train: np.ndarray,
    target_noise_levels: list,
    strategy: str = 'legacy',
    method: str = 'std_normalized',
    random_state: Optional[int] = None,
    **strategy_params
) -> dict:
    """
    Calibrate sigma values for multiple target noise levels.
    
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

"""
NoiseInject Calibration Module
Calibrate noise parameters to achieve target effective noise levels
Supports both regression (sigma) and classification (flip_probability)
"""

import numpy as np
from typing import Optional, Dict
from .core import NoiseInjectorClassification


# ============================================================================
# REGRESSION: THERE IS NOTHING TO CALIBRATE
# ============================================================================
# `calibrate_sigma` and `calibrate_multiple_sigmas` were deleted in 1.0.0.
#
# They binary-searched for the sigma that made mean |dy| / SD hit a target --
# the FIRST moment. The design controls the second: RMSE and R-squared are
# second-moment quantities, so matching the second moment is what makes the
# conditions comparable. At identical root-mean-square noise, mean|e|/rms is
# 0.797 for a Gaussian but 0.642 for Student-t at nu = 3, so calibrating on the
# first moment hands the heavy-tailed conditions up to 24% more actual noise at
# the same nominal level -- reintroducing exactly the confound the redesign
# exists to remove. The search was also run against a moving target: one
# injector was re-used across all 20 iterations, so every iteration drew fresh
# noise.
#
# `NoiseInjectorRegression` now solves for its own scale in closed form --
# scale = target_dose / unit_dose -- which is exact, deterministic, and
# identical to the Rust implementation. Pass the target dose to inject()
# directly; there is no calibration step. See NOISE_DESIGN.md sections 1 and 6.


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
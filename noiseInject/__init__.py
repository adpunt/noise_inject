"""
NoiseInject - A framework for testing ML model robustness to label noise

Example workflow:
    >>> from noiseInject import NoiseInjector, calculate_noise_metrics, calibrate_sigma
    >>> 
    >>> # 1. Calibrate sigma for fair comparison
    >>> sigma = calibrate_sigma(y_train, target_effective_noise=0.1)
    >>> 
    >>> # 2. Inject noise
    >>> injector = NoiseInjector(strategy='legacy')
    >>> y_noisy = injector.inject(y_train, sigma)
    >>> 
    >>> # 3. Train your model
    >>> model.fit(X_train, y_noisy)
    >>> 
    >>> # 4. Analyze robustness
    >>> predictions = {
    ...     0.0: model_baseline.predict(X_test),
    ...     0.1: model_noisy.predict(X_test)
    ... }
    >>> metrics = calculate_noise_metrics(y_test, predictions)
"""

from .core import NoiseInjector
from .metrics import calculate_noise_metrics
from .calibration import calibrate_sigma, calibrate_multiple_sigmas

__version__ = '0.1.0'
__all__ = ['NoiseInjector', 'calculate_noise_metrics', 'calibrate_sigma', 'calibrate_multiple_sigmas']

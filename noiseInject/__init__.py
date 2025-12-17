"""
NoiseInject - A framework for testing ML model robustness to label noise
Supports both regression (continuous noise) and classification (label flips)
"""

from .core import (
    NoiseInjectorRegression,
    NoiseInjectorClassification,
)

from .calibration import (
    # Regression calibration
    calibrate_sigma,
    calibrate_multiple_sigmas,
    # Classification calibration
    calibrate_flip_probability,
    calibrate_multiple_flip_probabilities
)

from .metrics import (
    # Regression metrics
    calculate_noise_metrics,
    # Classification metrics
    calculate_classification_metrics,
    calculate_confusion_matrix_metrics,
    get_most_robust_classes,
    get_least_robust_classes
)

__version__ = '0.2.0'

__all__ = [
    # Core classes
    'NoiseInjectorRegression',
    'NoiseInjectorClassification',
    'NoiseInjector',
    # Regression calibration
    'calibrate_sigma',
    'calibrate_multiple_sigmas',
    # Classification calibration
    'calibrate_flip_probability',
    'calibrate_multiple_flip_probabilities',
    # Regression metrics
    'calculate_noise_metrics',
    # Classification metrics
    'calculate_classification_metrics',
    'calculate_confusion_matrix_metrics',
    'get_most_robust_classes',
    'get_least_robust_classes',
]
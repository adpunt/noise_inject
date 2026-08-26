"""
NoiseInject - A framework for testing ML model robustness to label noise
Supports both regression (continuous noise) and classification (label flips)
"""

from .core import (
    NoiseInjectorRegression,
    NoiseInjectorClassification,
    InjectionResult,
    dose_tolerance,
    CONDITIONS,
    REGRESSION_STRATEGIES,
    REGRESSION_DISTRIBUTIONS,
)

from .calibration import (
    # Classification calibration. Regression needs none: every condition
    # solves for its own scale in closed form -- see NOISE_DESIGN.md section 1.
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

from .uncertainty import calculate_uncertainty_metrics

__version__ = '1.0.0'

__all__ = [
    # Core classes
    'NoiseInjectorRegression',
    'NoiseInjectorClassification',
    'InjectionResult',
    'dose_tolerance',
    # The condition registry
    'CONDITIONS',
    'REGRESSION_STRATEGIES',
    'REGRESSION_DISTRIBUTIONS',
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
    # Uncertainty metrics
    'calculate_uncertainty_metrics',
]
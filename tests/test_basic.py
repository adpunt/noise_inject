"""
Basic unit tests for NoiseInject
Run with: pytest tests/
"""

import numpy as np
import pytest
from noiseInject import NoiseInjector, calculate_noise_metrics, calibrate_sigma


def test_legacy_noise():
    """Test basic legacy noise injection"""
    injector = NoiseInjector('legacy', random_state=42)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_noisy = injector.inject(y, sigma=0.1)
    
    assert len(y_noisy) == len(y)
    assert not np.array_equal(y, y_noisy)  # Should be different


def test_all_strategies():
    """Test all noise strategies run without error"""
    strategies = ['legacy', 'quantile', 'threshold', 'outlier', 'hetero', 'valprop']
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    for strategy in strategies:
        injector = NoiseInjector(strategy, random_state=42)
        y_noisy = injector.inject(y, sigma=0.1)
        assert len(y_noisy) == len(y)


def test_effective_noise():
    """Test effective noise calculation"""
    injector = NoiseInjector('legacy', random_state=42)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_noisy = injector.inject(y, sigma=0.5)
    
    effective = injector.get_effective_noise(y, y_noisy, method='std_normalized')
    assert effective > 0
    assert effective < 2.0  # Reasonable range


def test_metrics_calculation():
    """Test metrics calculation"""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predictions = {
        0.0: np.array([1.1, 2.1, 2.9, 4.1, 4.9]),
        0.1: np.array([1.2, 2.0, 3.1, 3.9, 5.1]),
    }
    
    metrics = calculate_noise_metrics(y_true, predictions)
    
    assert 'sigma' in metrics.columns
    assert 'r2' in metrics.columns
    assert len(metrics) == 3  # 2 sigma levels + 1 summary


def test_calibration():
    """Test sigma calibration"""
    y = np.random.randn(100)
    sigma = calibrate_sigma(y, target_effective_noise=0.1, random_state=42)
    
    assert sigma > 0
    assert sigma < 1.0  # Reasonable range


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Minimal test suite for NoiseInject.

Run with:  pytest tests/
Covers the core injection/calibration/metrics paths plus a regression test for the
`binary_asymmetric` copy-assignment bug (it previously flipped zero labels).
"""

import numpy as np
import pytest

from noiseInject import (
    NoiseInjectorRegression,
    NoiseInjectorClassification,
    calibrate_sigma,
    calibrate_flip_probability,
    calculate_noise_metrics,
    calculate_classification_metrics,
)


# --- regression injection ---------------------------------------------------

def test_regression_inject_changes_values_and_preserves_shape():
    y = np.linspace(-5, 5, 200)
    inj = NoiseInjectorRegression('legacy', random_state=0)
    y_noisy = inj.inject(y, sigma=1.0)
    assert y_noisy.shape == y.shape
    assert not np.allclose(y_noisy, y)


def test_regression_seed_is_reproducible():
    y = np.linspace(-5, 5, 200)
    a = NoiseInjectorRegression('legacy', random_state=42).inject(y, 1.0)
    b = NoiseInjectorRegression('legacy', random_state=42).inject(y, 1.0)
    assert np.allclose(a, b)


def test_invalid_strategy_raises():
    with pytest.raises(ValueError):
        NoiseInjectorRegression('not_a_strategy')


# --- classification injection ----------------------------------------------

def test_uniform_flips_some_labels():
    y = np.array([0, 1, 2] * 100)
    y_noisy = NoiseInjectorClassification('uniform', random_state=0).inject(y, 0.3)
    assert 0.0 < np.mean(y != y_noisy) < 1.0


def test_binary_asymmetric_actually_flips():
    """Regression test: binary_asymmetric used to assign into a copy and flip nothing."""
    y = np.array([0, 1] * 500)
    inj = NoiseInjectorClassification('binary_asymmetric', random_state=0)
    y_noisy = inj.inject(y, 0.2, flip_01_mult=1.5, flip_10_mult=0.5)
    overall = np.mean(y != y_noisy)
    flip_0to1 = np.mean(y_noisy[y == 0] != 0)
    flip_1to0 = np.mean(y_noisy[y == 1] != 1)
    assert overall > 0.0                      # the bug made this exactly 0
    assert flip_0to1 > flip_1to0              # asymmetry holds (0->1 rate > 1->0 rate)


# --- calibration ------------------------------------------------------------

def test_calibrate_sigma_hits_target_effective_noise():
    rng = np.random.RandomState(0)
    y = rng.normal(0, 1, 1000)
    sigma = calibrate_sigma(y, target_effective_noise=0.1, random_state=0)
    inj = NoiseInjectorRegression('legacy', random_state=0)
    eff = inj.get_effective_noise(y, inj.inject(y, sigma))
    assert abs(eff - 0.1) < 0.03


# --- metrics ----------------------------------------------------------------

def test_regression_metrics_return_expected_columns():
    rng = np.random.RandomState(0)
    y_true = rng.normal(0, 1, 100)
    predictions = {0.0: y_true.copy(), 1.0: y_true + rng.normal(0, 0.5, 100)}
    per_sigma, summary = calculate_noise_metrics(y_true, predictions)
    assert 'r2' in per_sigma.columns
    assert 'nsi_r2' in summary.columns
    assert 'retention_pct_r2' in summary.columns


def test_classification_metrics_return_three_frames():
    rng = np.random.RandomState(0)
    y_true = np.array([0, 1, 2] * 30)
    predictions = {0.0: y_true.copy(), 0.2: rng.permutation(y_true)}
    per_flip, summary, per_class = calculate_classification_metrics(y_true, predictions)
    assert 'accuracy' in per_flip.columns
    assert 'nsi_accuracy' in summary.columns
    assert 'class' in per_class.columns

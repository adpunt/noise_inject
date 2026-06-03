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
    calculate_uncertainty_metrics,
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


# --- uncertainty metrics ----------------------------------------------------

def test_uncertainty_metrics_return_expected_columns():
    rng = np.random.RandomState(0)
    y_true = rng.normal(0, 1, 500)
    predictions, uncertainties = {}, {}
    for sigma in (0.0, 0.5):
        err = rng.normal(0, 0.2 + sigma, 500)
        predictions[sigma] = y_true + err
        uncertainties[sigma] = np.full(500, 0.2 + sigma)
    per_sigma, summary = calculate_uncertainty_metrics(y_true, predictions, uncertainties)
    for col in ('unc_error_rho', 'ece', 'coverage_1sigma', 'coverage_2sigma',
                'mean_interval_width'):
        assert col in per_sigma.columns
    assert 'baseline_ece' in summary.columns
    assert 'miscoverage_1sigma' in summary.columns


def test_uncertainty_error_rho_high_when_u_tracks_error():
    """If predicted uncertainty equals the absolute error, the rank correlation is ~1."""
    rng = np.random.RandomState(1)
    y_true = rng.normal(0, 1, 300)
    err = rng.uniform(0, 2, 300)
    y_pred = y_true + err
    predictions = {0.0: y_pred}
    uncertainties = {0.0: np.abs(err)}  # u == |error| exactly
    per_sigma, _ = calculate_uncertainty_metrics(y_true, predictions, uncertainties)
    assert per_sigma['unc_error_rho'].iloc[0] > 0.99


def test_coverage_matches_gaussian_targets_when_well_calibrated():
    """Perfectly-calibrated Gaussian errors should hit ~68% / ~95% coverage."""
    rng = np.random.RandomState(2)
    n = 20000
    y_true = np.zeros(n)
    true_sigma = 0.5
    y_pred = y_true + rng.normal(0, true_sigma, n)
    predictions = {0.0: y_pred}
    uncertainties = {0.0: np.full(n, true_sigma)}
    per_sigma, _ = calculate_uncertainty_metrics(y_true, predictions, uncertainties)
    assert abs(per_sigma['coverage_1sigma'].iloc[0] - 0.6827) < 0.02
    assert abs(per_sigma['coverage_2sigma'].iloc[0] - 0.9545) < 0.02


def test_uncertainty_noise_rho_appears_only_with_injected_noise():
    rng = np.random.RandomState(3)
    y_true = rng.normal(0, 1, 400)
    predictions = {0.0: y_true.copy(), 0.5: y_true + rng.normal(0, 0.5, 400)}
    uncertainties = {0.0: np.full(400, 0.1), 0.5: np.full(400, 0.5)}

    per_sigma, _ = calculate_uncertainty_metrics(y_true, predictions, uncertainties)
    assert 'unc_noise_rho' not in per_sigma.columns

    # Inject noise whose magnitude is mirrored by the predicted uncertainty.
    eps = {s: rng.uniform(0, 1, 400) for s in (0.0, 0.5)}
    noise_unc = {s: eps[s].copy() for s in (0.0, 0.5)}
    per_sigma2, _ = calculate_uncertainty_metrics(
        y_true, predictions, uncertainties,
        injected_noise=eps, noise_uncertainties=noise_unc,
    )
    assert 'unc_noise_rho' in per_sigma2.columns
    assert per_sigma2['unc_noise_rho'].iloc[0] > 0.99


def test_split_conformal_achieves_target_coverage():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression
    from noiseInject.wrappers import SplitConformalRegressor

    X, y = make_regression(n_samples=900, n_features=8, noise=10.0, random_state=0)
    X_tr, X_cal, X_te = X[:500], X[500:700], X[700:]
    y_tr, y_cal, y_te = y[:500], y[500:700], y[700:]

    cp = SplitConformalRegressor(RandomForestRegressor(random_state=0), coverage=0.9)
    cp.fit(X_tr, y_tr, X_cal, y_cal)
    lo, hi = cp.predict_interval(X_te)
    coverage = np.mean((y_te >= lo) & (y_te <= hi))
    assert coverage > 0.8  # finite-sample, but should be near the 0.9 target


def test_mc_dropout_wrapper_returns_mean_and_std():
    torch = pytest.importorskip("torch")  # requires noiseInject[uncertainty]
    import torch.nn as nn
    from noiseInject.wrappers import MCDropoutRegressor

    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Dropout(0.3), nn.Linear(16, 1))
    X = np.random.RandomState(0).normal(size=(40, 5))
    mean, std = MCDropoutRegressor(net, n_forward=50).predict(X)
    assert mean.shape == (40,) and std.shape == (40,)
    assert np.all(std >= 0) and std.mean() > 0  # dropout induces non-zero spread


def test_gauche_gp_wrapper_rbf_fits_and_predicts():
    pytest.importorskip("gpytorch")  # requires noiseInject[uncertainty]
    from noiseInject.wrappers import GaucheGPRegressor

    rng = np.random.RandomState(0)
    X = rng.normal(size=(60, 4))
    y = X[:, 0] * 2 - X[:, 1] + rng.normal(0, 0.1, 60)
    gp = GaucheGPRegressor(kernel='rbf', training_iter=30).fit(X, y)
    mean, std = gp.predict(X[:10])
    assert mean.shape == (10,) and std.shape == (10,)
    assert np.all(std > 0)

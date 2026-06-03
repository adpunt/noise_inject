"""
NoiseInject Uncertainty Module
Calibration / uncertainty-quality metrics for probabilistic regression models,
evaluated across noise levels (mirrors metrics.calculate_noise_metrics).

Metrics follow the definitions in Punt (2026), Methods:
  - Uncertainty-error Spearman rho:  spearman(u, |y - y_hat|)
  - Uncertainty-noise Spearman rho:  spearman(u, |injected noise|)
  - Coverage(k-sigma):  mean( |y - y_hat| <= k * u ),  targets 68% (k=1), 95% (k=2)
  - ECE:  sum_b (|B_b| / N) * |mean_u_b - mean_error_b|, binned into deciles by u
  - Mean interval width:  mean( 2 * u )
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional, Tuple


# Theoretical Gaussian coverage targets used for miscoverage reporting.
COVERAGE_TARGETS = {1: 0.6827, 2: 0.9545}


def calculate_uncertainty_metrics(
    y_true: np.ndarray,
    predictions: Dict[float, np.ndarray],
    uncertainties: Dict[float, np.ndarray],
    injected_noise: Optional[Dict[float, np.ndarray]] = None,
    noise_uncertainties: Optional[Dict[float, np.ndarray]] = None,
    n_bins: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate uncertainty-quality metrics across noise levels for regression.

    The first four metrics are evaluated on the (clean-target) test set from
    ``predictions``/``uncertainties``. The uncertainty-noise correlation needs the
    per-sample injected noise, which is only known on the corrupted (training) set,
    so it is computed from a separate, parallel pair of dicts
    (``injected_noise`` + ``noise_uncertainties``) when both are supplied.

    Args:
        y_true: True target values (clean, for the test set).
        predictions: Dict mapping sigma -> test predictions.
        uncertainties: Dict mapping sigma -> per-sample predicted uncertainty (std)
                       on the test set, parallel to ``predictions``.
        injected_noise: Optional dict mapping sigma -> per-sample injected noise
                        (e.g. y_noisy - y_clean) on the corrupted set.
        noise_uncertainties: Optional dict mapping sigma -> predicted uncertainty on
                             the same samples as ``injected_noise``. Required (with
                             ``injected_noise``) to produce the uncertainty-noise rho.
        n_bins: Number of equal-count bins (deciles by default) for ECE.

    Returns:
        per_sigma_df: One row per sigma level. Columns: sigma, unc_error_rho, ece,
                      coverage_1sigma, coverage_2sigma, mean_interval_width, and
                      unc_noise_rho when injected noise is provided.
        summary_df: Aggregate metrics. Slopes of each quantity vs sigma (slope_*),
                    clean baselines (baseline_*), and mean coverage / miscoverage.
    """
    y_true = np.asarray(y_true).flatten()

    sigma_values = sorted(predictions.keys())

    has_noise_tracking = injected_noise is not None and noise_uncertainties is not None

    results = []
    for sigma in sigma_values:
        y_pred = np.asarray(predictions[sigma]).flatten()
        u = np.asarray(uncertainties[sigma]).flatten()
        error = np.abs(y_true - y_pred)

        row = {'sigma': sigma}
        row['unc_error_rho'] = _spearman(u, error)
        row['ece'] = _ece(u, error, n_bins=n_bins)
        row['coverage_1sigma'] = _coverage(error, u, k=1)
        row['coverage_2sigma'] = _coverage(error, u, k=2)
        row['mean_interval_width'] = float(np.mean(2.0 * u))

        if has_noise_tracking and sigma in injected_noise and sigma in noise_uncertainties:
            eps = np.abs(np.asarray(injected_noise[sigma]).flatten())
            u_noise = np.asarray(noise_uncertainties[sigma]).flatten()
            row['unc_noise_rho'] = _spearman(u_noise, eps)

        results.append(row)

    per_sigma_df = pd.DataFrame(results)

    summary = {}
    track = ['unc_error_rho', 'ece', 'coverage_1sigma', 'coverage_2sigma',
             'mean_interval_width']
    if 'unc_noise_rho' in per_sigma_df.columns:
        track.append('unc_noise_rho')

    for metric in track:
        col = per_sigma_df[metric].values
        if len(per_sigma_df) > 1:
            slope, _, r_value, p_value, _ = stats.linregress(
                per_sigma_df['sigma'].values, col
            )
            summary[f'slope_{metric}'] = slope
            summary[f'slope_{metric}_pval'] = p_value
            summary[f'slope_{metric}_r'] = r_value

        baseline_val = per_sigma_df[per_sigma_df['sigma'] == 0.0][metric].values
        if len(baseline_val) > 0:
            summary[f'baseline_{metric}'] = baseline_val[0]

        summary[f'mean_{metric}'] = float(np.mean(col))

    # Miscoverage: how far empirical coverage sits from the Gaussian target.
    for k in (1, 2):
        col = f'coverage_{k}sigma'
        summary[f'miscoverage_{k}sigma'] = float(
            np.mean(np.abs(per_sigma_df[col].values - COVERAGE_TARGETS[k]))
        )

    # Pooled uncertainty-noise correlation across all levels. Within a single level,
    # feature-independent noise is not per-sample detectable (per-level rho ~ 0); the
    # signal lives across levels, where higher sigma raises both injected noise and
    # predicted uncertainty. This pooled value is the paper's noise-tracking metric.
    if has_noise_tracking:
        eps_all = np.concatenate([
            np.abs(np.asarray(injected_noise[s]).flatten())
            for s in sigma_values if s in injected_noise
        ])
        u_all = np.concatenate([
            np.asarray(noise_uncertainties[s]).flatten()
            for s in sigma_values if s in noise_uncertainties
        ])
        summary['unc_noise_rho_pooled'] = _spearman(u_all, eps_all)

    summary_df = pd.DataFrame([summary])

    return per_sigma_df, summary_df


def _spearman(u: np.ndarray, target: np.ndarray) -> float:
    """Spearman rank correlation, ignoring non-finite pairs. NaN if degenerate."""
    mask = np.isfinite(u) & np.isfinite(target)
    if mask.sum() < 3:
        return np.nan
    u, target = u[mask], target[mask]
    # Spearman is undefined when either side is constant (e.g. homoscedastic u).
    if np.ptp(u) == 0 or np.ptp(target) == 0:
        return np.nan
    rho, _ = stats.spearmanr(u, target)
    return rho


def _coverage(error: np.ndarray, u: np.ndarray, k: int) -> float:
    """Empirical coverage at k-sigma: fraction of |y - y_hat| within k * u."""
    mask = np.isfinite(error) & np.isfinite(u)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(error[mask] <= k * u[mask]))


def _ece(u: np.ndarray, error: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error: bin by predicted uncertainty (equal-count deciles)
    and take the sample-weighted absolute gap between mean uncertainty and mean error.
    """
    mask = np.isfinite(u) & np.isfinite(error) & (u > 0)
    u = u[mask]
    error = error[mask]
    if len(u) == 0:
        return np.nan

    bins = np.unique(np.percentile(u, np.linspace(0, 100, n_bins + 1)))

    ece = 0.0
    for i in range(len(bins) - 1):
        # Include the right edge in the final bin so the largest value is counted.
        if i == len(bins) - 2:
            in_bin = (u >= bins[i]) & (u <= bins[i + 1])
        else:
            in_bin = (u >= bins[i]) & (u < bins[i + 1])
        if in_bin.sum() > 0:
            bin_weight = in_bin.sum() / len(u)
            ece += bin_weight * np.abs(u[in_bin].mean() - error[in_bin].mean())

    return ece

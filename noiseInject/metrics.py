"""
NoiseInject Metrics Module
Calculate noise sensitivity metrics for model evaluation
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional


def calculate_noise_metrics(
    y_true: np.ndarray,
    predictions: Dict[float, np.ndarray],
    y_noisy_dict: Optional[Dict[float, np.ndarray]] = None,
    metrics: List[str] = ['r2', 'rmse', 'mae'],
    high_sigma: Optional[float] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate comprehensive noise sensitivity metrics.
    
    Args:
        y_true: True target values (clean, for test set)
        predictions: Dict mapping sigma -> predictions
                    Example: {0.0: pred_clean, 0.1: pred_low_noise, 0.3: pred_high_noise}
        y_noisy_dict: Optional dict mapping sigma -> noisy y values (for effective noise calc)
        metrics: List of metrics to calculate ['r2', 'rmse', 'mae']
        high_sigma: Sigma level to use for retention calculation (default: max sigma)
    
    Returns:
        per_sigma_df: DataFrame with one row per sigma level (columns: sigma, r2, rmse, mae)
        summary_df: DataFrame with aggregate metrics (columns: nsi_*, baseline_*, retention_*)
    """
    y_true = np.asarray(y_true).flatten()
    
    # Sort sigma values
    sigma_values = sorted(predictions.keys())
    
    if high_sigma is None:
        high_sigma = max(sigma_values)
    
    # Calculate metrics for each sigma
    results = []
    for sigma in sigma_values:
        y_pred = np.asarray(predictions[sigma]).flatten()
        
        row = {'sigma': sigma}
        
        # Calculate performance metrics
        for metric in metrics:
            if metric == 'r2':
                row['r2'] = _r2_score(y_true, y_pred)
            elif metric == 'rmse':
                row['rmse'] = _rmse(y_true, y_pred)
            elif metric == 'mae':
                row['mae'] = _mae(y_true, y_pred)
        
        # Calculate effective noise if provided
        if y_noisy_dict is not None and sigma in y_noisy_dict:
            from .core import NoiseInjector
            injector = NoiseInjector()
            row['effective_noise'] = injector.get_effective_noise(
                y_true, y_noisy_dict[sigma], method='std_normalized'
            )
        
        results.append(row)
    
    per_sigma_df = pd.DataFrame(results)
    
    # Calculate NSI (Noise Sensitivity Index) - slope of performance vs sigma
    summary = {}
    
    for metric in metrics:
        if metric in per_sigma_df.columns:
            # NSI: slope from linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                per_sigma_df['sigma'].values, per_sigma_df[metric].values
            )
            
            summary[f'nsi_{metric}'] = slope
            summary[f'nsi_{metric}_pval'] = p_value
            summary[f'nsi_{metric}_r'] = r_value
            
            # Relative NSI (normalized by baseline)
            baseline_val = per_sigma_df[per_sigma_df['sigma'] == 0.0][metric].values
            if len(baseline_val) > 0 and abs(baseline_val[0]) > 1e-10:
                summary[f'nsi_{metric}_relative'] = slope / abs(baseline_val[0])
            
            # Baseline performance
            if len(baseline_val) > 0:
                summary[f'baseline_{metric}'] = baseline_val[0]
            
            # Retention at high sigma
            high_sigma_val = per_sigma_df[per_sigma_df['sigma'] == high_sigma][metric].values
            if len(baseline_val) > 0 and len(high_sigma_val) > 0:
                if metric == 'r2':
                    # For R² (higher is better): (R²_high / R²_baseline) * 100
                    if baseline_val[0] > 1e-10:
                        summary[f'retention_pct_{metric}'] = (high_sigma_val[0] / baseline_val[0]) * 100
                else:
                    # For RMSE/MAE (lower is better): (baseline / high) * 100
                    if high_sigma_val[0] > 1e-10:
                        summary[f'retention_pct_{metric}'] = (baseline_val[0] / high_sigma_val[0]) * 100
    
    summary_df = pd.DataFrame([summary])
    
    return per_sigma_df, summary_df


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² score"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot < 1e-10:
        return 0.0
    
    return 1 - (ss_res / ss_tot)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate RMSE"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAE"""
    return np.mean(np.abs(y_true - y_pred))
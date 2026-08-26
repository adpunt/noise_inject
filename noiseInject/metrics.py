"""
NoiseInject Metrics Module
Calculate noise-robustness metrics for model evaluation
Supports both regression and classification

Robustness is summarised from the *retention curve* ret(x) = metric(x) / metric(0)
(x = sigma for regression, flip probability for classification):

  - auc_norm:  normalised area under the retention curve, trapezoid(ret, x) / range(x).
               The mean fraction of baseline skill retained across the noise sweep.
               Higher = more robust (~[0, 1]); decoupled from baseline performance.
  - weibull_beta / weibull_tau:  fit ret ~= exp(-(x / tau) ** beta). beta is the shape of
               failure (beta > 1 = holds then cliff, beta < 1 = early collapse then plateau),
               tau the decay scale. Supplementary; only meaningful once the model genuinely
               degrades.

These replace the old NSI (linear slope of performance vs noise), which mischaracterised the
nonlinear degradation curve and was coupled to baseline performance. See Punt (2026), Methods.

Both scalars are defined only on bounded, higher-is-better skill metrics (r2 for regression;
accuracy/f1/precision/recall for classification), whose retention falls from 1 toward 0.
Error metrics (rmse, mae) grow with noise and keep only their per-level curve plus
baseline/retention_pct. On a single curve auc_norm is unclipped and can blow up if a retrain
collapses (retention < 0); ``curve_stable_{metric}`` flags that case. For the most reliable
scalars, build ``predictions`` from seed-averaged per-noise-level performance (as in the paper).
"""

import warnings

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from typing import Dict, List, Optional, Tuple, Union


# Bounded, higher-is-better metrics whose retention curve falls from 1 toward 0, so
# auc_norm / Weibull are meaningful on them.
_CURVE_METRICS_REGRESSION = {'r2'}
_CURVE_METRICS_CLASSIFICATION = {
    'accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro',
}


def _retention_auc_norm(x: np.ndarray, y: np.ndarray, base: float) -> float:
    """Normalised area under the retention curve y(x) / base.

    Primary robustness scalar. HIGHER = more robust (~[0, 1]); baseline-decoupled, no shape
    assumption. Unclipped: a curve that collapses below zero retention yields a large negative
    value (caught by the ``curve_stable`` flag rather than silently masked).
    """
    ret = y / base
    xrange = x.max() - x.min()
    return float(trapezoid(ret, x) / xrange) if xrange > 0 else np.nan


def _retention_weibull(x: np.ndarray, y: np.ndarray, base: float) -> Tuple[float, float]:
    """Fit retention ~= exp(-(x / tau) ** beta); return (tau, beta).

    Supplementary shape descriptor. beta > 1 = delayed cliff (holds then collapses),
    beta < 1 = early collapse (fast then plateaus); tau is the x scale of decay. Needs at least
    4 noise levels for the 2-parameter fit; returns (nan, nan) otherwise or on fit failure.
    """
    if len(x) < 4:
        return np.nan, np.nan
    ret = np.clip(y / base, 1e-3, None)
    try:
        p, _ = curve_fit(
            lambda s, tau, beta: np.exp(-np.power(np.clip(s, 0, None) / tau, beta)),
            x, ret, p0=[0.5, 1.5], maxfev=10000,
            bounds=([1e-3, 0.1], [np.inf, 10.0]),
        )
        return float(p[0]), float(p[1])
    except Exception:
        return np.nan, np.nan


def _curve_robustness(
    x: np.ndarray, y: np.ndarray, base: float, baseline_threshold: Optional[float]
) -> Dict[str, float]:
    """auc_norm / weibull_beta / weibull_tau / curve_stable for one retention curve.

    Returns NaN scalars (and warns) when the clean baseline is below ``baseline_threshold`` or
    effectively zero -- a near-random model has no meaningful degradation to measure.
    """
    if abs(base) < 1e-10 or (baseline_threshold is not None and base < baseline_threshold):
        if baseline_threshold is not None and abs(base) >= 1e-10:
            warnings.warn(
                f"baseline {base:.4g} below baseline_threshold {baseline_threshold}; "
                "auc_norm / Weibull set to NaN (model too weak to assess degradation)."
            )
        return {
            'auc_norm': np.nan, 'weibull_beta': np.nan,
            'weibull_tau': np.nan, 'curve_stable': np.nan,
        }
    auc_norm = _retention_auc_norm(x, y, base)
    tau, beta = _retention_weibull(x, y, base)
    return {
        'auc_norm': auc_norm,
        'weibull_beta': beta,
        'weibull_tau': tau,
        'curve_stable': bool(np.min(y / base) >= 0.0),
    }


# ============================================================================
# REGRESSION METRICS
# ============================================================================

def calculate_noise_metrics(
    y_true: np.ndarray,
    predictions: Dict[float, np.ndarray],
    y_noisy_dict: Optional[Dict[float, np.ndarray]] = None,
    metrics: List[str] = ['r2', 'rmse', 'mae'],
    high_sigma: Optional[float] = None,
    baseline_threshold: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate comprehensive noise-robustness metrics for regression.

    Args:
        y_true: True target values (clean, for test set)
        predictions: Dict mapping sigma -> predictions
                    Example: {0.0: pred_clean, 0.1: pred_low_noise, 0.3: pred_high_noise}
        y_noisy_dict: Optional dict mapping sigma -> noisy y values (for effective noise calc)
        metrics: List of metrics to calculate ['r2', 'rmse', 'mae']
        high_sigma: Sigma level to use for retention calculation (default: max sigma)
        baseline_threshold: If set, skill metrics whose clean baseline is below this value get
                    NaN auc_norm/Weibull (a near-random model has no meaningful degradation to
                    measure); the paper uses 0.3. Default None = always compute.

    Returns:
        per_sigma_df: DataFrame with one row per sigma level (columns: sigma, r2, rmse, mae)
        summary_df: DataFrame with aggregate metrics. For skill metrics (r2): auc_norm_*,
                    weibull_beta_*, weibull_tau_*, curve_stable_*. For all metrics: baseline_*
                    and retention_pct_*.
    """
    y_true = np.asarray(y_true).flatten()
    
    sigma_values = sorted(predictions.keys())
    
    if high_sigma is None:
        high_sigma = max(sigma_values)
    
    results = []
    for sigma in sigma_values:
        y_pred = np.asarray(predictions[sigma]).flatten()
        
        row = {'sigma': sigma}
        
        for metric in metrics:
            if metric == 'r2':
                row['r2'] = _r2_score(y_true, y_pred)
            elif metric == 'rmse':
                row['rmse'] = _rmse(y_true, y_pred)
            elif metric == 'mae':
                row['mae'] = _mae(y_true, y_pred)
        
        if y_noisy_dict is not None and sigma in y_noisy_dict:
            from .core import NoiseInjectorRegression
            injector = NoiseInjectorRegression()
            row['effective_noise'] = injector.get_effective_noise(
                y_true, y_noisy_dict[sigma], method='std_normalized'
            )
        
        results.append(row)
    
    per_sigma_df = pd.DataFrame(results)
    
    summary = {}

    sigma_arr = per_sigma_df['sigma'].values.astype(float)

    for metric in metrics:
        if metric not in per_sigma_df.columns:
            continue

        baseline_val = per_sigma_df[per_sigma_df['sigma'] == 0.0][metric].values
        if len(baseline_val) == 0:
            continue
        base = baseline_val[0]
        summary[f'baseline_{metric}'] = base

        high_sigma_val = per_sigma_df[per_sigma_df['sigma'] == high_sigma][metric].values
        if len(high_sigma_val) > 0:
            if metric == 'r2':
                if base > 1e-10:
                    summary[f'retention_pct_{metric}'] = (high_sigma_val[0] / base) * 100
            else:
                if high_sigma_val[0] > 1e-10:
                    summary[f'retention_pct_{metric}'] = (base / high_sigma_val[0]) * 100

        # Curve-descriptive robustness scalars, skill metrics only.
        if metric in _CURVE_METRICS_REGRESSION:
            curve = _curve_robustness(
                sigma_arr, per_sigma_df[metric].values.astype(float), base, baseline_threshold
            )
            summary[f'auc_norm_{metric}'] = curve['auc_norm']
            summary[f'weibull_beta_{metric}'] = curve['weibull_beta']
            summary[f'weibull_tau_{metric}'] = curve['weibull_tau']
            summary[f'curve_stable_{metric}'] = curve['curve_stable']

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


# ============================================================================
# CLASSIFICATION METRICS
# ============================================================================

def calculate_classification_metrics(
    y_true: np.ndarray,
    predictions: Dict[float, np.ndarray],
    y_noisy_dict: Optional[Dict[float, np.ndarray]] = None,
    high_flip_prob: Optional[float] = None,
    baseline_threshold: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate comprehensive noise-robustness metrics for classification.

    Args:
        y_true: True class labels (clean, for test set)
        predictions: Dict mapping flip_probability -> predictions
                    Example: {0.0: pred_clean, 0.1: pred_low_noise, 0.3: pred_high_noise}
        y_noisy_dict: Optional dict mapping flip_probability -> noisy y values
                     (for effective flip rate calculation)
        high_flip_prob: Flip probability level to use for retention calculation
                       (default: max flip_prob)
        baseline_threshold: If set, skill metrics whose clean baseline is below this value get
                    NaN auc_norm/Weibull. Default None = always compute.

    Returns:
        per_flip_df: DataFrame with one row per flip probability level
                    Columns: flip_prob, accuracy, precision_macro, recall_macro,
                            f1_macro, f1_weighted, effective_flip_rate
        summary_df: DataFrame with aggregate metrics. For skill metrics (accuracy, f1_macro,
                    f1_weighted, precision_macro, recall_macro, and per-class f1): auc_norm_*,
                    weibull_beta_*, weibull_tau_*, curve_stable_*. For all: baseline_*,
                    retention_pct_*.
        per_class_df: DataFrame with per-class metrics
                     Columns: class, flip_prob, f1_score, precision, recall, support
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    y_true = np.asarray(y_true).flatten()
    
    flip_prob_values = sorted(predictions.keys())
    
    if high_flip_prob is None:
        high_flip_prob = max(flip_prob_values)
    
    unique_classes = np.unique(y_true)
    n_classes = len(unique_classes)
    
    results = []
    per_class_results = []
    
    for flip_prob in flip_prob_values:
        y_pred = np.asarray(predictions[flip_prob]).flatten()
        
        row = {'flip_prob': flip_prob}
        
        row['accuracy'] = accuracy_score(y_true, y_pred)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        row['precision_macro'] = np.mean(precision)
        row['recall_macro'] = np.mean(recall)
        row['f1_macro'] = np.mean(f1)
        
        row['precision_weighted'] = np.average(precision, weights=support)
        row['recall_weighted'] = np.average(recall, weights=support)
        row['f1_weighted'] = np.average(f1, weights=support)
        
        if y_noisy_dict is not None and flip_prob in y_noisy_dict:
            from .core import NoiseInjectorClassification
            injector = NoiseInjectorClassification()
            row['effective_flip_rate'] = injector.get_effective_flip_rate(
                y_true, y_noisy_dict[flip_prob]
            )
        
        results.append(row)
        
        for cls_idx, cls in enumerate(unique_classes):
            per_class_results.append({
                'class': cls,
                'flip_prob': flip_prob,
                'f1_score': f1[cls_idx],
                'precision': precision[cls_idx],
                'recall': recall[cls_idx],
                'support': support[cls_idx]
            })
    
    per_flip_df = pd.DataFrame(results)
    per_class_df = pd.DataFrame(per_class_results)
    
    summary = {}

    flip_arr = per_flip_df['flip_prob'].values.astype(float)

    metrics_to_track = ['accuracy', 'f1_macro', 'f1_weighted',
                       'precision_macro', 'recall_macro']

    for metric in metrics_to_track:
        if metric not in per_flip_df.columns:
            continue

        baseline_val = per_flip_df[per_flip_df['flip_prob'] == 0.0][metric].values
        if len(baseline_val) == 0:
            continue
        base = baseline_val[0]
        summary[f'baseline_{metric}'] = base

        high_flip_val = per_flip_df[per_flip_df['flip_prob'] == high_flip_prob][metric].values
        if len(high_flip_val) > 0 and base > 1e-10:
            summary[f'retention_pct_{metric}'] = (high_flip_val[0] / base) * 100

        if metric in _CURVE_METRICS_CLASSIFICATION:
            curve = _curve_robustness(
                flip_arr, per_flip_df[metric].values.astype(float), base, baseline_threshold
            )
            summary[f'auc_norm_{metric}'] = curve['auc_norm']
            summary[f'weibull_beta_{metric}'] = curve['weibull_beta']
            summary[f'weibull_tau_{metric}'] = curve['weibull_tau']
            summary[f'curve_stable_{metric}'] = curve['curve_stable']

    for cls in unique_classes:
        cls_data = per_class_df[per_class_df['class'] == cls].sort_values('flip_prob')

        if len(cls_data) <= 1:
            continue

        baseline_f1 = cls_data[cls_data['flip_prob'] == 0.0]['f1_score'].values
        if len(baseline_f1) == 0:
            continue
        base = baseline_f1[0]
        summary[f'baseline_f1_class_{cls}'] = base

        high_f1 = cls_data[cls_data['flip_prob'] == high_flip_prob]['f1_score'].values
        if len(high_f1) > 0 and base > 1e-10:
            summary[f'retention_pct_f1_class_{cls}'] = (high_f1[0] / base) * 100

        curve = _curve_robustness(
            cls_data['flip_prob'].values.astype(float),
            cls_data['f1_score'].values.astype(float), base, baseline_threshold
        )
        summary[f'auc_norm_f1_class_{cls}'] = curve['auc_norm']
        summary[f'weibull_beta_f1_class_{cls}'] = curve['weibull_beta']
        summary[f'weibull_tau_f1_class_{cls}'] = curve['weibull_tau']
        summary[f'curve_stable_f1_class_{cls}'] = curve['curve_stable']

    summary_df = pd.DataFrame([summary])
    
    return per_flip_df, summary_df, per_class_df


def calculate_confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> pd.DataFrame:
    """
    Calculate confusion matrix as a DataFrame (classification only).
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
    
    Returns:
        DataFrame with confusion matrix (rows=true, cols=predicted)
    """
    from sklearn.metrics import confusion_matrix
    
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    cm = confusion_matrix(y_true, y_pred)
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    
    df = pd.DataFrame(
        cm,
        index=[f'true_{cls}' for cls in unique_classes],
        columns=[f'pred_{cls}' for cls in unique_classes]
    )
    
    return df


def _class_auc_norms(summary_df: pd.DataFrame) -> List[Tuple[int, float]]:
    """Collect (class, auc_norm_f1) pairs from a classification summary, skipping NaN."""
    prefix = 'auc_norm_f1_class_'
    pairs = []
    for col in summary_df.columns:
        if not col.startswith(prefix):
            continue
        try:
            cls = int(col[len(prefix):])
        except ValueError:
            continue
        val = summary_df[col].values[0]
        if pd.notna(val):
            pairs.append((cls, val))
    return pairs


def get_most_robust_classes(summary_df: pd.DataFrame, n: int = 3) -> List[Tuple[int, float]]:
    """
    Identify the most robust classes (highest auc_norm) for classification.

    Args:
        summary_df: Summary DataFrame from calculate_classification_metrics
        n: Number of top classes to return

    Returns:
        List of (class, auc_norm_f1) tuples, sorted by robustness (highest auc_norm first)
    """
    pairs = _class_auc_norms(summary_df)
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:n]


def get_least_robust_classes(summary_df: pd.DataFrame, n: int = 3) -> List[Tuple[int, float]]:
    """
    Identify the least robust classes (lowest auc_norm) for classification.

    Args:
        summary_df: Summary DataFrame from calculate_classification_metrics
        n: Number of bottom classes to return

    Returns:
        List of (class, auc_norm_f1) tuples, sorted by fragility (lowest auc_norm first)
    """
    pairs = _class_auc_norms(summary_df)
    pairs.sort(key=lambda x: x[1])
    return pairs[:n]
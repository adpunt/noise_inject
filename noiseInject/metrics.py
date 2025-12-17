"""
NoiseInject Metrics Module
Calculate noise sensitivity metrics for model evaluation
Supports both regression and classification
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union


# ============================================================================
# REGRESSION METRICS
# ============================================================================

def calculate_noise_metrics(
    y_true: np.ndarray,
    predictions: Dict[float, np.ndarray],
    y_noisy_dict: Optional[Dict[float, np.ndarray]] = None,
    metrics: List[str] = ['r2', 'rmse', 'mae'],
    high_sigma: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate comprehensive noise sensitivity metrics for regression.
    
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
    
    for metric in metrics:
        if metric in per_sigma_df.columns:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                per_sigma_df['sigma'].values, per_sigma_df[metric].values
            )
            
            summary[f'nsi_{metric}'] = slope
            summary[f'nsi_{metric}_pval'] = p_value
            summary[f'nsi_{metric}_r'] = r_value
            
            baseline_val = per_sigma_df[per_sigma_df['sigma'] == 0.0][metric].values
            if len(baseline_val) > 0 and abs(baseline_val[0]) > 1e-10:
                summary[f'nsi_{metric}_relative'] = slope / abs(baseline_val[0])
            
            if len(baseline_val) > 0:
                summary[f'baseline_{metric}'] = baseline_val[0]
            
            high_sigma_val = per_sigma_df[per_sigma_df['sigma'] == high_sigma][metric].values
            if len(baseline_val) > 0 and len(high_sigma_val) > 0:
                if metric == 'r2':
                    if baseline_val[0] > 1e-10:
                        summary[f'retention_pct_{metric}'] = (high_sigma_val[0] / baseline_val[0]) * 100
                else:
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


# ============================================================================
# CLASSIFICATION METRICS
# ============================================================================

def calculate_classification_metrics(
    y_true: np.ndarray,
    predictions: Dict[float, np.ndarray],
    y_noisy_dict: Optional[Dict[float, np.ndarray]] = None,
    high_flip_prob: Optional[float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate comprehensive noise sensitivity metrics for classification.
    
    Args:
        y_true: True class labels (clean, for test set)
        predictions: Dict mapping flip_probability -> predictions
                    Example: {0.0: pred_clean, 0.1: pred_low_noise, 0.3: pred_high_noise}
        y_noisy_dict: Optional dict mapping flip_probability -> noisy y values 
                     (for effective flip rate calculation)
        high_flip_prob: Flip probability level to use for retention calculation 
                       (default: max flip_prob)
    
    Returns:
        per_flip_df: DataFrame with one row per flip probability level
                    Columns: flip_prob, accuracy, precision_macro, recall_macro, 
                            f1_macro, f1_weighted, effective_flip_rate
        summary_df: DataFrame with aggregate metrics
                   Columns: nsi_*, baseline_*, retention_pct_*
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
    
    metrics_to_track = ['accuracy', 'f1_macro', 'f1_weighted', 
                       'precision_macro', 'recall_macro']
    
    for metric in metrics_to_track:
        if metric in per_flip_df.columns:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                per_flip_df['flip_prob'].values, per_flip_df[metric].values
            )
            
            summary[f'nsi_{metric}'] = slope
            summary[f'nsi_{metric}_pval'] = p_value
            summary[f'nsi_{metric}_r'] = r_value
            
            baseline_val = per_flip_df[per_flip_df['flip_prob'] == 0.0][metric].values
            if len(baseline_val) > 0 and abs(baseline_val[0]) > 1e-10:
                summary[f'nsi_{metric}_relative'] = slope / abs(baseline_val[0])
            
            if len(baseline_val) > 0:
                summary[f'baseline_{metric}'] = baseline_val[0]
            
            high_flip_val = per_flip_df[per_flip_df['flip_prob'] == high_flip_prob][metric].values
            if len(baseline_val) > 0 and len(high_flip_val) > 0:
                if baseline_val[0] > 1e-10:
                    summary[f'retention_pct_{metric}'] = (high_flip_val[0] / baseline_val[0]) * 100
    
    for cls in unique_classes:
        cls_data = per_class_df[per_class_df['class'] == cls]
        
        if len(cls_data) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                cls_data['flip_prob'].values, cls_data['f1_score'].values
            )
            
            summary[f'nsi_f1_class_{cls}'] = slope
            summary[f'nsi_f1_class_{cls}_pval'] = p_value
            
            baseline_f1 = cls_data[cls_data['flip_prob'] == 0.0]['f1_score'].values
            if len(baseline_f1) > 0:
                summary[f'baseline_f1_class_{cls}'] = baseline_f1[0]
                
                high_f1 = cls_data[cls_data['flip_prob'] == high_flip_prob]['f1_score'].values
                if len(high_f1) > 0 and baseline_f1[0] > 1e-10:
                    summary[f'retention_pct_f1_class_{cls}'] = (high_f1[0] / baseline_f1[0]) * 100
    
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


def get_most_robust_classes(summary_df: pd.DataFrame, n: int = 3) -> List[Tuple[int, float]]:
    """
    Identify the most robust classes (least negative NSI) for classification.
    
    Args:
        summary_df: Summary DataFrame from calculate_classification_metrics
        n: Number of top classes to return
    
    Returns:
        List of (class, nsi_f1) tuples, sorted by robustness (least negative NSI first)
    """
    class_nsi = []
    for col in summary_df.columns:
        if col.startswith('nsi_f1_class_') and not col.endswith('_pval'):
            cls_str = col.replace('nsi_f1_class_', '')
            try:
                cls = int(cls_str)
                nsi_val = summary_df[col].values[0]
                class_nsi.append((cls, nsi_val))
            except ValueError:
                continue
    
    class_nsi.sort(key=lambda x: x[1], reverse=True)
    
    return class_nsi[:n]


def get_least_robust_classes(summary_df: pd.DataFrame, n: int = 3) -> List[Tuple[int, float]]:
    """
    Identify the least robust classes (most negative NSI) for classification.
    
    Args:
        summary_df: Summary DataFrame from calculate_classification_metrics
        n: Number of bottom classes to return
    
    Returns:
        List of (class, nsi_f1) tuples, sorted by fragility (most negative NSI first)
    """
    class_nsi = []
    for col in summary_df.columns:
        if col.startswith('nsi_f1_class_') and not col.endswith('_pval'):
            cls_str = col.replace('nsi_f1_class_', '')
            try:
                cls = int(cls_str)
                nsi_val = summary_df[col].values[0]
                class_nsi.append((cls, nsi_val))
            except ValueError:
                continue
    
    class_nsi.sort(key=lambda x: x[1])
    
    return class_nsi[:n]
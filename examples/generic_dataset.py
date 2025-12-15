"""
Example: Generic Dataset with NoiseInject
For any regression task (not chemistry-specific)

This example shows:
1. Loading your own CSV/pandas data
2. Using any sklearn-compatible model
3. Testing noise robustness systematically
4. Comparing models and strategies
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import os

from noiseInject import NoiseInjector, calculate_noise_metrics, calibrate_sigma

# Disable multiprocessing
os.environ['OMP_NUM_THREADS'] = '1'


# ============================================================================
# LOAD YOUR DATA
# ============================================================================

def load_custom_data():
    """
    Load your own dataset.
    
    Replace this function with your own data loading code.
    Expected format:
        X: feature matrix (N x D)
        y: target values (N,)
    """
    # Example: California housing dataset
    data = fetch_california_housing()
    X = data.data
    y = data.target
    
    print(f"Dataset: California Housing")
    print(f"Samples: {len(X)}, Features: {X.shape[1]}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    return X, y


def load_from_csv(filepath, target_column, feature_columns=None):
    """
    Load data from CSV file.
    
    Args:
        filepath: Path to CSV file
        target_column: Name of target column
        feature_columns: List of feature column names (None = all except target)
    
    Returns:
        X, y: Features and targets
    """
    df = pd.read_csv(filepath)
    
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    
    X = df[feature_columns].values
    y = df[target_column].values
    
    print(f"Loaded from: {filepath}")
    print(f"Samples: {len(X)}, Features: {X.shape[1]}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    return X, y


# ============================================================================
# SINGLE MODEL ANALYSIS
# ============================================================================

def analyze_model_robustness(X_train, y_train, X_test, y_test, 
                             model, model_name="Model"):
    """
    Analyze noise robustness for a single model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        model: sklearn-compatible model
        model_name: Name for display
    """
    print(f"\n{'='*70}")
    print(f"ANALYZING: {model_name}")
    print(f"{'='*70}")
    
    # Calibrate sigma
    print("\n1. Calibrating sigma for 10% effective noise...")
    injector = NoiseInjector('legacy', random_state=42)
    sigma_cal = calibrate_sigma(y_train, target_effective_noise=0.1, random_state=42)
    print(f"   Calibrated σ = {sigma_cal:.4f}")
    
    # Test at multiple levels
    print("\n2. Testing at multiple noise levels...")
    predictions = {}
    sigma_multipliers = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    for mult in sigma_multipliers:
        sigma = sigma_cal * mult
        
        # Inject noise
        if mult == 0.0:
            y_noisy = y_train
        else:
            y_noisy = injector.inject(y_train, sigma)
        
        # Train
        model_copy = type(model)(**model.get_params())
        model_copy.fit(X_train, y_noisy)
        
        # Predict
        predictions[sigma] = model_copy.predict(X_test)
        
        r2 = model_copy.score(X_test, y_test)
        print(f"   σ={sigma:.4f} (×{mult}): R²={r2:.4f}")
    
    # Calculate metrics
    print("\n3. Calculating metrics...")
    per_sigma, summary_df = calculate_noise_metrics(y_test, predictions)
    
    # Display results
    summary = summary_df.iloc[0]
    
    print(f"\n{'='*70}")
    print(f"RESULTS FOR {model_name}")
    print(f"{'='*70}")
    print(f"Baseline R²:     {summary['baseline_r2']:.4f}")
    print(f"NSI (R²):        {summary['nsi_r2']:.4f}")
    print(f"Retention:       {summary['retention_pct_r2']:.2f}%")
    print(f"{'='*70}")
    
    return per_sigma, summary_df


# ============================================================================
# COMPARE MULTIPLE MODELS
# ============================================================================

def compare_models(X_train, y_train, X_test, y_test):
    """
    Compare noise robustness across multiple model types.
    """
    print("\n" + "="*70)
    print("COMPARING MULTIPLE MODELS")
    print("="*70)
    
    # Define models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Ridge Regression': Ridge(alpha=1.0),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100,), random_state=42, max_iter=500)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*70}")
        print(f"Testing: {name}")
        print(f"{'='*70}")
        
        per_sigma, summary_df = analyze_model_robustness(X_train, y_train, X_test, y_test, 
                                          model, name)
        results[name] = {'per_sigma': per_sigma, 'summary': summary_df}
    
    # Summary comparison
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Model':<20} {'Baseline R²':<15} {'NSI':<15} {'Retention %':<15}")
    print("-"*70)
    
    for name, result_dict in results.items():
        summary = result_dict['summary'].iloc[0]
        print(f"{name:<20} {summary['baseline_r2']:<15.4f} "
              f"{summary['nsi_r2']:<15.4f} {summary['retention_pct_r2']:<15.2f}")
    
    print("="*70)
    
    # Save results
    import os
    os.makedirs('results', exist_ok=True)
    
    for name, result_dict in results.items():
        safe_name = name.lower().replace(' ', '_')
        result_dict['per_sigma'].to_csv(f'results/generic_{safe_name}_per_sigma.csv', index=False)
        result_dict['summary'].to_csv(f'results/generic_{safe_name}_summary.csv', index=False)
        print(f"✓ Saved {safe_name} results to results/")
    
    return results


# ============================================================================
# COMPARE NOISE STRATEGIES
# ============================================================================

def compare_strategies(X_train, y_train, X_test, y_test, model=None):
    """
    Compare different noise strategies on same model.
    """
    print("\n" + "="*70)
    print("COMPARING NOISE STRATEGIES")
    print("="*70)
    
    if model is None:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    
    strategies = ['legacy', 'quantile', 'hetero', 'outlier', 'valprop']
    results = {}
    
    for strategy in strategies:
        print(f"\n{strategy.upper()} strategy...")
        
        injector = NoiseInjector(strategy, random_state=42)
        
        # Calibrate
        sigma = calibrate_sigma(y_train, 0.1, strategy=strategy, random_state=42)
        print(f"  Calibrated σ = {sigma:.4f}")
        
        # Test
        predictions = {}
        for mult in [0.0, 1.0, 2.0]:
            s = sigma * mult
            y_noisy = injector.inject(y_train, s) if mult > 0 else y_train
            
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_train, y_noisy)
            predictions[s] = model_copy.predict(X_test)
        
        per_sigma, summary_df = calculate_noise_metrics(y_test, predictions)
        results[strategy] = {'per_sigma': per_sigma, 'summary': summary_df}
    
    # Summary
    print("\n" + "="*70)
    print("STRATEGY COMPARISON")
    print("="*70)
    print(f"\n{'Strategy':<15} {'Baseline R²':<15} {'NSI':<15} {'Retention %':<15}")
    print("-"*70)
    
    for strategy, result_dict in results.items():
        summary = result_dict['summary'].iloc[0]
        print(f"{strategy:<15} {summary['baseline_r2']:<15.4f} "
              f"{summary['nsi_r2']:<15.4f} {summary['retention_pct_r2']:<15.2f}")
    
    print("="*70)
    
    return results


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """
    Complete workflow for generic datasets.
    """
    print("="*70)
    print("GENERIC DATASET + NOISEINJECT WORKFLOW")
    print("="*70)
    
    # 1. Load data
    print("\n1. Loading data...")
    X, y = load_custom_data()
    
    # Alternative: Load from CSV
    # X, y = load_from_csv('your_data.csv', target_column='target')
    
    # 2. Train/test split
    print("\n2. Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 3. Analyze single model
    print("\n3. Single model analysis...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    per_sigma, summary_df = analyze_model_robustness(X_train, y_train, X_test, y_test, 
                                       model, "Random Forest")
    
    # 4. Compare models (optional)
    print("\n4. Comparing multiple models...")
    model_results = compare_models(X_train, y_train, X_test, y_test)
    
    # 5. Compare strategies (optional)
    print("\n5. Comparing noise strategies...")
    strategy_results = compare_strategies(X_train, y_train, X_test, y_test)
    
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
    print("\nResults saved to CSV files.")


if __name__ == "__main__":
    main()
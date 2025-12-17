"""
Example: Classification with NoiseInject
Cheminformatics classification task - Toxicity Prediction

This example shows:
1. Loading Tox21 dataset (molecular toxicity classification)
2. Converting SMILES to ECFP fingerprints
3. Testing classification robustness with NoiseInject
4. Comparing noise strategies for classification
5. Per-class robustness analysis
"""

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import deepchem as dc
import os

from noiseInject import (
    NoiseInjectorClassification,
    calibrate_flip_probability,
    calculate_classification_metrics,
    get_most_robust_classes,
    get_least_robust_classes
)

RDLogger.DisableLog('rdApp.*')
os.environ['OMP_NUM_THREADS'] = '1'


# ============================================================================
# DATA LOADING
# ============================================================================

def load_tox21_sr_p53():
    """
    Load Tox21 SR-p53 task (binary classification).
    Predicts p53 pathway activation (toxicity indicator).
    """
    print("Loading Tox21 SR-p53 dataset...")
    tasks, datasets, transformers = dc.molnet.load_tox21(
        featurizer='Raw',
        splitter='scaffold'
    )
    
    train_dataset, val_dataset, test_dataset = datasets
    
    # SR-p53 is task index 5
    task_idx = 5
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Task: {tasks[task_idx]}")
    
    return train_dataset, val_dataset, test_dataset, task_idx


def smiles_to_ecfp(smiles, radius=2, n_bits=1024):
    """Convert SMILES to ECFP fingerprint"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def deepchem_to_ecfp(dc_dataset, task_idx):
    """Convert DeepChem dataset to ECFP features"""
    X_list = []
    y_list = []
    
    for i in range(len(dc_dataset)):
        smiles = dc_dataset.ids[i]
        y = dc_dataset.y[i, task_idx]
        
        # Skip NaN labels
        if np.isnan(y):
            continue
        
        fp = smiles_to_ecfp(smiles)
        if fp is not None:
            X_list.append(fp)
            y_list.append(int(y))
    
    X = np.vstack(X_list)
    y = np.array(y_list)
    
    return X, y


# ============================================================================
# ROBUSTNESS ANALYSIS
# ============================================================================

def test_classification_robustness(X_train, y_train, X_test, y_test, strategy='uniform'):
    """Test classification robustness with NoiseInject"""
    
    print(f"\n{'='*70}")
    print(f"TESTING {strategy.upper()} STRATEGY")
    print(f"{'='*70}")
    
    # Class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\nClass distribution (train):")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} ({count/len(y_train)*100:.1f}%)")
    
    # Calibrate flip probability
    print("\nCalibrating flip probability for 10% flip rate...")
    injector = NoiseInjectorClassification(strategy=strategy, random_state=42)
    flip_prob_cal = calibrate_flip_probability(
        y_train, target_flip_rate=0.1, strategy=strategy, random_state=42
    )
    print(f"Calibrated flip_prob: {flip_prob_cal:.4f}")
    
    # Test at multiple noise levels
    print("\nTesting at multiple noise levels...")
    predictions = {}
    flip_multipliers = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    for mult in flip_multipliers:
        fp = flip_prob_cal * mult
        
        if mult == 0.0:
            y_train_noisy = y_train
        else:
            y_train_noisy = injector.inject(y_train, fp)
            actual_flip = injector.get_effective_flip_rate(y_train, y_train_noisy)
            print(f"  flip_prob={fp:.4f} (×{mult}): actual flip rate={actual_flip:.4f}")
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        model.fit(X_train, y_train_noisy)
        
        # Predict on clean test set
        predictions[fp] = model.predict(X_test)
        
        # Show accuracy
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, predictions[fp])
        print(f"    → Test accuracy: {acc:.4f}")
    
    # Calculate metrics
    print("\nCalculating robustness metrics...")
    per_flip_df, summary_df, per_class_df = calculate_classification_metrics(
        y_test, predictions
    )
    
    # Display results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    
    print("\nOverall Performance:")
    print(per_flip_df[['flip_prob', 'accuracy', 'f1_macro', 'f1_weighted']].to_string(index=False))
    
    summary = summary_df.iloc[0]
    print(f"\nSummary Metrics:")
    print(f"  Baseline Accuracy: {summary['baseline_accuracy']:.4f}")
    print(f"  NSI (Accuracy):    {summary['nsi_accuracy']:.4f}")
    print(f"  Retention:         {summary['retention_pct_accuracy']:.2f}%")
    
    print(f"\n  Baseline F1:       {summary['baseline_f1_macro']:.4f}")
    print(f"  NSI (F1):          {summary['nsi_f1_macro']:.4f}")
    print(f"  Retention (F1):    {summary['retention_pct_f1_macro']:.2f}%")
    
    # Per-class robustness
    print("\nPer-Class Robustness:")
    for cls in unique:
        nsi_key = f'nsi_f1_class_{cls}'
        baseline_key = f'baseline_f1_class_{cls}'
        retention_key = f'retention_pct_f1_class_{cls}'
        
        if nsi_key in summary_df.columns:
            print(f"  Class {cls}:")
            print(f"    Baseline F1: {summary[baseline_key]:.4f}")
            print(f"    NSI:         {summary[nsi_key]:.4f}")
            print(f"    Retention:   {summary[retention_key]:.2f}%")
    
    return per_flip_df, summary_df, per_class_df


def compare_classification_strategies(X_train, y_train, X_test, y_test):
    """Compare multiple classification noise strategies"""
    
    print(f"\n{'='*70}")
    print("COMPARING CLASSIFICATION STRATEGIES")
    print(f"{'='*70}")
    
    strategies = ['uniform', 'class_imbalance', 'instance_noise', 'class_dependent']
    results = {}
    
    for strategy in strategies:
        print(f"\n--- {strategy.upper()} ---")
        
        injector = NoiseInjectorClassification(strategy, random_state=42)
        
        # Calibrate for fair comparison
        flip_prob = calibrate_flip_probability(
            y_train, target_flip_rate=0.1, strategy=strategy, random_state=42
        )
        print(f"Calibrated flip_prob: {flip_prob:.4f}")
        
        # Test at 3 levels
        predictions = {}
        for mult in [0.0, 1.0, 2.0]:
            fp = flip_prob * mult
            y_noisy = y_train if mult == 0.0 else injector.inject(y_train, fp)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
            model.fit(X_train, y_noisy)
            predictions[fp] = model.predict(X_test)
        
        per_flip_df, summary_df, per_class_df = calculate_classification_metrics(
            y_test, predictions
        )
        results[strategy] = {
            'per_flip': per_flip_df,
            'summary': summary_df,
            'per_class': per_class_df
        }
    
    # Comparison table
    print(f"\n{'='*70}")
    print("STRATEGY COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Strategy':<20} {'Baseline Acc':<15} {'NSI':<15} {'Retention %':<15}")
    print("-" * 70)
    
    for strategy, result_dict in results.items():
        summary = result_dict['summary'].iloc[0]
        print(f"{strategy:<20} {summary['baseline_accuracy']:<15.4f} "
              f"{summary['nsi_accuracy']:<15.4f} {summary['retention_pct_accuracy']:<15.2f}")
    
    print("=" * 70)
    
    # Most/least robust classes
    print("\nClass Robustness Analysis:")
    for strategy, result_dict in results.items():
        print(f"\n{strategy.upper()}:")
        
        robust = get_most_robust_classes(result_dict['summary'], n=1)
        fragile = get_least_robust_classes(result_dict['summary'], n=1)
        
        if robust:
            print(f"  Most robust:  Class {robust[0][0]} (NSI={robust[0][1]:.4f})")
        if fragile:
            print(f"  Least robust: Class {fragile[0][0]} (NSI={fragile[0][1]:.4f})")
    
    return results


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Complete classification workflow with NoiseInject"""
    
    print("="*70)
    print("TOX21 CLASSIFICATION ROBUSTNESS WORKFLOW")
    print("="*70)
    
    # Load data
    train_ds, val_ds, test_ds, task_idx = load_tox21_sr_p53()
    
    # Convert to ECFP
    print("\nConverting molecules to ECFP fingerprints...")
    X_train, y_train = deepchem_to_ecfp(train_ds, task_idx)
    X_test, y_test = deepchem_to_ecfp(test_ds, task_idx)
    
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    print(f"Class balance (train): {np.bincount(y_train)}")
    print(f"Class balance (test):  {np.bincount(y_test)}")
    
    # Test single strategy
    per_flip_df, summary_df, per_class_df = test_classification_robustness(
        X_train, y_train, X_test, y_test, strategy='uniform'
    )
    
    # Compare strategies
    results = compare_classification_strategies(X_train, y_train, X_test, y_test)
    
    # Save results
    print("\nSaving results...")
    os.makedirs('results', exist_ok=True)
    
    for strategy, result_dict in results.items():
        result_dict['per_flip'].to_csv(f'results/tox21_{strategy}_per_flip.csv', index=False)
        result_dict['summary'].to_csv(f'results/tox21_{strategy}_summary.csv', index=False)
        result_dict['per_class'].to_csv(f'results/tox21_{strategy}_per_class.csv', index=False)
        print(f"✓ Saved {strategy} results")
    
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
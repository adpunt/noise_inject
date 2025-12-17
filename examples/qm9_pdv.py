"""
Example Workflow: QM9 Dataset with PDV Representation
Using NoiseInject framework to test model robustness

This example shows:
1. Loading QM9 dataset
2. Converting SMILES to PDV (physicochemical descriptors)
3. Testing noise robustness with NoiseInject
4. Comparing results across noise levels
"""

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import QM9
import torch
import os.path as osp

from noiseInject import NoiseInjectorRegression, calculate_noise_metrics, calibrate_sigma

# Disable multiprocessing to avoid segfaults
torch.set_num_threads(1)
import os
os.environ['OMP_NUM_THREADS'] = '1'

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')


# ============================================================================
# DESCRIPTOR LIST (from your original code)
# ============================================================================

DEFAULT_DESCRIPTOR_LIST = [
    'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v',
    'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'EState_VSA1', 'EState_VSA10',
    'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6',
    'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2',
    'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount', 'HeavyAtomMolWt',
    'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'MaxAbsEStateIndex', 'MaxAbsPartialCharge',
    'MaxEStateIndex', 'MaxPartialCharge', 'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex',
    'MinPartialCharge', 'MolLogP', 'MolMR', 'MolWt', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles',
    'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
    'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRadicalElectrons',
    'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings',
    'NumValenceElectrons', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13',
    'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7',
    'PEOE_VSA8', 'PEOE_VSA9', 'RingCount', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4',
    'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11',
    'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7',
    'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3',
    'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'fr_Al_COO',
    'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO',
    'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2',
    'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde',
    'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline',
    'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic',
    'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido',
    'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan',
    'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile',
    'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime',
    'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester',
    'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide',
    'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan',
    'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'qed'
]


# ============================================================================
# DATA LOADING AND PROCESSING
# ============================================================================

def smiles_to_pdv(smiles_string):
    """
    Convert SMILES string to PDV (physicochemical descriptor vector).
    
    Args:
        smiles_string: SMILES representation of molecule
    
    Returns:
        numpy array of 200 descriptors (or None if conversion fails)
    """
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return None
        
        calculator = MolecularDescriptorCalculator(DEFAULT_DESCRIPTOR_LIST)
        descriptors = calculator.CalcDescriptors(mol)
        return np.array(descriptors)
    except:
        return None


def load_qm9_with_pdv(target_property='homo_lumo_gap', sample_size=5000, random_seed=42):
    """
    Load QM9 dataset and convert to PDV representation.
    
    Args:
        target_property: Property to predict (default: homo_lumo_gap)
        sample_size: Number of molecules to use
        random_seed: Random seed for reproducibility
    
    Returns:
        X_pdv: PDV feature matrix (N x 200)
        y: Target values (N,)
        smiles_list: List of SMILES strings
    """
    # Property mapping
    properties = {
        'homo_lumo_gap': 4, 'alpha': 1, 'G': 10, 'H': 9, 'U': 8,
        'G_a': 15, 'H_a': 14, 'U_a': 13, 'mu': 0, 'A': 16, 'B': 17, 'C': 18
    }
    
    print(f"Loading QM9 dataset (target: {target_property})...")
    
    # Load QM9
    qm9 = QM9(root=osp.join('.', 'data', 'QM9'))
    
    # Filter valid indices (molecules that RDKit can process)
    # You may need to create this file or skip this filtering step
    valid_indices_path = osp.join('.', 'data', 'valid_qm9_indices.pth')
    if osp.exists(valid_indices_path):
        valid_indices = torch.load(valid_indices_path)
        qm9 = qm9.index_select(valid_indices)
        print(f"Filtered to {len(qm9)} valid molecules")
    
    # Extract target property BEFORE sampling
    property_idx = properties[target_property]
    y_full = qm9.data.y[:, property_idx].numpy()
    
    # Shuffle and sample
    torch.manual_seed(random_seed)
    indices = torch.randperm(len(qm9))[:sample_size]
    qm9_sample = qm9.index_select(indices)
    y_sample = y_full[indices.numpy()]
    
    print(f"Converting {len(qm9_sample)} molecules to PDV...")
    
    # Convert SMILES to PDV
    X_pdv_list = []
    y_valid = []
    smiles_valid = []
    
    for i, data in enumerate(qm9_sample):
        smiles = data.smiles
        pdv = smiles_to_pdv(smiles)
        
        if pdv is not None:
            X_pdv_list.append(pdv)
            y_valid.append(y_sample[i])
            smiles_valid.append(smiles)
        
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(qm9_sample)} molecules...")
    
    X_pdv = np.vstack(X_pdv_list)
    y = np.array(y_valid)
    
    print(f"Successfully converted {len(y)} molecules")
    print(f"Feature matrix shape: {X_pdv.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
    
    return X_pdv, y, smiles_valid


# ============================================================================
# NOISE ROBUSTNESS WORKFLOW
# ============================================================================

def test_noise_robustness(X_train, y_train, X_test, y_test, 
                         strategy='legacy', model_name='XGBoost'):
    """
    Test model robustness to label noise using NoiseInject.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data (clean labels)
        strategy: Noise strategy to use
        model_name: Name for display
    
    Returns:
        metrics_df: DataFrame with robustness metrics
    """
    print(f"\n{'='*70}")
    print(f"Testing {model_name} with {strategy.upper()} noise strategy")
    print(f"{'='*70}")
    
    # Step 1: Calibrate sigma for fair comparison
    print("\n1. Calibrating sigma for 10% effective noise...")
    injector = NoiseInjectorRegression(strategy=strategy, random_state=42)
    sigma_cal = calibrate_sigma(y_train, target_effective_noise=0.1, 
                                strategy=strategy, random_state=42)
    print(f"   Calibrated sigma: {sigma_cal:.4f}")
    
    # Step 2: Test at multiple noise levels
    print("\n2. Testing at multiple noise levels...")
    sigma_levels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]  # Higher multipliers to see degradation
    
    predictions = {}
    
    for mult in sigma_levels:
        sigma = sigma_cal * mult
        print(f"   Testing sigma = {sigma:.4f} (mult = {mult})...")
        
        # Inject noise (or use clean labels for baseline)
        if mult == 0.0:
            y_train_noisy = y_train
        else:
            y_train_noisy = injector.inject(y_train, sigma)
        
        # Train model
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=100, random_state=42, n_jobs=1)
        model.fit(X_train, y_train_noisy)
        
        # Predict on clean test set
        predictions[sigma] = model.predict(X_test)
    
    # Step 3: Calculate metrics
    print("\n3. Calculating robustness metrics...")
    per_sigma_df, summary_df = calculate_noise_metrics(
        y_test, 
        predictions,
        metrics=['r2', 'rmse', 'mae']
    )
    
    # Display results
    print("\nPer-sigma performance:")
    print(per_sigma_df.to_string(index=False))
    
    print("\nSummary statistics:")
    summary = summary_df.iloc[0]
    print(f"  Baseline R²:     {summary['baseline_r2']:.4f}")
    print(f"  NSI (R²):        {summary['nsi_r2']:.4f} (p={summary['nsi_r2_pval']:.4f})")
    print(f"  Retention:       {summary['retention_pct_r2']:.2f}%")
    
    return per_sigma_df, summary_df


def compare_strategies(X_train, y_train, X_test, y_test):
    """
    Compare multiple noise strategies.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
    
    Returns:
        results: Dictionary of results for each strategy
    """
    strategies = ['legacy', 'quantile', 'hetero', 'outlier']
    results = {}
    
    for strategy in strategies:
        per_sigma_df, summary_df = test_noise_robustness(
            X_train, y_train, X_test, y_test,
            strategy=strategy
        )
        results[strategy] = {'per_sigma': per_sigma_df, 'summary': summary_df}
    
    # Print comparison
    print(f"\n{'='*70}")
    print("STRATEGY COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Strategy':<15} {'Baseline R²':<15} {'NSI (R²)':<15} {'Retention %':<15}")
    print("-" * 70)
    
    for strategy, result_dict in results.items():
        summary = result_dict['summary'].iloc[0]
        baseline_r2 = summary['baseline_r2']
        nsi_r2 = summary['nsi_r2']
        retention = summary['retention_pct_r2']
        
        print(f"{strategy:<15} {baseline_r2:<15.4f} {nsi_r2:<15.4f} {retention:<15.2f}")
    
    return results


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """
    Complete workflow: QM9 → PDV → NoiseInject → Robustness Analysis
    """
    print("="*70)
    print("QM9 NOISE ROBUSTNESS WORKFLOW")
    print("="*70)
    
    # Step 1: Load data and convert to PDV
    X, y, smiles = load_qm9_with_pdv(
        target_property='homo_lumo_gap',
        sample_size=10000,  # Increased for more robust analysis
        random_seed=42
    )
    
    # Step 2: Train/test split
    print("\nSplitting data (80/20 train/test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    
    # Step 3: Test single strategy in detail
    print("\n" + "="*70)
    print("DETAILED ANALYSIS - LEGACY STRATEGY")
    print("="*70)
    
    per_sigma_df, summary_df = test_noise_robustness(
        X_train, y_train, X_test, y_test,
        strategy='legacy'
    )
    
    # Already printed in function, no need to print again
    
    # Step 4: Compare strategies
    print("\n" + "="*70)
    print("COMPARING MULTIPLE STRATEGIES")
    print("="*70)
    
    results = compare_strategies(X_train, y_train, X_test, y_test)
    
    # Step 5: Save results
    print("\nSaving results to CSV...")
    
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    for strategy, result_dict in results.items():
        result_dict['per_sigma'].to_csv(f"results/qm9_pdv_{strategy}_per_sigma.csv", index=False)
        result_dict['summary'].to_csv(f"results/qm9_pdv_{strategy}_summary.csv", index=False)
        print(f"  Saved {strategy} results to results/")
    
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
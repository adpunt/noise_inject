"""
Example: MoleculeNet with GNN Encoder + NoiseInject
Using pre-trained molecular representations for noise robustness testing

This example shows:
1. Loading MoleculeNet dataset (ESOL - solubility prediction)
2. Converting molecules to graph representations
3. Using GNN to generate embeddings
4. Testing noise robustness with NoiseInject
5. Comparing different noise strategies
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from rdkit import Chem, RDLogger
import deepchem as dc

from noiseInject import NoiseInjectorRegression, calculate_noise_metrics, calibrate_sigma

# Disable warnings and multiprocessing issues
RDLogger.DisableLog('rdApp.*')
torch.set_num_threads(1)
import os
os.environ['OMP_NUM_THREADS'] = '1'


# ============================================================================
# SIMPLE GNN ENCODER
# ============================================================================

class SimpleGNNEncoder(torch.nn.Module):
    """
    Simple 2-layer GCN encoder for molecular graphs.
    Pre-train or use as feature extractor.
    """
    def __init__(self, num_node_features=9, hidden_dim=64, embedding_dim=32):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        
        # Global pooling (mean of all node embeddings)
        x = global_mean_pool(x, batch)
        
        return x


# ============================================================================
# DATA LOADING
# ============================================================================

def load_moleculenet_esol():
    """
    Load ESOL dataset from MoleculeNet via DeepChem.
    ESOL: Water solubility (regression, ~1100 molecules)
    """
    print("Loading ESOL dataset from MoleculeNet...")
    tasks, datasets, transformers = dc.molnet.load_delaney(
        featurizer='Raw',
        splitter='scaffold'
    )
    
    train_dataset, val_dataset, test_dataset = datasets
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


def smiles_to_graph(smiles):
    """
    Convert SMILES to PyTorch Geometric graph.
    
    Node features: [atomic_num, degree, formal_charge, num_hs, 
                    hybridization, is_aromatic, is_in_ring, chirality, mass]
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Node features
    node_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            atom.GetHybridization().real,
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            int(atom.HasProp('_ChiralityPossible')),
            atom.GetMass()
        ]
        node_features.append(features)
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Edge indices
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])  # Undirected
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    
    return Data(x=x, edge_index=edge_index)


def deepchem_to_graphs(dc_dataset):
    """Convert DeepChem dataset to list of PyG graphs with labels"""
    graphs = []
    labels = []
    
    for i in range(len(dc_dataset)):
        smiles = dc_dataset.ids[i]
        y = dc_dataset.y[i, 0]  # First task
        
        if np.isnan(y):
            continue
            
        graph = smiles_to_graph(smiles)
        if graph is not None:
            graphs.append(graph)
            labels.append(y)
    
    return graphs, np.array(labels)


# ============================================================================
# GENERATE GNN EMBEDDINGS
# ============================================================================

def extract_gnn_embeddings(graphs, model=None, batch_size=32):
    """
    Extract embeddings using GNN encoder.
    
    Args:
        graphs: List of PyG Data objects
        model: Pre-trained GNN or None (will create new one)
        batch_size: Batch size for embedding extraction
    
    Returns:
        embeddings: numpy array (N x embedding_dim)
    """
    if model is None:
        # Create simple GNN
        model = SimpleGNNEncoder(num_node_features=9, hidden_dim=64, embedding_dim=32)
    
    model.eval()
    
    # Create DataLoader
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            emb = model(batch)
            embeddings.append(emb.numpy())
    
    return np.vstack(embeddings)


# ============================================================================
# WORKFLOW
# ============================================================================

def test_moleculenet_robustness():
    """
    Complete workflow: MoleculeNet → GNN embeddings → NoiseInject
    """
    print("="*70)
    print("MOLECULENET + GNN ENCODER + NOISEINJECT WORKFLOW")
    print("="*70)
    
    # 1. Load data
    train_ds, val_ds, test_ds = load_moleculenet_esol()
    
    # 2. Convert to graphs
    print("\nConverting molecules to graphs...")
    train_graphs, y_train = deepchem_to_graphs(train_ds)
    test_graphs, y_test = deepchem_to_graphs(test_ds)
    
    print(f"Train graphs: {len(train_graphs)}, Test graphs: {len(test_graphs)}")
    
    # 3. Extract GNN embeddings
    print("\nExtracting GNN embeddings...")
    gnn_model = SimpleGNNEncoder(num_node_features=9, hidden_dim=64, embedding_dim=32)
    
    X_train = extract_gnn_embeddings(train_graphs, gnn_model)
    X_test = extract_gnn_embeddings(test_graphs, gnn_model)
    
    print(f"Embedding shape: {X_train.shape}")
    print(f"Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    
    # 4. Test noise robustness
    print("\n" + "="*70)
    print("TESTING NOISE ROBUSTNESS")
    print("="*70)
    
    # Calibrate sigma for 10% effective noise
    print("\nCalibrating sigma...")
    injector = NoiseInjectorRegression('legacy', random_state=42)
    sigma_cal = calibrate_sigma(y_train, target_effective_noise=0.1, random_state=42)
    print(f"Calibrated sigma: {sigma_cal:.4f}")
    
    # Test at multiple levels
    print("\nTesting at multiple noise levels...")
    predictions = {}
    sigma_levels = [0.0, 1.0, 2.0, 3.0, 4.0]  # Higher multipliers
    
    for mult in sigma_levels:
        sigma = sigma_cal * mult
        print(f"  σ = {sigma:.4f} (mult={mult})...")
        
        # Inject noise
        if mult == 0.0:
            y_train_noisy = y_train
        else:
            y_train_noisy = injector.inject(y_train, sigma)
        
        # Train model (n_jobs=1 to avoid segfault)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
        model.fit(X_train, y_train_noisy)
        
        # Predict
        predictions[sigma] = model.predict(X_test)
    
    # 5. Calculate metrics
    print("\nCalculating robustness metrics...")
    per_sigma, summary_df = calculate_noise_metrics(y_test, predictions)
    
    # 6. Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print("\nPer-sigma performance:")
    print(per_sigma.to_string(index=False))
    
    summary = summary_df.iloc[0]
    print(f"\n{'='*70}")
    print("SUMMARY METRICS")
    print(f"{'='*70}")
    print(f"Baseline R²:     {summary['baseline_r2']:.4f}")
    print(f"NSI (R²):        {summary['nsi_r2']:.4f} (p={summary['nsi_r2_pval']:.4f})")
    print(f"Retention:       {summary['retention_pct_r2']:.2f}%")
    print(f"{'='*70}")
    
    # Save results
    import os
    os.makedirs('results', exist_ok=True)
    
    per_sigma.to_csv('results/moleculenet_esol_gnn_per_sigma.csv', index=False)
    summary_df.to_csv('results/moleculenet_esol_gnn_summary.csv', index=False)
    print("\n✓ Results saved to results/ directory")


def compare_noise_strategies():
    """
    Compare different noise strategies on MoleculeNet
    """
    print("\n" + "="*70)
    print("COMPARING NOISE STRATEGIES")
    print("="*70)
    
    # Load and prepare data (shortened for comparison)
    train_ds, _, test_ds = load_moleculenet_esol()
    
    train_graphs, y_train = deepchem_to_graphs(train_ds)
    test_graphs, y_test = deepchem_to_graphs(test_ds)
    
    gnn_model = SimpleGNNEncoder()
    X_train = extract_gnn_embeddings(train_graphs, gnn_model)
    X_test = extract_gnn_embeddings(test_graphs, gnn_model)
    
    # Test strategies
    strategies = ['legacy', 'hetero', 'quantile', 'outlier']
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting {strategy.upper()} strategy...")
        injector = NoiseInjectorRegression(strategy, random_state=42)
        
        # Calibrate
        sigma = calibrate_sigma(y_train, 0.1, strategy=strategy, random_state=42)
        
        # Test
        predictions = {}
        for mult in [0.0, 1.0, 2.0]:
            s = sigma * mult
            y_noisy = injector.inject(y_train, s) if mult > 0 else y_train
            
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
            model.fit(X_train, y_noisy)
            predictions[s] = model.predict(X_test)
        
        per_sigma, summary_df = calculate_noise_metrics(y_test, predictions)
        results[strategy] = {'per_sigma': per_sigma, 'summary': summary_df}
    
    # Compare
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


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run main workflow
    test_moleculenet_robustness()
    
    # Optional: compare strategies
    compare_noise_strategies()
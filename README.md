# NoiseInject

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)
[![DOI](https://zenodo.org/badge/1117122976.svg)](https://doi.org/10.5281/zenodo.20532710)

A lightweight Python framework for testing ML model robustness to label noise in both **regression** and **classification** tasks.

## Installation
```bash
pip install -e .
```

## Quick Start

### Regression
```python
from noiseInject import NoiseInjectorRegression, calibrate_sigma, calculate_noise_metrics
from sklearn.ensemble import RandomForestRegressor

# Calibrate sigma for target effective noise
sigma = calibrate_sigma(y_train, target_effective_noise=0.1)

# Test at multiple noise levels
injector = NoiseInjectorRegression('legacy', random_state=42)
predictions = {}

for mult in [0.0, 1.0, 2.0]:
    y_noisy = injector.inject(y_train, sigma * mult) if mult > 0 else y_train
    model = RandomForestRegressor()
    model.fit(X_train, y_noisy)
    predictions[sigma * mult] = model.predict(X_test)

# Analyze robustness
per_sigma_df, summary_df = calculate_noise_metrics(y_test, predictions)
print(f"NSI: {summary_df['nsi_r2'].values[0]:.4f}")
print(f"Retention: {summary_df['retention_pct_r2'].values[0]:.1f}%")
```

### Classification
```python
from noiseInject import NoiseInjectorClassification, calibrate_flip_probability, calculate_classification_metrics
from sklearn.ensemble import RandomForestClassifier

# Calibrate flip probability for target flip rate
flip_prob = calibrate_flip_probability(y_train, target_flip_rate=0.1)

# Test at multiple noise levels
injector = NoiseInjectorClassification('uniform', random_state=42)
predictions = {}

for mult in [0.0, 1.0, 2.0]:
    y_noisy = injector.inject(y_train, flip_prob * mult) if mult > 0 else y_train
    model = RandomForestClassifier()
    model.fit(X_train, y_noisy)
    predictions[flip_prob * mult] = model.predict(X_test)

# Analyze robustness
per_flip_df, summary_df, per_class_df = calculate_classification_metrics(y_test, predictions)
print(f"NSI: {summary_df['nsi_accuracy'].values[0]:.4f}")
print(f"Retention: {summary_df['retention_pct_accuracy'].values[0]:.1f}%")
```

## Strategies

### Regression (continuous noise)
- **legacy**: Homogeneous Gaussian noise
- **quantile**: More noise at distribution extremes
- **threshold**: More noise above/below thresholds
- **outlier**: More noise for z-score outliers
- **hetero**: Heteroscedastic (variance ∝ |y|)
- **valprop**: Value-proportional noise

### Classification (label flips)
- **uniform**: Equal flip probability for all
- **class_imbalance**: Varies by class frequency
- **binary_asymmetric**: Asymmetric binary flips
- **instance_noise**: Random per-sample variation
- **class_dependent**: Each class has own flip rate
- **confusion_directed**: Realistic confusion patterns

## Key Metrics

- **NSI (Noise Sensitivity Index)**: Slope of performance vs noise. More negative = less robust.
- **Retention**: Performance preservation at high noise. Higher = more robust.
- **Baseline**: Performance with clean labels (σ=0 or flip_prob=0).

## Uncertainty Quantification

For probabilistic models, `calculate_uncertainty_metrics` tracks how well predicted
uncertainty stays calibrated across noise levels: uncertainty–error and uncertainty–noise
Spearman correlations, coverage at 1σ/2σ, ECE, and mean interval width.

```python
from noiseInject import calculate_uncertainty_metrics

# predictions[sigma] -> mean,  uncertainties[sigma] -> per-sample predicted std
per_sigma_df, summary_df = calculate_uncertainty_metrics(y_test, predictions, uncertainties)
```

Optional model wrappers that *produce* per-sample uncertainty (install via
`pip install noiseInject[uncertainty]`):

- `SplitConformalRegressor` - distribution-free intervals around any sklearn model (no extra deps)
- `GaucheGPRegressor` - Gaussian process with Tanimoto/RBF kernels (gpytorch + gauche)
- `MCDropoutRegressor` - Monte Carlo dropout for any PyTorch network

See `notebooks/02_uncertainty.ipynb` for a worked example.

## Features

✓ Model-agnostic (works with any sklearn-compatible model)  
✓ Calibration for fair strategy comparison  
✓ Per-class robustness analysis (classification)  
✓ Backward compatible (NoiseInjector alias maintained)  
✓ Minimal dependencies (numpy, pandas, scipy, sklearn)

## Examples

Runnable notebooks in `notebooks/`:
- `01_quickstart.ipynb` - Regression + classification end-to-end (public sklearn data, no extra deps)
- `02_uncertainty.ipynb` - Tracking uncertainty calibration under noise with NGBoost (ECE, coverage, uncertainty–error/noise correlation)

Scripts in `examples/`:
- `generic_dataset.py` - Any regression dataset
- `qm9_pdv.py` - QM9 molecular properties
- `moleculenet_gnn.py` - MoleculeNet with GNN embeddings
- `tox21_classification.py` - Molecular toxicity classification

## Citation

If you use NoiseInject in your research, please cite the paper and the archived software:

```bibtex
@article{punt2026noiseinject,
  title   = {NoiseInject: Systematic evaluation of model robustness to label noise in cheminformatics},
  author  = {Punt, Adelaide},
  journal = {Journal of Cheminformatics},
  year    = {2026},
  note    = {Software: https://github.com/adpunt/noise_inject}
}

@software{punt_noiseinject_software,
  title     = {NoiseInject: A Framework for Label Noise Robustness Testing},
  author    = {Punt, Adelaide},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v0.3.0},
  doi       = {10.5281/zenodo.20532710},
  url       = {https://doi.org/10.5281/zenodo.20532710}
}
```

## License

MIT
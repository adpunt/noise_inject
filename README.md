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
from noiseInject import NoiseInjectorRegression, calculate_noise_metrics
from sklearn.ensemble import RandomForestRegressor

# The noise level is the amount of noise you want DELIVERED -- the
# root-mean-square error added, in the label's own units. Each condition solves
# for its own internal scale to hit it. There is no calibration step.
sd = y_train.std()
predictions = {}

for k in [0.0, 0.2, 0.3, 0.5, 0.75, 1.0]:
    injector = NoiseInjectorRegression.from_condition('gaussian', random_state=42)
    result = injector.inject_verbose(y_train, dose=k * sd)
    model = RandomForestRegressor().fit(X_train, result.y_noisy)
    predictions[k] = model.predict(X_test)
    print(result.as_row())      # what was actually delivered, for the record

# Analyze robustness
per_sigma_df, summary_df = calculate_noise_metrics(y_test, predictions)
s = summary_df.iloc[0]
print(f"auc_norm (R2):  {s['auc_norm_r2']:.4f}   (higher = more robust)")
print(f"Weibull beta:   {s['weibull_beta_r2']:.2f}   (>1 holds-then-cliff, <1 early collapse)")
print(f"Retention:      {s['retention_pct_r2']:.1f}%")
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

for mult in [0.0, 1.0, 2.0, 3.0]:
    y_noisy = injector.inject(y_train, flip_prob * mult) if mult > 0 else y_train
    model = RandomForestClassifier()
    model.fit(X_train, y_noisy)
    predictions[flip_prob * mult] = model.predict(X_test)

# Analyze robustness
per_flip_df, summary_df, per_class_df = calculate_classification_metrics(y_test, predictions)
print(f"auc_norm (accuracy): {summary_df['auc_norm_accuracy'].values[0]:.4f}")
print(f"Retention:           {summary_df['retention_pct_accuracy'].values[0]:.1f}%")
```

## Strategies

### Regression (continuous noise)

**Every condition delivers the same amount of noise at a given level.** That is the
point: comparing conditions at a common nominal *setting* compares amount, not
shape. The six strategies this replaced delivered between 0.49× and 2.00× the same
amount at one setting, and their entire apparent severity ordering was explained by
that. Four of them also assumed error depends on the measured value, which has been
directly tested on 16,844 repeat measurements and refuted.

Shape and targeting are chosen separately.

*Shape* (`distribution`) — how each draw is distributed:
- **gaussian**: the reference case
- **student_t**: heavy-tailed, `nu > 2`. Gaussian is its `nu → ∞` limit, so the two nest
- **laplace**: the shape actually fitted to real bioactivity measurement disagreements

*Targeting* (`strategy`) — who gets hit, and how hard:
- **uniform**: every record gets the same scale
- **grouped_wider**: records in some groups get a wider error, still centred on the truth
- **grouped_shifted**: whole groups are pushed in one direction by a constant
- **outlier**: a randomly chosen fraction get a wider error. Selection is *random*, not
  by value — a mistyped unit is a property of the record, not of the value
- **censoring**: values past a limit are recorded as the limit. The only condition that
  is neither zero-mean nor dose-matched, so it takes a censored *fraction* rather than a dose

`CONDITIONS` names each combination the study runs (`gaussian`, `student_t_nu5`,
`grouped_shifted`, `outlier_p05`, `censoring_25`, …); `from_condition` builds one.
`inject_verbose` returns the noisy labels together with the noise actually drawn per
record and the provenance — unit dose, solved scale, delivered dose, affected fraction —
so no downstream figure is ever untraceable to the amount of noise that produced it.

It also **checks** the delivered amount rather than only recording it, and raises
`DoseError` when a draw lands outside the band `dose_tolerance` allows for that
condition. The band is three standard errors derived from the draw's own fourth
moment and its effective size, so a heavy tail and a group-level term widen it on
their own. Two consequences worth knowing before you catch it:

- `DoseError` subclasses `RuntimeError`, but it has its own class on purpose. Callers
  that wrap work in `except Exception` and carry on will otherwise turn a wrong
  injection into a missing result rather than a failure. Re-raise it.
- Three legitimate draws in a thousand fall outside a three-sigma band by chance. On
  a few hundred records with a heavy tail that rate is higher, so on small datasets
  decide deliberately whether a missed draw should stop the run.

Censoring is exempt: it is swept on the fraction of labels clipped and has no target
amount to check against.

### Classification (label flips)
- **uniform**: Equal flip probability for all
- **class_imbalance**: Varies by class frequency
- **binary_asymmetric**: Asymmetric binary flips
- **instance_noise**: Random per-sample variation
- **class_dependent**: Each class has own flip rate
- **confusion_directed**: Realistic confusion patterns

## Key Metrics

Robustness is read off the **retention curve** `ret(x) = metric(x) / metric(0)` (higher-is-better
skill metrics only: R² for regression; accuracy/F1/precision/recall for classification):

- **auc_norm** (primary): normalised area under the retention curve — the mean fraction of
  baseline skill retained across the noise sweep. Higher = more robust (~[0, 1]), decoupled from
  baseline performance. Replaces the old NSI slope, which mischaracterised a nonlinear
  degradation curve and was coupled to baseline skill.
- **Weibull β / τ** (supplementary): fit `ret ≈ exp(-(x/τ)^β)`; β is the *shape* of failure
  (β>1 = holds-then-cliff, β<1 = early collapse then plateau), τ the decay scale. Only meaningful
  once the model genuinely degrades, and needs ≥4 noise levels to fit.
- **Retention**: Performance preservation at the highest noise level. Higher = more robust.
- **Baseline**: Performance with clean labels (σ=0 or flip_prob=0).

`curve_stable_*` flags a single curve that dipped below 0 retention (a collapsed retrain makes
auc_norm untrustworthy). Curve metrics are most reliable when `predictions` is built from
**seed-averaged** per-noise-level performance; pass `baseline_threshold=` (e.g. 0.3) to NaN-out
models too weak to have meaningful degradation.

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
✓ **Dose-matched by construction** — every regression condition delivers the noise
  level you asked for, so a comparison between conditions measures shape, not amount  
✓ **Traceable** — `inject_verbose` returns the noise actually drawn per record and the
  provenance of the run, so no downstream figure is untraceable to what produced it  
✓ Calibration for fair comparison of classification flip rates  
✓ Per-class robustness analysis (classification)  
✓ Minimal dependencies (numpy, pandas, scipy, sklearn)

**1.0.0 is a breaking change and is not backward compatible.** The six regression
strategies (`legacy`, `quantile`, `threshold`, `outlier`, `hetero`, `valprop`) and the
regression calibrators are gone; see *Strategies* above. Classification is unchanged.

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
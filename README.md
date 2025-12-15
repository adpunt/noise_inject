# NoiseInject

A lightweight Python framework for testing ML model robustness to label noise in regression tasks.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from noiseInject import NoiseInjector, calculate_noise_metrics, calibrate_sigma
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Your data
X_train, y_train = ...  # Your training data
X_test, y_test = ...    # Your test data (clean labels)

# 1. Calibrate sigma for target effective noise
sigma = calibrate_sigma(y_train, target_effective_noise=0.1)
print(f"Calibrated sigma: {sigma}")

# 2. Create noise injector
injector = NoiseInjector(strategy='legacy', random_state=42)

# 3. Test at multiple noise levels
predictions = {}
for s in [0.0, 0.1, 0.3, 0.5]:
    # Inject noise
    if s == 0.0:
        y_train_noisy = y_train
    else:
        y_train_noisy = injector.inject(y_train, sigma=s)
    
    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train_noisy)
    
    # Store predictions
    predictions[s] = model.predict(X_test)

# 4. Analyze robustness
metrics_df = calculate_noise_metrics(y_test, predictions)
print(metrics_df)
```

## Noise Strategies

### 1. Legacy (Homoscedastic Gaussian)
Simple Gaussian noise: `y_noisy = y + σ * N(0,1)`

```python
injector = NoiseInjector('legacy')
y_noisy = injector.inject(y_train, sigma=0.1)
```

### 2. Quantile-Based
More noise at distribution extremes:

```python
injector = NoiseInjector('quantile')
y_noisy = injector.inject(y_train, sigma=0.1,
    high_quantile=0.9,
    low_quantile=0.1,
    high_sigma_mult=2.0,
    low_sigma_mult=2.0,
    mid_sigma_mult=0.1
)
```

### 3. Threshold-Based
More noise above/below absolute thresholds:

```python
injector = NoiseInjector('threshold')
y_noisy = injector.inject(y_train, sigma=0.1,
    high_threshold=1.0,
    low_threshold=-1.0,
    high_sigma_mult=2.0,
    low_sigma_mult=2.0,
    mid_sigma_mult=0.1
)
```

### 4. Outlier-Focused
More noise for z-score outliers:

```python
injector = NoiseInjector('outlier')
y_noisy = injector.inject(y_train, sigma=0.1,
    outlier_z_threshold=2.0,
    outlier_sigma_mult=3.0,
    normal_sigma_mult=0.1
)
```

### 5. Heteroscedastic
Variance proportional to absolute value: `σ_i = sqrt(α*σ² + β*σ²*|y_i|)`

```python
injector = NoiseInjector('hetero')
y_noisy = injector.inject(y_train, sigma=0.1,
    alpha_mult=0.1,
    beta_mult=0.05
)
```

### 6. Value-Proportional
Linear dependence on absolute value: `σ_i = base_sigma + prop_factor * |y_i|`

```python
injector = NoiseInjector('valprop')
y_noisy = injector.inject(y_train, sigma=0.1,
    proportionality_factor=0.1
)
```

## Metrics

### NSI (Noise Sensitivity Index)
Slope of performance degradation: `nsi = slope(R² vs σ)`

More negative NSI → faster degradation → less robust model

### Retention Percentage
Performance preservation at high noise:
- For R²: `(R²_high / R²_baseline) * 100`
- For RMSE/MAE: `(baseline / high) * 100`

Higher retention → more robust model

### Effective Noise
Actual noise level added to data:
```python
effective = injector.get_effective_noise(y_clean, y_noisy, method='std_normalized')
```

Methods:
- `std_normalized`: `mean(|y_noisy - y_clean|) / std(y_clean)` (default)
- `range_normalized`: `mean(|y_noisy - y_clean|) / (max - min)`
- `absolute`: `mean(|y_noisy - y_clean|)`

## Calibration

Find sigma that produces target effective noise:

```python
# Single target
sigma = calibrate_sigma(y_train, target_effective_noise=0.1)

# Multiple targets
sigma_dict = calibrate_multiple_sigmas(y_train, [0.05, 0.1, 0.2, 0.3])
# Returns: {0.05: 0.12, 0.1: 0.24, 0.2: 0.48, 0.3: 0.72}
```

This enables **fair comparison** between strategies:
```python
# Both produce ~10% effective noise
sigma_legacy = calibrate_sigma(y_train, target=0.1, strategy='legacy')
sigma_hetero = calibrate_sigma(y_train, target=0.1, strategy='hetero')
```

## Complete Example

```python
from noiseInject import NoiseInjector, calculate_noise_metrics, calibrate_sigma
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load data
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Compare strategies at same effective noise
strategies = ['legacy', 'quantile', 'hetero', 'outlier']
target_noise = 0.1

for strategy in strategies:
    print(f"\n{strategy.upper()} Strategy")
    print("=" * 50)
    
    # Calibrate sigma
    sigma = calibrate_sigma(y_train, target_noise, strategy=strategy)
    print(f"Calibrated sigma: {sigma:.4f}")
    
    # Test at multiple noise levels
    injector = NoiseInjector(strategy, random_state=42)
    predictions = {}
    
    for s_mult in [0.0, 0.5, 1.0, 2.0]:
        s = sigma * s_mult
        
        # Inject noise
        if s == 0.0:
            y_noisy = y_train
        else:
            y_noisy = injector.inject(y_train, s)
        
        # Train and predict
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_noisy)
        predictions[s] = model.predict(X_test)
    
    # Analyze
    metrics = calculate_noise_metrics(y_test, predictions)
    
    # Print summary
    summary = metrics[metrics['sigma'] == 'SUMMARY'].iloc[0]
    print(f"Baseline R²: {summary['baseline_r2']:.4f}")
    print(f"NSI (R²): {summary['nsi_r2']:.4f}")
    print(f"Retention: {summary['retention_pct_r2']:.2f}%")
```

## Output Format

```python
metrics_df = calculate_noise_metrics(y_test, predictions)
```

Returns DataFrame with:

**Per-sigma rows:**
- `sigma`: Noise level
- `r2`, `rmse`, `mae`: Performance metrics
- `effective_noise`: Actual noise added

**Summary row** (sigma='SUMMARY'):
- `nsi_<metric>`: Slope of degradation
- `nsi_<metric>_pval`: Statistical significance
- `nsi_<metric>_relative`: Normalized by baseline
- `baseline_<metric>`: Performance at σ=0
- `retention_pct_<metric>`: Retention at high noise

## Advanced Usage

### Custom Strategy Parameters

```python
injector = NoiseInjector('quantile')
y_noisy = injector.inject(y_train, sigma=0.2,
    high_quantile=0.95,  # Top 5% get more noise
    high_sigma_mult=5.0  # 5x more noise
)
```

### Multiple Representations Comparison

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

models = {
    'RF': RandomForestRegressor(),
    'GBM': GradientBoostingRegressor()
}

for model_name, model in models.items():
    predictions = {}
    for sigma in [0.0, 0.1, 0.2, 0.3]:
        y_noisy = injector.inject(y_train, sigma) if sigma > 0 else y_train
        model.fit(X_train, y_noisy)
        predictions[sigma] = model.predict(X_test)
    
    metrics = calculate_noise_metrics(y_test, predictions)
    print(f"\n{model_name}:")
    print(metrics[metrics['sigma'] == 'SUMMARY'])
```

## What NOT to Do

 **Don't inject noise into test set** - Always evaluate on clean labels
 **Don't compare strategies without calibration** - Use `calibrate_sigma()` for fair comparison
 **Don't ignore effective noise** - Different strategies produce different actual noise levels

## Design Principles

1. **Model-agnostic**: Works with any ML framework
2. **User-controlled training**: You train your models, we just inject noise
3. **Minimal dependencies**: Only numpy, pandas, scipy
4. **Generalized**: Not chemistry-specific (but works great for QSAR!)

## License

MIT

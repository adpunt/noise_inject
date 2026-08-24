"""
NoiseInject Core Module
Implements noise injection strategies for ML robustness testing
Supports both regression (continuous noise) and classification (label flips)
"""

import numpy as np
from typing import Optional, Dict, Any, Union


class NoiseInjectorRegression:
    """
    Inject noise into continuous target values using various strategies.
    
    Strategies:
        - legacy: Simple Gaussian σ * N(0,1)
        - quantile: More noise at distribution extremes
        - threshold: More noise above/below absolute thresholds
        - outlier: More noise for z-score outliers
        - hetero: Heteroscedastic noise (variance ∝ |y|)
        - valprop: Value-proportional noise
    """
    
    def __init__(self, strategy: str = 'legacy', random_state: Optional[int] = None):
        """
        Initialize NoiseInjectorRegression.
        
        Args:
            strategy: One of ['legacy', 'quantile', 'threshold', 'outlier', 'hetero', 'valprop']
            random_state: Random seed for reproducibility
        """
        valid_strategies = ['legacy', 'quantile', 'threshold', 'outlier', 'hetero', 'valprop']
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
        
        self.strategy = strategy
        self.rng = np.random.RandomState(random_state)
    
    def inject(self, y: np.ndarray, sigma: float, **strategy_params) -> np.ndarray:
        """
        Inject noise into target values.
        
        Args:
            y: Target values (1D numpy array)
            sigma: Base noise level
            **strategy_params: Strategy-specific parameters
        
        Returns:
            Noisy target values
        """
        y = np.asarray(y).flatten()
        
        if self.strategy == 'legacy':
            return self._legacy(y, sigma)
        elif self.strategy == 'quantile':
            return self._quantile(y, sigma, **strategy_params)
        elif self.strategy == 'threshold':
            return self._threshold(y, sigma, **strategy_params)
        elif self.strategy == 'outlier':
            return self._outlier(y, sigma, **strategy_params)
        elif self.strategy == 'hetero':
            return self._hetero(y, sigma, **strategy_params)
        elif self.strategy == 'valprop':
            return self._valprop(y, sigma, **strategy_params)
    
    # ------------------------------------------------------------------
    # PER-MOLECULE NOISE SCALE
    # ------------------------------------------------------------------
    # Every strategy below works the same way: it computes a per-molecule
    # noise scale sigma_i from the label, then draws noise ~ N(0, 1) and
    # multiplies. The `_scale_*` methods return sigma_i WITHOUT touching the
    # random number generator, so the scale can be computed for molecules that
    # are never actually corrupted (e.g. a held-out test set) in order to ask
    # "how noisy is this molecule's region of the label distribution?".
    #
    # `reference` is the array whose distribution defines the cut-points
    # (quantiles for `quantile`, mean/sd for `outlier`). It defaults to `y`
    # itself, which reproduces the original behaviour exactly. To score
    # held-out molecules against the noise pattern the TRAINING labels were
    # exposed to, pass reference=y_train.
    #
    # NOTE ON REPRODUCIBILITY: `_legacy` previously drew
    # rng.normal(0, sigma, n); it now draws rng.normal(0, 1, n) * sigma_i.
    # These are bit-identical for numpy RandomState (verified), so results
    # generated before and after this refactor match exactly.

    def _scale_legacy(self, y, sigma):
        return np.full(len(y), float(sigma))

    def _scale_quantile(self, y, sigma, reference=None,
                        high_quantile: float = 0.9,
                        low_quantile: float = 0.1,
                        high_sigma_mult: float = 2.0,
                        low_sigma_mult: float = 2.0,
                        mid_sigma_mult: float = 0.1):
        ref = y if reference is None else np.asarray(reference).flatten()
        high_thresh = np.quantile(ref, high_quantile)
        low_thresh = np.quantile(ref, low_quantile)
        sigma_values = np.full(len(y), sigma * mid_sigma_mult)
        sigma_values[y >= high_thresh] = sigma * high_sigma_mult
        sigma_values[y <= low_thresh] = sigma * low_sigma_mult
        return sigma_values

    def _scale_threshold(self, y, sigma, reference=None,
                         high_threshold: float = 1.0,
                         low_threshold: float = -1.0,
                         high_sigma_mult: float = 2.0,
                         low_sigma_mult: float = 2.0,
                         mid_sigma_mult: float = 0.1):
        # Absolute cut-points: independent of the reference distribution.
        sigma_values = np.full(len(y), sigma * mid_sigma_mult)
        sigma_values[y >= high_threshold] = sigma * high_sigma_mult
        sigma_values[y <= low_threshold] = sigma * low_sigma_mult
        return sigma_values

    def _scale_outlier(self, y, sigma, reference=None,
                       outlier_z_threshold: float = 2.0,
                       outlier_sigma_mult: float = 3.0,
                       normal_sigma_mult: float = 0.1):
        ref = y if reference is None else np.asarray(reference).flatten()
        z_scores = np.abs((y - np.mean(ref)) / np.std(ref))
        sigma_values = np.full(len(y), sigma * normal_sigma_mult)
        sigma_values[z_scores > outlier_z_threshold] = sigma * outlier_sigma_mult
        return sigma_values

    def _scale_hetero(self, y, sigma, reference=None,
                      alpha_mult: float = 0.1,
                      beta_mult: float = 0.05):
        alpha = sigma * sigma * alpha_mult
        beta = sigma * sigma * beta_mult
        return np.sqrt(alpha + beta * np.abs(y))

    def _scale_valprop(self, y, sigma, reference=None,
                       proportionality_factor: float = 0.05):
        return sigma * (1 + proportionality_factor * np.abs(y))

    def noise_scale(self, y: np.ndarray, sigma: float,
                    reference: Optional[np.ndarray] = None,
                    **strategy_params) -> np.ndarray:
        """Per-molecule noise scale sigma_i, with NO noise drawn.

        Pure function of the labels. Does not advance the random number
        generator, so it is safe to call at any point without changing what
        `inject` would produce.

        Args:
            y: labels to compute a scale for.
            reference: distribution defining the cut-points (see note above).
                Ignored by strategies whose scale is a pointwise function of y.
            **strategy_params: same overrides accepted by `inject`.

        Returns:
            Array of per-molecule noise standard deviations, same length as y.
        """
        y = np.asarray(y).flatten()
        if self.strategy == 'legacy':
            return self._scale_legacy(y, sigma)
        if self.strategy == 'quantile':
            return self._scale_quantile(y, sigma, reference, **strategy_params)
        if self.strategy == 'threshold':
            return self._scale_threshold(y, sigma, reference, **strategy_params)
        if self.strategy == 'outlier':
            return self._scale_outlier(y, sigma, reference, **strategy_params)
        if self.strategy == 'hetero':
            return self._scale_hetero(y, sigma, reference, **strategy_params)
        if self.strategy == 'valprop':
            return self._scale_valprop(y, sigma, reference, **strategy_params)
        raise ValueError(f"Unknown strategy: {self.strategy}")

    def inject_verbose(self, y: np.ndarray, sigma: float,
                       reference: Optional[np.ndarray] = None,
                       **strategy_params):
        """`inject`, but it also hands back what it did.

        Returns:
            y_noisy: the corrupted labels (identical to `inject` for the same
                RNG state).
            sigma_i: the per-molecule noise scale actually applied.
            epsilon: the noise actually added to each molecule.

        This is the entry point for any analysis that asks whether a model's
        predicted uncertainty tracks the corruption, because it is the only
        way to know per molecule how much corruption there was.
        """
        y = np.asarray(y).flatten()
        sigma_i = self.noise_scale(y, sigma, reference, **strategy_params)
        epsilon = self.rng.normal(0, 1, size=len(y)) * sigma_i
        return y + epsilon, sigma_i, epsilon

    def _legacy(self, y: np.ndarray, sigma: float) -> np.ndarray:
        """Simple Gaussian: y_noisy = y + sigma * N(0,1)"""
        return y + self.rng.normal(0, 1, size=len(y)) * self._scale_legacy(y, sigma)

    def _quantile(self, y: np.ndarray, sigma: float, **p) -> np.ndarray:
        """More noise at distribution extremes."""
        return y + self.rng.normal(0, 1, size=len(y)) * self._scale_quantile(y, sigma, None, **p)

    def _threshold(self, y: np.ndarray, sigma: float, **p) -> np.ndarray:
        """More noise above/below absolute thresholds."""
        return y + self.rng.normal(0, 1, size=len(y)) * self._scale_threshold(y, sigma, None, **p)

    def _outlier(self, y: np.ndarray, sigma: float, **p) -> np.ndarray:
        """More noise for z-score outliers."""
        return y + self.rng.normal(0, 1, size=len(y)) * self._scale_outlier(y, sigma, None, **p)

    def _hetero(self, y: np.ndarray, sigma: float, **p) -> np.ndarray:
        """Heteroscedastic noise: sigma_i = sqrt(alpha*s^2 + beta*s^2*|y_i|)"""
        return y + self.rng.normal(0, 1, size=len(y)) * self._scale_hetero(y, sigma, None, **p)

    def _valprop(self, y: np.ndarray, sigma: float, **p) -> np.ndarray:
        """Value-proportional: sigma_i = sigma * (1 + f * |y_i|)"""
        return y + self.rng.normal(0, 1, size=len(y)) * self._scale_valprop(y, sigma, None, **p)

    def get_effective_noise(self, y_clean: np.ndarray, y_noisy: np.ndarray, 
                           method: str = 'std_normalized') -> float:
        """
        Calculate effective noise level.
        
        Args:
            y_clean: Original clean values
            y_noisy: Noisy values
            method: One of ['std_normalized', 'range_normalized', 'absolute']
        
        Returns:
            Effective noise level
        """
        y_clean = np.asarray(y_clean).flatten()
        y_noisy = np.asarray(y_noisy).flatten()
        
        absolute_diff = np.mean(np.abs(y_noisy - y_clean))
        
        if method == 'absolute':
            return absolute_diff
        elif method == 'std_normalized':
            std = np.std(y_clean)
            return absolute_diff / std if std > 1e-10 else 0.0
        elif method == 'range_normalized':
            range_val = np.ptp(y_clean)
            return absolute_diff / range_val if range_val > 1e-10 else 0.0
        else:
            raise ValueError(f"Invalid method: {method}")


class NoiseInjectorClassification:
    """
    Inject label noise into classification targets using various strategies.
    
    Strategies:
        - uniform: Equal flip probability for all samples
        - class_imbalance: Flip rate varies by class frequency
        - binary_asymmetric: Asymmetric flip rates for binary classification
        - instance_noise: Random per-sample flip probability
        - class_dependent: Each class has its own flip probability
        - confusion_directed: Realistic confusion patterns with directed flips
    """
    
    def __init__(self, strategy: str = 'uniform', random_state: Optional[int] = None):
        """
        Initialize NoiseInjectorClassification.
        
        Args:
            strategy: One of ['uniform', 'class_imbalance', 'binary_asymmetric', 
                              'instance_noise', 'class_dependent', 'confusion_directed']
            random_state: Random seed for reproducibility
        """
        valid_strategies = ['uniform', 'class_imbalance', 'binary_asymmetric', 
                          'instance_noise', 'class_dependent', 'confusion_directed']
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
        
        self.strategy = strategy
        self.rng = np.random.RandomState(random_state)
    
    def inject(self, y: np.ndarray, flip_probability: float, **strategy_params) -> np.ndarray:
        """
        Inject label noise into classification targets.
        
        Args:
            y: Class labels (1D numpy array of integers)
            flip_probability: Base flip probability (0.0 to 1.0)
            **strategy_params: Strategy-specific parameters
        
        Returns:
            Noisy class labels
        """
        y = np.asarray(y).flatten()
        
        if not np.issubdtype(y.dtype, np.integer):
            # Try to convert to integer
            y = y.astype(int)
        
        if self.strategy == 'uniform':
            return self._uniform(y, flip_probability)
        elif self.strategy == 'class_imbalance':
            return self._class_imbalance(y, flip_probability, **strategy_params)
        elif self.strategy == 'binary_asymmetric':
            return self._binary_asymmetric(y, flip_probability, **strategy_params)
        elif self.strategy == 'instance_noise':
            return self._instance_noise(y, flip_probability, **strategy_params)
        elif self.strategy == 'class_dependent':
            return self._class_dependent(y, flip_probability, **strategy_params)
        elif self.strategy == 'confusion_directed':
            return self._confusion_directed(y, flip_probability, **strategy_params)
    
    def _uniform(self, y: np.ndarray, flip_probability: float) -> np.ndarray:
        """Uniform label flipping: each sample has equal probability to flip."""
        y_noisy = y.copy()
        n_classes = len(np.unique(y))
        
        flip_mask = self.rng.rand(len(y)) < flip_probability
        
        for idx in np.where(flip_mask)[0]:
            true_class = y[idx]
            wrong_classes = [c for c in range(n_classes) if c != true_class]
            y_noisy[idx] = self.rng.choice(wrong_classes)
        
        return y_noisy
    
    def _class_imbalance(self, y: np.ndarray, flip_probability: float,
                        mode: str = 'punish_rare',
                        rare_flip_mult: float = 2.0,
                        common_flip_mult: float = 0.5,
                        frequency_threshold: float = 0.5) -> np.ndarray:
        """Flip rate varies by class frequency."""
        y_noisy = y.copy()
        n_classes = len(np.unique(y))
        
        unique, counts = np.unique(y, return_counts=True)
        frequencies = counts / len(y)
        
        if 0 <= frequency_threshold <= 1 and frequency_threshold < np.min(frequencies):
            freq_threshold = np.quantile(frequencies, frequency_threshold)
        else:
            freq_threshold = frequency_threshold
        
        class_flip_probs = {}
        for cls, freq in zip(unique, frequencies):
            if mode == 'punish_rare':
                if freq < freq_threshold:
                    class_flip_probs[cls] = flip_probability * rare_flip_mult
                else:
                    class_flip_probs[cls] = flip_probability * common_flip_mult
            elif mode == 'punish_common':
                if freq >= freq_threshold:
                    class_flip_probs[cls] = flip_probability * common_flip_mult
                else:
                    class_flip_probs[cls] = flip_probability * rare_flip_mult
            else:
                raise ValueError(f"mode must be 'punish_rare' or 'punish_common', got {mode}")
        
        class_flip_probs = {k: min(1.0, max(0.0, v)) for k, v in class_flip_probs.items()}
        
        for cls in unique:
            cls_mask = (y == cls)
            flip_mask = self.rng.rand(np.sum(cls_mask)) < class_flip_probs[cls]
            
            cls_indices = np.where(cls_mask)[0]
            flip_indices = cls_indices[flip_mask]
            
            for idx in flip_indices:
                wrong_classes = [c for c in range(n_classes) if c != cls]
                y_noisy[idx] = self.rng.choice(wrong_classes)
        
        return y_noisy
    
    def _binary_asymmetric(self, y: np.ndarray, flip_probability: float,
                          flip_01_mult: float = 1.5,
                          flip_10_mult: float = 0.5) -> np.ndarray:
        """Asymmetric flip rates for binary classification."""
        unique_classes = np.unique(y)
        
        if len(unique_classes) != 2:
            raise ValueError(f"binary_asymmetric requires exactly 2 classes, got {len(unique_classes)}")
        
        y_noisy = y.copy()
        
        class_0 = unique_classes[0]
        class_1 = unique_classes[1]

        # Use integer indices: chained boolean indexing (y_noisy[mask][submask] = ...)
        # assigns into a copy and silently does nothing.
        idx_0 = np.where(y == class_0)[0]
        flip_prob_01 = min(1.0, max(0.0, flip_probability * flip_01_mult))
        flip_mask_01 = self.rng.rand(len(idx_0)) < flip_prob_01
        y_noisy[idx_0[flip_mask_01]] = class_1

        idx_1 = np.where(y == class_1)[0]
        flip_prob_10 = min(1.0, max(0.0, flip_probability * flip_10_mult))
        flip_mask_10 = self.rng.rand(len(idx_1)) < flip_prob_10
        y_noisy[idx_1[flip_mask_10]] = class_0

        return y_noisy
    
    def _instance_noise(self, y: np.ndarray, flip_probability: float,
                       noise_std: float = 0.3,
                       min_mult: float = 0.1,
                       max_mult: float = 3.0) -> np.ndarray:
        """Random per-sample flip probability."""
        y_noisy = y.copy()
        n_classes = len(np.unique(y))
        
        multipliers = self.rng.normal(1.0, noise_std, size=len(y))
        multipliers = np.clip(multipliers, min_mult, max_mult)
        
        sample_flip_probs = flip_probability * multipliers
        sample_flip_probs = np.clip(sample_flip_probs, 0.0, 1.0)
        
        flip_mask = self.rng.rand(len(y)) < sample_flip_probs
        
        for idx in np.where(flip_mask)[0]:
            true_class = y[idx]
            wrong_classes = [c for c in range(n_classes) if c != true_class]
            y_noisy[idx] = self.rng.choice(wrong_classes)
        
        return y_noisy
    
    def _class_dependent(self, y: np.ndarray, flip_probability: float,
                        class_flip_rates: Optional[Union[Dict[int, float], np.ndarray]] = None,
                        auto_mode: str = 'inverse_frequency') -> np.ndarray:
        """Each class has its own flip probability."""
        y_noisy = y.copy()
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        if class_flip_rates is None:
            unique, counts = np.unique(y, return_counts=True)
            frequencies = counts / len(y)
            
            if auto_mode == 'inverse_frequency':
                rates = {cls: flip_probability * (freq / np.max(frequencies)) * 2.0 
                        for cls, freq in zip(unique, frequencies)}
            elif auto_mode == 'proportional_frequency':
                inv_freq = 1.0 - frequencies
                rates = {cls: flip_probability * (inv / np.max(inv_freq)) * 2.0 
                        for cls, inv in zip(unique, inv_freq)}
            elif auto_mode == 'uniform':
                rates = {cls: flip_probability for cls in unique}
            else:
                raise ValueError(f"Invalid auto_mode: {auto_mode}")
            
            class_flip_rates = rates
        
        if isinstance(class_flip_rates, np.ndarray):
            class_flip_rates = {i: class_flip_rates[i] for i in range(len(class_flip_rates))}
        
        for cls in unique_classes:
            if cls not in class_flip_rates:
                class_flip_rates[cls] = flip_probability
        
        class_flip_rates = {k: min(1.0, max(0.0, v)) for k, v in class_flip_rates.items()}
        
        for cls in unique_classes:
            cls_mask = (y == cls)
            flip_prob = class_flip_rates[cls]
            flip_mask = self.rng.rand(np.sum(cls_mask)) < flip_prob
            
            cls_indices = np.where(cls_mask)[0]
            flip_indices = cls_indices[flip_mask]
            
            for idx in flip_indices:
                wrong_classes = [c for c in range(n_classes) if c != cls]
                y_noisy[idx] = self.rng.choice(wrong_classes)
        
        return y_noisy
    
    def _confusion_directed(self, y: np.ndarray, flip_probability: float,
                          confusion_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """Realistic confusion patterns with directed flips."""
        y_noisy = y.copy()
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        if confusion_matrix is None:
            confusion_matrix = np.ones((n_classes, n_classes)) * (flip_probability / (n_classes - 1))
            np.fill_diagonal(confusion_matrix, 1.0 - flip_probability)
        
        if confusion_matrix.shape != (n_classes, n_classes):
            raise ValueError(f"confusion_matrix must be {n_classes}×{n_classes}, got {confusion_matrix.shape}")
        
        row_sums = confusion_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError(f"confusion_matrix rows must sum to 1.0, got {row_sums}")
        
        for cls_idx, cls in enumerate(unique_classes):
            cls_mask = (y == cls)
            cls_indices = np.where(cls_mask)[0]
            
            new_labels = self.rng.choice(
                unique_classes, 
                size=len(cls_indices),
                p=confusion_matrix[cls_idx]
            )
            
            y_noisy[cls_mask] = new_labels
        
        return y_noisy
    
    def get_effective_flip_rate(self, y_clean: np.ndarray, y_noisy: np.ndarray) -> float:
        """Calculate effective flip rate (fraction of labels that changed)."""
        y_clean = np.asarray(y_clean).flatten()
        y_noisy = np.asarray(y_noisy).flatten()
        
        return np.mean(y_clean != y_noisy)
    
    def get_per_class_flip_rates(self, y_clean: np.ndarray, y_noisy: np.ndarray) -> Dict[int, float]:
        """Calculate flip rate for each class separately."""
        y_clean = np.asarray(y_clean).flatten()
        y_noisy = np.asarray(y_noisy).flatten()
        
        per_class_rates = {}
        for cls in np.unique(y_clean):
            cls_mask = (y_clean == cls)
            cls_flip_rate = np.mean(y_clean[cls_mask] != y_noisy[cls_mask])
            per_class_rates[cls] = cls_flip_rate
        
        return per_class_rates
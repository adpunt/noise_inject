"""
NoiseInject Core Module
Implements noise injection strategies for ML robustness testing
Supports both regression (continuous noise) and classification (label flips)

REGRESSION: THE DOSE IS THE RESULT, NOT THE KNOB
------------------------------------------------
Every regression condition solves for its own internal scale so that the
root-mean-square noise actually added equals the requested `dose`, in the
label's own units. Comparing conditions at a common nominal setting -- the
previous behaviour -- compares amount, not shape: the six superseded
strategies delivered between 0.49x and 2.00x the same amount of noise at one
setting, and their entire apparent severity ordering was explained by that.

The specification is `NOISE_DESIGN.md` in the qsar_qm_models repository
(sections 1, 2 and 6). It is implemented twice -- here and in
`rust/src/main.rs` -- and the two implementations are held together by an
executable cross-check (`scripts/crosscheck_injectors.py`, gate 2 of
`RERUN_PLAN.md` section 8), because they have silently drifted apart before.

Shape and targeting are separately selectable, mirroring the Rust side:

    distribution  -- the shape of each draw: gaussian, student_t, laplace
    strategy      -- who gets hit and how hard: uniform, grouped_wider,
                     grouped_shifted, outlier, censoring

Censoring is the one condition that is neither zero-mean nor dose-matched; it
is parameterised by the fraction of labels clipped and reports its delivered
dose as a diagnostic.
"""

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# The condition registry
# ---------------------------------------------------------------------------
# One string per run condition, so a job script, a results row and a figure
# label all carry the same name. Every entry maps to a (strategy,
# distribution, parameters) triple.
#
# Parameter provenance (NOISE_DESIGN.md section 2):
#   lambda = 3      within-laboratory error must be multiplied by about three
#                   to reach between-laboratory error (Avdeef 2019, corroborated
#                   by Llinas & Avdeef 2019 and Kalliokoski et al. 2013)
#   p = 1-10%       "for scientific routine data, not taken with utmost care,
#                   their fraction is typically between 1 percent and 10
#                   percent" (Hampel 2001)
#   group_fraction  no published number exists; 0.2 is a choice and is declared
#                   as one
#   rho = 0.62      share of total measurement variance carried by the
#                   group-level term (Bentz et al. 2013, Table 7)

CONDITIONS: Dict[str, Dict[str, Any]] = {
    'gaussian':        dict(strategy='uniform',         distribution='gaussian'),
    'student_t_nu10':  dict(strategy='uniform',         distribution='student_t', nu=10.0),
    'student_t_nu5':   dict(strategy='uniform',         distribution='student_t', nu=5.0),
    'student_t_nu3':   dict(strategy='uniform',         distribution='student_t', nu=3.0),
    'laplace':         dict(strategy='uniform',         distribution='laplace'),
    'grouped_wider':   dict(strategy='grouped_wider',   distribution='gaussian', lam=3.0, group_fraction=0.2),
    'grouped_shifted': dict(strategy='grouped_shifted', distribution='gaussian', rho=0.62),
    'outlier_p01':     dict(strategy='outlier',         distribution='gaussian', p=0.01, lam=3.0),
    'outlier_p05':     dict(strategy='outlier',         distribution='gaussian', p=0.05, lam=3.0),
    'outlier_p10':     dict(strategy='outlier',         distribution='gaussian', p=0.10, lam=3.0),
    'censoring_10':    dict(strategy='censoring',       distribution='gaussian', censored_fraction=0.10),
    'censoring_20':    dict(strategy='censoring',       distribution='gaussian', censored_fraction=0.20),
    'censoring_25':    dict(strategy='censoring',       distribution='gaussian', censored_fraction=0.25),
    'censoring_30':    dict(strategy='censoring',       distribution='gaussian', censored_fraction=0.30),
    'censoring_40':    dict(strategy='censoring',       distribution='gaussian', censored_fraction=0.40),
    'censoring_50':    dict(strategy='censoring',       distribution='gaussian', censored_fraction=0.50),
}

REGRESSION_STRATEGIES = ('uniform', 'grouped_wider', 'grouped_shifted', 'outlier', 'censoring')
REGRESSION_DISTRIBUTIONS = ('gaussian', 'student_t', 'laplace')

# Conditions whose per-molecule noise scale is the same for every molecule.
# For these the question "which region of the label space is unreliable" is
# undefined rather than answered with zero -- see `InjectionResult.scale_is_degenerate`.
_CONSTANT_SCALE_STRATEGIES = ('uniform',)


class InjectionResult:
    """Everything one injection produced, including its provenance.

    The single reason the dose confound went unnoticed for the life of the
    project is that nothing recorded how much noise was actually delivered.
    Every field below exists to be written to a results row
    (`RERUN_PLAN.md` section 5.2).
    """

    __slots__ = ('y_clean', 'y_noisy', 'epsilon', 'noise_scale', 'condition',
                 'strategy', 'distribution', 'params', 'target_dose',
                 'unit_dose', 'solved_scale', 'delivered_dose',
                 'delivered_dose_fraction_of_sd', 'affected_molecule_fraction',
                 'mean_shift', 'n_groups', 'largest_group_share', 'seed',
                 'scale_is_degenerate', 'censoring_limit')

    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs.get(name))

    def as_row(self) -> Dict[str, Any]:
        """The provenance fields, for writing beside every result."""
        return {
            'condition': self.condition,
            'strategy': self.strategy,
            'distribution': self.distribution,
            'target_dose': self.target_dose,
            'unit_dose': self.unit_dose,
            'solved_scale': self.solved_scale,
            'delivered_dose': self.delivered_dose,
            'delivered_dose_fraction_of_sd': self.delivered_dose_fraction_of_sd,
            'affected_molecule_fraction': self.affected_molecule_fraction,
            'mean_shift': self.mean_shift,
            'n_groups': self.n_groups,
            'largest_group_share': self.largest_group_share,
            'censoring_limit': self.censoring_limit,
            'seed': self.seed,
            'scale_is_degenerate': self.scale_is_degenerate,
        }

    def __iter__(self):
        """Backwards-compatible unpacking: y_noisy, noise_scale, epsilon."""
        return iter((self.y_noisy, self.noise_scale, self.epsilon))

    def __repr__(self):
        return (f"InjectionResult(condition={self.condition!r}, "
                f"target_dose={self.target_dose:.4g}, "
                f"delivered_dose={self.delivered_dose:.4g}, "
                f"affected={self.affected_molecule_fraction:.4g})")


class NoiseInjectorRegression:
    """
    Inject dose-matched noise into continuous target values.

    Shape (`distribution`):
        - gaussian:  the reference case
        - student_t: heavy-tailed; nu must exceed 2 or the variance -- and so
                     "the same amount of noise" -- is undefined
        - laplace:   the shape actually fitted to real bioactivity error

    Targeting (`strategy`):
        - uniform:         every molecule gets the same scale
        - grouped_wider:   molecules in a fraction of groups get a wider error,
                           still centred on the true value
        - grouped_shifted: every group's labels are pushed in one direction by
                           a constant, plus a within-molecule error
        - outlier:         a randomly chosen fraction get a wider error.
                           Selection is RANDOM, not by label value: a mistyped
                           unit is a property of the record, not of the value
        - censoring:       values past a limit are recorded as the limit.
                           Not zero-mean, not dose-matched

    The `dose` argument to `inject`/`inject_verbose` is the root-mean-square
    noise to deliver, in the label's own units. The RMS is the right moment to
    match because R-squared and RMSE are second-moment quantities.

    Reproducibility: a fresh generator is built per instance and advanced by
    every draw, so calling inject() twice on one instance gives different
    noise. Callers that need a deterministic realisation construct a fresh
    injector per call.
    """

    def __init__(self, strategy: str = 'uniform', distribution: str = 'gaussian',
                 random_state: Optional[int] = None, **params):
        """
        Args:
            strategy: one of REGRESSION_STRATEGIES
            distribution: one of REGRESSION_DISTRIBUTIONS
            random_state: seed
            **params: condition parameters (nu, lam, group_fraction, rho, p,
                      censored_fraction, side). May also be supplied per call.
        """
        if strategy not in REGRESSION_STRATEGIES:
            raise ValueError(f"strategy must be one of {REGRESSION_STRATEGIES}, got {strategy!r}")
        if distribution not in REGRESSION_DISTRIBUTIONS:
            raise ValueError(f"distribution must be one of {REGRESSION_DISTRIBUTIONS}, got {distribution!r}")

        nu = params.get('nu')
        if distribution == 'student_t':
            if nu is None:
                raise ValueError("student_t requires nu")
            if float(nu) <= 2.0:
                raise ValueError(
                    f"student_t requires nu > 2 (got {nu}); at nu <= 2 the variance is "
                    "undefined and dose matching stops meaning anything")

        self.strategy = strategy
        self.distribution = distribution
        self.params = dict(params)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    # ------------------------------------------------------------------
    @classmethod
    def from_condition(cls, condition: str, random_state: Optional[int] = None):
        """Build from a registry name, e.g. 'student_t_nu5' or 'censoring_25'."""
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition!r}; known: {sorted(CONDITIONS)}")
        spec = dict(CONDITIONS[condition])
        strategy = spec.pop('strategy')
        distribution = spec.pop('distribution')
        inj = cls(strategy=strategy, distribution=distribution,
                  random_state=random_state, **spec)
        inj.condition = condition
        return inj

    @property
    def condition(self) -> str:
        """The registry name for this configuration, if it has one."""
        if getattr(self, '_condition', None):
            return self._condition
        for name, spec in CONDITIONS.items():
            if spec['strategy'] != self.strategy or spec['distribution'] != self.distribution:
                continue
            rest = {k: v for k, v in spec.items() if k not in ('strategy', 'distribution')}
            if all(float(self.params.get(k, float('nan'))) == float(v) for k, v in rest.items()):
                return name
        return f"{self.strategy}/{self.distribution}"

    @condition.setter
    def condition(self, value):
        self._condition = value

    # ------------------------------------------------------------------
    # SHAPE
    # ------------------------------------------------------------------
    def _shape_unit_sd(self, **params) -> float:
        """The standard deviation of one unscaled draw from the shape."""
        if self.distribution == 'student_t':
            nu = float(self._param('nu', params))
            return math.sqrt(nu / (nu - 2.0))
        if self.distribution == 'laplace':
            return math.sqrt(2.0)
        return 1.0

    def _draw_shape(self, n: int, **params) -> np.ndarray:
        """n draws from the shape, at its natural scale (unit_sd as above)."""
        if self.distribution == 'student_t':
            nu = float(self._param('nu', params))
            z = self.rng.standard_normal(n)
            v = self.rng.chisquare(nu, size=n)
            return z / np.sqrt(v / nu)
        if self.distribution == 'laplace':
            # Inverse transform, unit scale b = 1 -> sd = sqrt(2)
            u = self.rng.random_sample(n) - 0.5
            return -np.sign(u) * np.log1p(-2.0 * np.abs(u))
        return self.rng.standard_normal(n)

    def _param(self, name: str, params: Dict[str, Any], default=None):
        if name in params and params[name] is not None:
            return params[name]
        if name in self.params and self.params[name] is not None:
            return self.params[name]
        if default is not None:
            return default
        raise ValueError(f"{self.strategy}/{self.distribution} requires parameter {name!r}")

    # ------------------------------------------------------------------
    # TARGETING -- the per-molecule scale map
    # ------------------------------------------------------------------
    def scale_map(self, y: np.ndarray, groups: Optional[np.ndarray] = None,
                  **params) -> Tuple[np.ndarray, float]:
        """Per-molecule multipliers at unit scale, and the affected fraction.

        Draws from the generator for the two selection rules (which groups,
        which records), and is pure for the rest.
        """
        y = np.asarray(y).flatten()
        n = len(y)

        if self.strategy == 'uniform':
            return np.ones(n), 0.0

        if self.strategy == 'grouped_wider':
            lam = float(self._param('lam', params))
            f = float(self._param('group_fraction', params))
            groups = self._require_groups(groups, n)
            affected = self._select_groups_by_molecule_fraction(groups, f)
            scales = np.where(affected, lam, 1.0)
            return scales, float(affected.mean())

        if self.strategy == 'grouped_shifted':
            # The scale map is uniform; the group structure enters as an
            # additive offset in `_draw_epsilon`, not as a multiplier.
            return np.ones(n), 1.0

        if self.strategy == 'outlier':
            lam = float(self._param('lam', params))
            p = float(self._param('p', params))
            hit = self.rng.random_sample(n) < p
            scales = np.where(hit, lam, 1.0)
            return scales, float(hit.mean())

        if self.strategy == 'censoring':
            # Not dose-matched; the scale map is not used to solve anything.
            return np.ones(n), float(self._param('censored_fraction', params))

        raise AssertionError(f"unhandled strategy {self.strategy!r}")

    @staticmethod
    def _require_groups(groups, n):
        if groups is None:
            raise ValueError(
                "grouped noise needs a group assignment: pass groups=<array of ints>, "
                "one per molecule (e.g. from assign_scaffold_groups)")
        groups = np.asarray(groups).flatten()
        if len(groups) != n:
            raise ValueError(f"groups has length {len(groups)}, labels have {n}")
        return groups

    def _select_groups_by_molecule_fraction(self, groups: np.ndarray, f: float) -> np.ndarray:
        """Choose whole groups until the affected MOLECULE fraction is closest to f.

        Selecting a fraction of *groups* is what the first draft did, and it does
        not control who gets hit: real Murcko scaffolds are very unevenly sized.
        Measured on 10,000 QM9 molecules, where 32% share one empty (acyclic)
        scaffold, a nominal group fraction of 0.2 delivered an affected molecule
        fraction anywhere between 0.067 and 0.551. So select by molecule count,
        stop at the closest approach to f, and record what was realised.
        """
        uniq, inverse = np.unique(groups, return_inverse=True)
        sizes = np.bincount(inverse, minlength=len(uniq))
        n = len(groups)
        order = self.rng.permutation(len(uniq))

        chosen = []
        cum = 0
        for g in order:
            if cum > 0 and abs((cum + sizes[g]) / n - f) > abs(cum / n - f):
                continue  # this group overshoots; try a smaller one
            chosen.append(g)
            cum += sizes[g]
            if cum / n >= f:
                break
        chosen_set = np.zeros(len(uniq), dtype=bool)
        chosen_set[chosen] = True
        return chosen_set[inverse]

    # ------------------------------------------------------------------
    # THE DOSE SOLVER
    # ------------------------------------------------------------------
    def unit_dose(self, scales: np.ndarray, **params) -> float:
        """G -- the root-mean-square of the scale map times the shape's unit sd.

        The scale that delivers a target dose tau is then simply tau / G.
        """
        scales = np.asarray(scales, dtype=float)
        return float(np.sqrt(np.mean(scales ** 2)) * self._shape_unit_sd(**params))

    # ------------------------------------------------------------------
    # INJECTION
    # ------------------------------------------------------------------
    def inject_verbose(self, y: np.ndarray, dose: float,
                       groups: Optional[np.ndarray] = None,
                       reference: Optional[np.ndarray] = None,
                       **params) -> InjectionResult:
        """Inject noise and return it with everything needed to trace it.

        Args:
            y: clean labels
            dose: target root-mean-square noise, in the label's own units.
                  Ignored by censoring, which is parameterised by the fraction
                  of labels clipped.
            groups: integer group id per molecule; required by the two grouped
                    conditions
            reference: the labels whose distribution defines any cut-point.
                       Defaults to y. Pass the TRAINING labels to score
                       held-out molecules against the pattern the training set
                       was exposed to.

        Returns:
            InjectionResult. It also unpacks as (y_noisy, noise_scale, epsilon)
            for callers that want the three arrays.
        """
        y = np.asarray(y, dtype=float).flatten()
        n = len(y)
        dose = float(dose)
        if dose < 0:
            raise ValueError(f"dose must be non-negative, got {dose}")

        ref = y if reference is None else np.asarray(reference, dtype=float).flatten()
        n_groups, largest_share = self._group_summary(groups)

        if self.strategy == 'censoring':
            return self._inject_censoring(y, ref, n_groups, largest_share, **params)

        # Zero dose is exactly zero. Not a small number -- the negative control
        # the old reconstruction never had.
        if dose == 0.0:
            zeros = np.zeros(n)
            return self._result(y, zeros, zeros, dose, unit_dose=float('nan'),
                                solved_scale=0.0, affected=0.0,
                                n_groups=n_groups, largest_share=largest_share,
                                params=params)

        scales, affected = self.scale_map(y, groups=groups, **params)
        g = self.unit_dose(scales, **params)
        solved = dose / g

        if self.strategy == 'grouped_shifted':
            epsilon = self._draw_grouped_shifted(y, dose, groups, **params)
            noise_scale = np.full(n, dose)
        else:
            epsilon = self._draw_shape(n, **params) * (solved * scales)
            noise_scale = solved * scales * self._shape_unit_sd(**params)

        return self._result(y, epsilon, noise_scale, dose, unit_dose=g,
                            solved_scale=solved, affected=affected,
                            n_groups=n_groups, largest_share=largest_share,
                            params=params)

    def _draw_grouped_shifted(self, y, dose, groups, **params):
        """Group-level offset plus a within-molecule error.

            eps_i = sqrt(rho)*tau*b_g(i) + sqrt(1-rho)*tau*e_i

        The two variances sum to tau^2 by construction, so the condition is
        dose-matched without a solver step. rho is the share of total variance
        carried by the group-level term: 0.62, from Bentz et al. (2013) Table 7,
        where the laboratory term carries 62% of the variance in log efflux
        ratio across 23 laboratories.

        The offsets are NOT centred. This condition is not zero-mean in any one
        run, and that is the mechanism being tested: error pushed in one
        direction hurts far more than error that scatters.
        """
        groups = self._require_groups(groups, len(y))
        rho = float(self._param('rho', params, default=0.62))
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"rho must lie in [0, 1], got {rho}")
        uniq, inverse = np.unique(groups, return_inverse=True)
        b = self._draw_shape(len(uniq), **params)
        e = self._draw_shape(len(y), **params)
        return dose * (math.sqrt(rho) * b[inverse] + math.sqrt(1.0 - rho) * e)

    def _inject_censoring(self, y, ref, n_groups, largest_share, **params):
        """Clip values past an assay limit to the limit itself.

        The most prevalent real mechanism -- Svensson et al. report 25-63% of
        labels censored in ten of fifteen real industrial assays -- and the only
        one that biases labels in one direction instead of scattering them.
        Not zero-mean, so it is not dose-matched; it gets its own axis.
        """
        frac = float(self._param('censored_fraction', params))
        side = str(self._param('side', params, default='upper'))
        if not 0.0 <= frac < 1.0:
            raise ValueError(f"censored_fraction must lie in [0, 1), got {frac}")

        if frac == 0.0:
            zeros = np.zeros(len(y))
            return self._result(y, zeros, zeros, target_dose=float('nan'),
                                unit_dose=float('nan'), solved_scale=float('nan'),
                                affected=0.0, n_groups=n_groups,
                                largest_share=largest_share, params=params)

        if side == 'upper':
            limit = float(np.quantile(ref, 1.0 - frac))
            y_noisy = np.minimum(y, limit)
        elif side == 'lower':
            limit = float(np.quantile(ref, frac))
            y_noisy = np.maximum(y, limit)
        else:
            raise ValueError(f"side must be 'upper' or 'lower', got {side!r}")

        epsilon = y_noisy - y
        # The noise scale of a censored molecule is the distance it was moved;
        # unlike the zero-mean conditions this is a deterministic function of
        # the label, which is what makes censoring the label-keyed condition.
        noise_scale = np.abs(epsilon)
        return self._result(y, epsilon, noise_scale, target_dose=float('nan'),
                            unit_dose=float('nan'), solved_scale=float('nan'),
                            affected=float((epsilon != 0).mean()),
                            n_groups=n_groups, largest_share=largest_share,
                            params=params, censoring_limit=limit)

    def _result(self, y, epsilon, noise_scale, target_dose, unit_dose,
                solved_scale, affected, n_groups, largest_share, params,
                censoring_limit=None):
        y_noisy = y + epsilon
        sd = float(np.std(y))
        delivered = float(np.sqrt(np.mean(epsilon ** 2)))
        return InjectionResult(
            y_clean=y, y_noisy=y_noisy, epsilon=epsilon, noise_scale=noise_scale,
            condition=self.condition, strategy=self.strategy,
            distribution=self.distribution,
            params={**self.params, **params},
            target_dose=target_dose, unit_dose=unit_dose, solved_scale=solved_scale,
            delivered_dose=delivered,
            delivered_dose_fraction_of_sd=(delivered / sd if sd > 1e-12 else float('nan')),
            affected_molecule_fraction=affected,
            mean_shift=float(np.mean(epsilon)),
            n_groups=n_groups, largest_group_share=largest_share,
            seed=self.random_state, censoring_limit=censoring_limit,
            scale_is_degenerate=(self.strategy in _CONSTANT_SCALE_STRATEGIES),
        )

    @staticmethod
    def _group_summary(groups):
        if groups is None:
            return None, None
        groups = np.asarray(groups).flatten()
        _, counts = np.unique(groups, return_counts=True)
        return int(len(counts)), float(counts.max() / len(groups))

    def inject(self, y: np.ndarray, dose: float, groups: Optional[np.ndarray] = None,
               reference: Optional[np.ndarray] = None, **params) -> np.ndarray:
        """Inject noise, returning the noisy labels only."""
        return self.inject_verbose(y, dose, groups=groups, reference=reference,
                                   **params).y_noisy

    # ------------------------------------------------------------------
    # THE PER-MOLECULE SCALE, WITHOUT DRAWING
    # ------------------------------------------------------------------
    def noise_scale(self, y: np.ndarray, dose: float,
                    reference: Optional[np.ndarray] = None,
                    groups: Optional[np.ndarray] = None, **params) -> np.ndarray:
        """The noise scale each molecule's region receives, without corrupting it.

        This is what scores held-out molecules against the pattern the TRAINING
        labels were exposed to -- pass reference=y_train and the training group
        assignment. It is the input to the "does the model learn where the data
        is unreliable" question.

        For the three shape-only conditions the scale is the same for every
        molecule, so that question is UNDEFINED there rather than answered with
        zero. The array is still returned, and `scale_is_degenerate` on an
        InjectionResult says so; callers must not report a correlation against
        a constant.
        """
        y = np.asarray(y, dtype=float).flatten()
        dose = float(dose)

        if self.strategy == 'censoring':
            ref = y if reference is None else np.asarray(reference, dtype=float).flatten()
            frac = float(self._param('censored_fraction', params))
            side = str(self._param('side', params, default='upper'))
            if frac == 0.0:
                return np.zeros(len(y))
            if side == 'upper':
                limit = float(np.quantile(ref, 1.0 - frac))
                return np.maximum(y - limit, 0.0)
            limit = float(np.quantile(ref, frac))
            return np.maximum(limit - y, 0.0)

        if self.strategy == 'grouped_shifted':
            return np.full(len(y), dose)

        scales, _ = self.scale_map(y, groups=groups, **params)
        if dose == 0.0:
            return np.zeros(len(y))
        g = self.unit_dose(scales, **params)
        return (dose / g) * scales * self._shape_unit_sd(**params)

    # ------------------------------------------------------------------
    def get_effective_noise(self, y_clean: np.ndarray, y_noisy: np.ndarray,
                            method: str = 'rms_normalized') -> float:
        """Measure the noise that was actually delivered.

        The default is the ROOT-MEAN-SQUARE, in units of the label's spread.
        The previous default was the mean absolute deviation, which is the
        first moment; matching it hands the heavy-tailed conditions up to 24%
        more actual noise than Gaussian at the same nominal setting, because
        mean|e|/rms is 0.797 for a Gaussian but 0.642 for Student-t at nu = 3.
        R-squared and RMSE are second-moment quantities, so the second moment
        is what has to be matched.
        """
        y_clean = np.asarray(y_clean, dtype=float).flatten()
        y_noisy = np.asarray(y_noisy, dtype=float).flatten()
        eps = y_noisy - y_clean
        rms = float(np.sqrt(np.mean(eps ** 2)))

        if method == 'rms':
            return rms
        if method == 'rms_normalized':
            sd = float(np.std(y_clean))
            return rms / sd if sd > 1e-10 else 0.0
        if method == 'range_normalized':
            rng = float(np.ptp(y_clean))
            return rms / rng if rng > 1e-10 else 0.0
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
        

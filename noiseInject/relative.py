"""
Noise sized in a unit the caller supplies, rather than in the label's own SD.

The conditions in `core.CONDITIONS` take a dose in label units, and the study
usually states that dose as a multiple of the label SD. The conditions here
take a LEVEL and a UNIT instead. The unit is whatever scale the caller wants
the noise measured against -- KIRBy passes the SD of a model's out-of-fold
residual on the clean training labels, so the noise is sized against the error
the model already makes rather than against the spread of the labels. How the
unit is computed is the caller's business; this module only draws the noise.

Conditions (`RELATIVE_CONDITIONS`):

    dense_oofmse          every label gets N(0, level * unit^2). The level is
                          the injected variance as a share of unit^2.
    sparse_dprime_pXX     a share XX% of the labels moved by level * unit, the
                          sign drawn at random per label.
    sparse_dprime_up_pXX  the same, every one moved up.
    hetero_seen           every label gets Gaussian noise whose SD is
                          proportional to a per-molecule covariate the caller
                          passes (`heavy_atoms`, e.g. a heavy-atom count), so
                          the scale follows something a model can see.
    hetero_hidden         every label gets Gaussian noise whose SD is a
                          per-molecule lognormal(0, 1) draw, which nothing the
                          model has can see.

Both hetero conditions are scaled so the MEAN injected variance is
level * unit^2, the same as dense_oofmse at the same level. Nothing here
depends on the label's own value.

Seeding follows `NoiseInjectorRegression`:

    random_state     seeds the draws of the noise itself -- the Gaussian draws
                     and the sign of each sparse shift. One generator per
                     instance, advanced by every draw.
    selection_state  seeds WHO gets hit and how hard: which labels a sparse
                     condition moves, and hetero_hidden's per-molecule scale.
                     Re-seeded on every call from (seed ^ 0x5CA1E), so the
                     selection is the same at every level, including zero.
                     Defaults to random_state.

Level 0 returns the labels unchanged and marks nothing as moved.
"""

import math
import warnings
from typing import Any, Dict, Optional

import numpy as np

from .core import DoseWarning, dose_tolerance

RELATIVE_SPARSE_SHARES = {'p02': 0.02, 'p05': 0.05, 'p10': 0.10}

RELATIVE_CONDITIONS: Dict[str, Dict[str, Any]] = {
    'dense_oofmse': dict(kind='dense'),
    **{f'sparse_dprime_{k}': dict(kind='sparse', share=v, one_direction=False)
       for k, v in RELATIVE_SPARSE_SHARES.items()},
    **{f'sparse_dprime_up_{k}': dict(kind='sparse', share=v, one_direction=True)
       for k, v in RELATIVE_SPARSE_SHARES.items()},
    'hetero_seen': dict(kind='hetero_seen'),
    'hetero_hidden': dict(kind='hetero_hidden', lognormal_sigma=1.0),
}


class RelativeInjectionResult:
    """One relative injection: the labels, which ones moved, and its provenance."""

    __slots__ = ('y_clean', 'y_noisy', 'epsilon', 'moved_mask', 'noise_scale',
                 'condition', 'level', 'unit', 'seed', 'selection_seed',
                 'provenance', 'scale_is_degenerate')

    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs.get(name))

    def as_row(self) -> Dict[str, Any]:
        """The provenance fields, for writing beside every result."""
        return dict(self.provenance)

    def __iter__(self):
        """Unpacks as (y_noisy, moved_mask, provenance)."""
        return iter((self.y_noisy, self.moved_mask, self.as_row()))

    def __repr__(self):
        return (f"RelativeInjectionResult(condition={self.condition!r}, "
                f"level={self.level:.4g}, unit={self.unit:.4g}, "
                f"n_moved={int(self.moved_mask.sum())})")


class RelativeNoiseInjector:
    """Inject a `RELATIVE_CONDITIONS` member, sized in a caller-supplied unit.

    Usage:
        inj = RelativeNoiseInjector.from_condition('sparse_dprime_p05',
                                                   random_state=seed)
        res = inj.inject_verbose(y_train, level=2.0, unit=resid_sd)
        res.y_noisy, res.moved_mask, res.as_row()
    """

    def __init__(self, condition: str, random_state: Optional[int] = None,
                 selection_state: Optional[int] = None):
        if condition not in RELATIVE_CONDITIONS:
            raise ValueError(f"unknown relative condition {condition!r}; "
                             f"known: {list(RELATIVE_CONDITIONS)}")
        self.condition = condition
        self.spec = dict(RELATIVE_CONDITIONS[condition])
        self.random_state = random_state
        self.selection_state = selection_state
        self.rng = np.random.RandomState(random_state)

    @classmethod
    def from_condition(cls, condition: str, random_state: Optional[int] = None,
                       selection_state: Optional[int] = None):
        """Build from a registry name, as `NoiseInjectorRegression.from_condition`."""
        return cls(condition, random_state=random_state,
                   selection_state=selection_state)

    def _selection_seed(self):
        seed = self.selection_state if self.selection_state is not None \
            else self.random_state
        return None if seed is None else (int(seed) ^ 0x5CA1E) & 0xFFFFFFFF

    def _selection_rng(self) -> np.random.RandomState:
        """Re-seeded per call; same rule as NoiseInjectorRegression._selection_rng."""
        return np.random.RandomState(self._selection_seed())

    @property
    def scale_is_degenerate(self) -> bool:
        """True when every label gets the same noise scale (dense_oofmse)."""
        return self.spec['kind'] == 'dense'

    def inject_verbose(self, y: np.ndarray, level: float, unit: float,
                       heavy_atoms: Optional[np.ndarray] = None
                       ) -> RelativeInjectionResult:
        """Inject noise and return it with its provenance.

        Args:
            y:           clean labels
            level:       share of unit^2 injected as variance (dense_oofmse,
                         hetero_*), or the shift in units (sparse_dprime_*)
            unit:        the scale the level is measured in, in label units
            heavy_atoms: per-molecule covariate the noise SD follows, one per
                         label; required by hetero_seen and ignored otherwise.
                         Non-finite entries take the median of the rest.

        Returns:
            RelativeInjectionResult. It also unpacks as
            (y_noisy, moved_mask, provenance). Under the conditions that move
            every label the mask is all True at any level above zero.
        """
        y = np.asarray(y, dtype=float).ravel()
        n = len(y)
        level = float(level)
        unit = float(unit)
        if level < 0 or not math.isfinite(level):
            raise ValueError(f"level must be finite and non-negative, got {level}")
        if unit < 0 or not math.isfinite(unit):
            raise ValueError(f"unit must be finite and non-negative, got {unit}")
        kind = self.spec['kind']
        if kind == 'hetero_seen':
            if heavy_atoms is None:
                raise ValueError("hetero_seen needs heavy_atoms, one per label")
            if len(np.asarray(heavy_atoms).ravel()) != n:
                raise ValueError(f"heavy_atoms has length "
                                 f"{len(np.asarray(heavy_atoms).ravel())}, "
                                 f"labels have {n}")

        prov = {'condition': self.condition, 'level': level, 'unit': unit,
                'n_training': int(n), 'seed': self.random_state,
                'selection_seed': self._selection_seed()}
        moved = np.zeros(n, dtype=bool)
        eps = np.zeros(n)
        scale = np.zeros(n)

        if level == 0.0:
            prov['n_moved'] = 0
            return self._result(y, eps, moved, scale, level, unit, prov)

        if kind == 'sparse':
            share = float(self.spec['share'])
            one_way = bool(self.spec['one_direction'])
            n_move = int(round(share * n))
            if n_move > 0:
                moved[self._selection_rng().choice(n, size=n_move,
                                                   replace=False)] = True
            sign = (np.ones(n) if one_way
                    else self.rng.choice([-1.0, 1.0], size=n))
            shift = level * unit
            eps[moved] = sign[moved] * shift
            scale[moved] = shift
            prov.update({'share': share, 'one_direction': one_way,
                         'shift_label_units': shift,
                         'n_moved': int(moved.sum()),
                         'n_up': int((moved & (sign > 0)).sum())})
            return self._result(y, eps, moved, scale, level, unit, prov)

        target_var = level * unit ** 2
        if kind == 'dense':
            s = np.ones(n)
        elif kind == 'hetero_seen':
            h = np.asarray(heavy_atoms, dtype=float).ravel().copy()
            finite = np.isfinite(h)
            if not finite.any():
                raise ValueError("hetero_seen: heavy_atoms has no finite entry")
            h[~finite] = np.median(h[finite])
            if np.mean(h ** 2) <= 0:
                raise ValueError("hetero_seen: heavy_atoms is zero everywhere")
            s = h
        else:  # hetero_hidden
            s = self._selection_rng().lognormal(
                0.0, float(self.spec['lognormal_sigma']), size=n)
        scale = s / np.sqrt(np.mean(s ** 2)) * np.sqrt(target_var)
        z = self.rng.standard_normal(n)
        eps = z * scale
        moved[:] = True

        realised_var = float(np.mean(eps ** 2))
        if target_var > 0:
            s2 = float(np.sum(scale ** 2))
            s4 = float(np.sum(scale ** 4))
            effective_n = (s2 * s2 / s4) if s4 > 0 else float(n)
            tol = dose_tolerance(z, effective_n)
            ratio = math.sqrt(realised_var / target_var)
            if abs(ratio - 1.0) > tol:
                warnings.warn(
                    f"{self.condition} delivered RMS {math.sqrt(realised_var):.6f} "
                    f"against a target of {math.sqrt(target_var):.6f} "
                    f"({100 * (ratio - 1.0):+.2f}%), outside the {100 * tol:.2f}% "
                    f"band that {effective_n:.0f} effective observations allow. "
                    f"The delivered amount is recorded on the row.",
                    DoseWarning, stacklevel=2)
        prov.update({'injected_variance_target': target_var,
                     'injected_variance_realised': realised_var,
                     'sd_min': float(scale.min()), 'sd_max': float(scale.max()),
                     'n_moved': int(n)})
        return self._result(y, eps, moved, scale, level, unit, prov)

    def inject(self, y: np.ndarray, level: float, unit: float,
               heavy_atoms: Optional[np.ndarray] = None) -> np.ndarray:
        """Inject noise, returning the noisy labels only."""
        return self.inject_verbose(y, level, unit, heavy_atoms=heavy_atoms).y_noisy

    def _result(self, y, eps, moved, scale, level, unit, prov):
        return RelativeInjectionResult(
            y_clean=y, y_noisy=y + eps, epsilon=eps, moved_mask=moved,
            noise_scale=scale, condition=self.condition, level=level, unit=unit,
            seed=self.random_state, selection_seed=self._selection_seed(),
            provenance=prov, scale_is_degenerate=self.scale_is_degenerate)

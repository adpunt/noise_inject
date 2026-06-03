"""
Optional uncertainty-producing model wrappers for NoiseInject.

These are thin reference implementations that turn a point predictor into one that
returns per-sample uncertainty, so its calibration can be tracked under noise with
``noiseInject.calculate_uncertainty_metrics``. They carry heavier dependencies than
the core package and are therefore installed via the ``uncertainty`` extra::

    pip install noiseInject[uncertainty]

Imports are lazy: pulling one wrapper does not require the others' dependencies.
"""

from .conformal import SplitConformalRegressor

__all__ = ['SplitConformalRegressor', 'MCDropoutRegressor', 'GaucheGPRegressor']


def __getattr__(name):
    # Lazy access so importing the subpackage doesn't require torch / gauche.
    if name == 'MCDropoutRegressor':
        from .mc_dropout import MCDropoutRegressor
        return MCDropoutRegressor
    if name == 'GaucheGPRegressor':
        from .gp import GaucheGPRegressor
        return GaucheGPRegressor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

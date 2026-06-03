"""
Split-conformal prediction for any sklearn-compatible regressor.

Split conformal calibrates a held-out set of absolute residuals into a distribution-
free prediction interval with finite-sample coverage guarantees (Papadopoulos et al.,
2002; Lei et al., 2018). Only numpy + a fitted base estimator are required.

The wrapper exposes a per-sample ``uncertainty`` so it can be fed straight into
``calculate_uncertainty_metrics``. Plain split conformal is homoscedastic (one width
for all samples); set ``normalize=True`` for a locally-adaptive width that scales with
a kNN estimate of local difficulty.
"""

import numpy as np


class SplitConformalRegressor:
    """
    Wrap an sklearn-style regressor with split-conformal prediction intervals.

    Args:
        base_estimator: Any object with ``fit(X, y)`` and ``predict(X)``.
        coverage: Target coverage used to convert the interval to a 1-sigma-like
                  ``uncertainty`` (default 0.6827, i.e. a one-sigma band).
        normalize: If True, scale residuals by a local-difficulty estimate so the
                   interval width varies per sample (normalized conformal).
        n_neighbors: Neighbourhood size for the difficulty estimate when normalizing.
    """

    def __init__(self, base_estimator, coverage=0.6827, normalize=False, n_neighbors=20):
        self.base_estimator = base_estimator
        self.coverage = coverage
        self.normalize = normalize
        self.n_neighbors = n_neighbors

    def fit(self, X_train, y_train, X_calib, y_calib):
        """Fit the base estimator on the training split and calibrate on the held-out split."""
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train).flatten()
        X_calib = np.asarray(X_calib)
        y_calib = np.asarray(y_calib).flatten()

        self.base_estimator.fit(X_train, y_train)

        residuals = np.abs(y_calib - self.base_estimator.predict(X_calib))

        if self.normalize:
            self._fit_difficulty(X_train, y_train)
            scores = residuals / self._difficulty(X_calib)
        else:
            scores = residuals

        # Conformal quantile with the finite-sample correction.
        n = len(scores)
        level = min(1.0, np.ceil((n + 1) * self.coverage) / n)
        self.quantile_ = np.quantile(scores, level, method='higher')
        return self

    def predict(self, X):
        """
        Return ``(mean, uncertainty)`` where ``uncertainty`` is the conformal
        half-width at the target coverage (a one-sigma-like scale).
        """
        X = np.asarray(X)
        mean = self.base_estimator.predict(X)
        if self.normalize:
            u = self.quantile_ * self._difficulty(X)
        else:
            u = np.full(len(mean), self.quantile_)
        return mean, u

    def predict_interval(self, X, coverage=None):
        """Return ``(lower, upper)`` at the requested coverage (defaults to ``self.coverage``)."""
        mean, u = self.predict(X)
        if coverage is not None and coverage != self.coverage:
            # Rescale the one-sigma-like width to the requested Gaussian coverage.
            from scipy import stats
            k = stats.norm.ppf(0.5 + coverage / 2) / stats.norm.ppf(0.5 + self.coverage / 2)
            u = u * k
        return mean - u, mean + u

    def _fit_difficulty(self, X_train, y_train):
        from sklearn.neighbors import NearestNeighbors
        self._nn = NearestNeighbors(n_neighbors=self.n_neighbors).fit(X_train)
        self._y_train = y_train

    def _difficulty(self, X):
        # Local label spread as a cheap difficulty proxy; +1 keeps the scale well-behaved.
        _, idx = self._nn.kneighbors(np.asarray(X))
        return self._y_train[idx].std(axis=1) + 1.0

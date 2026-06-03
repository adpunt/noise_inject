"""
Gaussian-process regression with per-sample uncertainty, via Gauche / GPyTorch.

This mirrors the exact-GP setup used in the accompanying study: a ``ConstantMean`` with
a ``ScaleKernel`` over a base kernel, trained by maximizing the marginal log-likelihood.
A GP yields a predictive standard deviation per sample for free, which is what feeds the
calibration metrics. Use the Tanimoto kernel for binary fingerprints (the Gauche default)
or an RBF kernel for continuous descriptors.

Requires the ``uncertainty`` extra (gpytorch, gauche, torch).
"""

import numpy as np


class GaucheGPRegressor:
    """
    Exact-GP regressor returning predictive mean and standard deviation.

    Args:
        kernel: 'tanimoto' for binary fingerprints (Gauche) or 'rbf' for continuous
                features. A GPyTorch kernel instance may also be passed directly.
        training_iter: Number of marginal-likelihood optimization steps.
        lr: Adam learning rate for hyperparameter fitting.
    """

    def __init__(self, kernel='tanimoto', training_iter=100, lr=0.1):
        self.kernel = kernel
        self.training_iter = training_iter
        self.lr = lr

    def _build_kernel(self):
        import gpytorch
        if not isinstance(self.kernel, str):
            return self.kernel
        if self.kernel == 'tanimoto':
            from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
            return TanimotoKernel()
        if self.kernel == 'rbf':
            return gpytorch.kernels.RBFKernel()
        raise ValueError(f"Unknown kernel '{self.kernel}' (use 'tanimoto', 'rbf', or a kernel instance)")

    def fit(self, X_train, y_train):
        """Fit the GP by maximizing the marginal log-likelihood."""
        import torch
        import gpytorch

        self._x = torch.as_tensor(np.asarray(X_train), dtype=torch.float32)
        self._y = torch.as_tensor(np.asarray(y_train).flatten(), dtype=torch.float32)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        kernel = self._build_kernel()

        class _ExactGP(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood, covar):
                super().__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(covar)

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(x), self.covar_module(x)
                )

        self.model = _ExactGP(self._x, self._y, self.likelihood, kernel)

        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for _ in range(self.training_iter):
            optimizer.zero_grad()
            loss = -mll(self.model(self._x), self._y)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X):
        """Return ``(mean, std)`` of the predictive distribution (including observation noise)."""
        import torch
        import gpytorch

        x = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(x))
            mean = pred.mean.numpy()
            std = pred.stddev.numpy()
        return mean, std

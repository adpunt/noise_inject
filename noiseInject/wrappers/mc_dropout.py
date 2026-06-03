"""
Monte Carlo dropout uncertainty for a trained PyTorch model.

Keeping dropout active at inference and averaging over many stochastic forward passes
approximates Bayesian predictive uncertainty (Gal & Ghahramani, 2016). The wrapper is
model-agnostic: it takes any ``torch.nn.Module`` containing dropout layers and reports
the per-sample mean and standard deviation across passes.
"""

import numpy as np


class MCDropoutRegressor:
    """
    Wrap a trained PyTorch regressor to produce MC-dropout uncertainty.

    Args:
        model: A trained ``torch.nn.Module`` with dropout layers; called as ``model(X)``.
        n_forward: Number of stochastic forward passes (default 100).
        device: Optional torch device string (e.g. 'cpu', 'cuda').
    """

    def __init__(self, model, n_forward=100, device=None):
        self.model = model
        self.n_forward = n_forward
        self.device = device

    def predict(self, X):
        """
        Return ``(mean, std)`` per sample over ``n_forward`` dropout-enabled passes.

        ``X`` may be a tensor or array-like; arrays are converted to a float tensor.
        """
        import torch

        if self.device is not None:
            self.model = self.model.to(self.device)

        if not torch.is_tensor(X):
            X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        if self.device is not None:
            X = X.to(self.device)

        self.model.train()  # keep dropout active; no grads needed for inference
        passes = []
        with torch.no_grad():
            for _ in range(self.n_forward):
                passes.append(self.model(X).cpu().numpy().reshape(len(X), -1))

        stacked = np.stack(passes, axis=0).squeeze(-1)  # (n_forward, n_samples)
        return stacked.mean(axis=0), stacked.std(axis=0)

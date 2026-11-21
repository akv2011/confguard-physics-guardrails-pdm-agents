"""Universal sensor perturbation (Basak et al.).

One additive perturbation delta of shape [window, n_features], bounded by
||delta||_inf <= epsilon, trained to maximise the regressor's squared error over
the training windows. The same delta is then added to every attacked window.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def universal_perturbation(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
                           epsilon: float = 0.05, n_iters: int = 500, batch_size: int = 64,
                           lr: float = 1e-3, seed: int = 0) -> torch.Tensor:
    """Return delta with shape X.shape[1:]."""
    torch.manual_seed(seed)
    model.eval()
    delta = torch.zeros_like(X[0], requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    it = iter(loader)
    for _ in range(n_iters):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(loader)
            xb, yb = next(it)
        loss = -F.mse_loss(model(xb + delta), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(-epsilon, epsilon)
    return delta.detach()


def apply_perturbation(X: torch.Tensor, delta: torch.Tensor, epsilon: float | None = None) -> torch.Tensor:
    d = delta.to(X.device)
    return X + (d.clamp(-epsilon, epsilon) if epsilon is not None else d)

"""Split and Mondrian conformal quantiles for a point RUL predictor.

The nonconformity score is r_i = |y_i - f(x_i)|. The (1 - alpha) split-conformal
quantile q_hat gives the interval [f(x) - q_hat, f(x) + q_hat], which covers y with
probability at least 1 - alpha under exchangeability. The Mondrian variant computes
one q_hat per group, for example per operating condition.
"""

from __future__ import annotations

import numpy as np


def split_conformal_quantile(cal_pred: np.ndarray, cal_truth: np.ndarray, alpha: float = 0.1) -> float:
    """Marginal split-conformal quantile of |y - f(x)| on a calibration set."""
    pred = np.asarray(cal_pred, dtype=np.float64).reshape(-1)
    truth = np.asarray(cal_truth, dtype=np.float64).reshape(-1)
    if pred.shape != truth.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {truth.shape}")
    if len(pred) == 0:
        raise ValueError("empty calibration set")
    r = np.abs(truth - pred)
    level = min(1.0, np.ceil((len(r) + 1) * (1 - alpha)) / len(r))
    return float(np.quantile(r, level, method="higher"))


def mondrian_conformal_quantiles(
    cal_pred: np.ndarray,
    cal_truth: np.ndarray,
    cal_group: np.ndarray,
    alpha: float = 0.1,
    min_samples_per_group: int = 50,
    fallback: float | None = None,
) -> dict[int, float]:
    """One q_hat per group; groups with too few samples use the pooled q_hat."""
    pred = np.asarray(cal_pred, dtype=np.float64).reshape(-1)
    truth = np.asarray(cal_truth, dtype=np.float64).reshape(-1)
    group = np.asarray(cal_group).reshape(-1)
    pooled = fallback if fallback is not None else split_conformal_quantile(pred, truth, alpha)
    out: dict[int, float] = {}
    for g in np.unique(group):
        m = group == g
        out[int(g)] = (split_conformal_quantile(pred[m], truth[m], alpha)
                       if m.sum() >= min_samples_per_group else float(pooled))
    return out


def empirical_coverage(pred: np.ndarray, truth: np.ndarray, q_hat: float) -> float:
    """Share of points whose interval [pred - q_hat, pred + q_hat] covers the truth."""
    pred = np.asarray(pred).reshape(-1)
    truth = np.asarray(truth).reshape(-1)
    return float((np.abs(truth - pred) <= q_hat).mean())

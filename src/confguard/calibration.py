"""Trajectory-level control of false abstention for the deployed clock monitor.

The deployed monitor admits a window only if it passes, so its history, and every
later score, depends on its own threshold. Windows of one asset are also strongly
dependent. Calibration therefore treats each whole clean trajectory as one
exchangeable unit.

Thresholds are lam * tau0, with tau0 a fixed shape (for example `clock.calibrate`)
and lam on a grid in [1, inf]. The loss of a trajectory is the share of its windows
the deployed monitor abstains on. That loss need not be monotone in lam, so it is
replaced by its upper envelope over lam' >= lam, and lam is chosen by conformal risk
control (Angelopoulos et al., ICLR 2024). For a fresh trajectory exchangeable with
the m calibration trajectories, the expected false-abstention share is at most rho.
With fewer than 1/rho - 1 calibration trajectories the rule returns lam = inf, a
monitor that never abstains.
"""

from __future__ import annotations

import math

import numpy as np

from .clock import ClockMonitor, Thresholds

GRID = np.r_[np.arange(1.0, 2.0, 0.02), np.arange(2.0, 4.0, 0.1), np.arange(4.0, 10.01, 0.5), np.inf]


def scaled(t0: Thresholds, lam: float) -> Thresholds:
    s = lambda v: math.inf if math.isinf(v) or math.isinf(lam) else v * lam
    return Thresholds(s(t0.tau_F), s(t0.tau_C), s(t0.tau_T))


def deployed_rate(f, q_hat: float, k: int, thr: Thresholds, qs=None, commit_every: int = 0) -> float:
    """Share of one clean trajectory's windows the deployed monitor abstains on.

    `qs`, if given, is a per-window conformal width (Mondrian calibration).
    """
    mon, abstained = ClockMonitor(q_hat, k, thr, commit_every), 0
    for t, ft in enumerate(f):
        if qs is not None:
            mon.q = float(qs[t])
        abstained += not mon.step(float(ft), t)
    return abstained / len(f)


def loss_matrix(trajectories, q_hat: float, k: int, t0: Thresholds, grid=GRID, commit_every: int = 0):
    """Per-trajectory loss on the grid, raw and as its upper envelope over lam' >= lam.

    Each trajectory is a dict with key "f" (predictions) and optional "qs".
    """
    L = np.array([[deployed_rate(tr["f"], q_hat, k, scaled(t0, g), tr.get("qs"), commit_every)
                   for g in grid] for tr in trajectories])
    return L, np.maximum.accumulate(L[:, ::-1], axis=1)[:, ::-1]


def pick(L_env: np.ndarray, rho: float, grid=GRID) -> float:
    """Smallest lam with (sum_i L_i(lam) + 1) / (m + 1) <= rho; inf if none."""
    m = len(L_env)
    ok = (L_env.sum(0) + 1.0) / (m + 1) <= rho
    return float(grid[int(np.argmax(ok))]) if ok.any() else math.inf


def risk_control(trajectories, q_hat: float, k: int, t0: Thresholds, rho: float = 0.05,
                 grid=GRID, commit_every: int = 0) -> tuple[float, Thresholds]:
    """Choose lam on clean calibration trajectories; return lam and the thresholds."""
    _, env = loss_matrix(trajectories, q_hat, k, t0, grid, commit_every)
    lam = pick(env, rho, grid)
    return lam, scaled(t0, lam)

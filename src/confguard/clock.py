"""Physics-clock monitor for remaining-useful-life (RUL) predictions.

Capped RUL is never negative, never rises, and falls by exactly one cycle per cycle
below the cap. A sensor attacker cannot change these facts about the asset. The
monitor keeps the last k predictions it admitted for one asset, carries them forward
by the clock, and scores a new prediction by how far it breaks that physics:

    s_F  floor:          the lower interval end falls below zero
    s_C  ceiling:        the upper interval end rises above the cap (optional)
    s_T  clock:          the prediction fell faster, or rose, relative to the
                         median of the admitted history carried forward in time

Each score is monotone in the prediction, so the predictions that pass form one
interval [lo, hi] (`ClockMonitor.admissible`). The strongest covert attacker places
its prediction at an end of that interval.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

RUL_CAP = 125.0
C_STAR = 100.0  # predictions at or above this level may legitimately sit at the cap


@dataclass(frozen=True)
class Thresholds:
    tau_F: float = math.inf
    tau_C: float = math.inf
    tau_T: float = math.inf


@dataclass(frozen=True)
class Scores:
    s_F: float
    s_C: float
    s_T: float

    def exceeds(self, thr: Thresholds) -> bool:
        return self.s_F > thr.tau_F or self.s_C > thr.tau_C or self.s_T > thr.tau_T


class ClockMonitor:
    """Monitor for one asset. Only admitted windows enter the history.

    `commit_every` > 0 freezes the anchors between inspection events: the reference
    is a snapshot of the history refreshed every `commit_every` cycles, so windows
    admitted between inspections cannot move it.
    """

    def __init__(self, q_hat: float, k: int, thr: Thresholds, commit_every: int = 0,
                 rul_cap: float = RUL_CAP, c_star: float = C_STAR):
        self.q, self.k, self.thr = q_hat, k, thr
        self.rul_cap, self.c_star = rul_cap, c_star
        self.hist: deque[tuple[float, float]] = deque(maxlen=max(k, 1))
        self.commit_every = commit_every
        self._snap: tuple[tuple[float, float], ...] = ()
        self._last_commit: float | None = None

    def _anchors(self, t: float) -> tuple[float, float] | None:
        if self.k == 0:
            return None
        src = self._snap if self.commit_every else tuple(self.hist)
        if not src:
            return None
        ti = np.array([h[0] for h in src], dtype=float)
        fi = np.array([h[1] for h in src], dtype=float)
        age = t - ti
        down = float(np.median(fi - age))
        # At the cap an asset may hold still, so only the one-cycle fall applies below it.
        up = float(np.median(np.where(fi < self.c_star, fi - age, fi)))
        return down, up

    def scores(self, f: float, t: float) -> Scores:
        s_F = max(0.0, self.q - f)
        s_C = max(0.0, f + self.q - self.rul_cap)
        a = self._anchors(t)
        s_T = 0.0 if a is None else max(0.0, a[0] - f, f - a[1])
        return Scores(s_F, s_C, s_T)

    def admissible(self, t: float) -> tuple[float, float]:
        """The interval of predictions that pass at time t."""
        lo, hi = -math.inf, math.inf
        if math.isfinite(self.thr.tau_F):
            lo = max(lo, self.q - self.thr.tau_F)
        if math.isfinite(self.thr.tau_C):
            hi = min(hi, self.rul_cap - self.q + self.thr.tau_C)
        a = self._anchors(t)
        if a is not None and math.isfinite(self.thr.tau_T):
            lo = max(lo, a[0] - self.thr.tau_T)
            hi = min(hi, a[1] + self.thr.tau_T)
        # Inward by 1e-7 so a prediction placed on the boundary is not lost to rounding.
        return lo + 1e-7, hi - 1e-7

    def passes(self, f: float, t: float) -> bool:
        return not self.scores(f, t).exceeds(self.thr)

    def admit(self, f: float, t: float) -> None:
        self.hist.append((t, float(f)))
        if self.commit_every and (self._last_commit is None
                                  or t - self._last_commit >= self.commit_every):
            self._snap, self._last_commit = tuple(self.hist), t

    def step(self, f: float, t: float) -> bool:
        """Deployed use: test the prediction and admit it only if it passes."""
        ok = self.passes(f, t)
        if ok:
            self.admit(f, t)
        return ok


def conformal_tau(values: np.ndarray, share: float) -> float:
    """The ceil((1 - share)(m + 1))-th smallest value; +inf when that rank exceeds m."""
    v = np.sort(np.asarray(values, dtype=float))
    rank = math.ceil((1.0 - share) * (len(v) + 1))
    return math.inf if rank > len(v) else float(v[rank - 1])


def clean_scores(trajectories, q_hat: float, k: int, commit_every: int = 0, times=None, **kw) -> np.ndarray:
    """Scores (s_F, s_C, s_T) along clean trajectories with every window admitted.

    `times` gives each trajectory's window times; it defaults to the window index.
    """
    rows = []
    for n, f in enumerate(trajectories):
        mon = ClockMonitor(q_hat, k, Thresholds(), commit_every, **kw)
        for i, ft in enumerate(f):
            t = i if times is None else float(times[n][i])
            s = mon.scores(float(ft), t)
            rows.append((s.s_F, s.s_C, s.s_T))
            mon.admit(float(ft), t)
    return np.array(rows)


def calibrate(trajectories, q_hat: float, k: int, rho: float, use_floor: bool = True,
              use_ceiling: bool = False, commit_every: int = 0, times=None, **kw) -> Thresholds:
    """Per-window threshold shape: the (1 - rho / n_active) quantile of each active score.

    These per-window quantiles are the shape tau0 that `calibration.risk_control`
    widens until the false-abstention rate of the deployed monitor is controlled.
    """
    active = [use_floor, use_ceiling, k > 0]
    share = rho / sum(active)
    S = clean_scores(trajectories, q_hat, k, commit_every, times, **kw)
    return Thresholds(
        tau_F=conformal_tau(S[:, 0], share) if use_floor else math.inf,
        tau_C=conformal_tau(S[:, 1], share) if use_ceiling else math.inf,
        tau_T=conformal_tau(S[:, 2], share) if k > 0 else math.inf,
    )

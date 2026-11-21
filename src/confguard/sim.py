"""Simulated fleets of clean RUL trajectories for testing calibration.

A trajectory has a lifetime of 128 to 362 cycles, true capped RUL, and a predictor
output with an AR(1) error plus one named degradation pattern. It starts at the first
full 30-cycle window and is either observed to failure or cut at a random point.
"""

from __future__ import annotations

import numpy as np

CAP = 125.0

PATTERNS = {
    "smooth": dict(sigma=13.0, phi=0.95),
    "noisy": dict(sigma=20.0, phi=0.70),
    "abrupt": dict(sigma=10.0, phi=0.90, abrupt=True),    # held at the cap, then drops
    "regimes": dict(sigma=8.0, phi=0.90, regimes=True),   # six operating-condition offsets
    "rate": dict(sigma=10.0, phi=0.90, rate=0.3),         # asset degrades faster or slower
}


def trajectory(rng: np.random.Generator, pattern: str = "smooth", truncated: bool = True) -> dict:
    p = PATTERNS[pattern]
    T = int(rng.integers(128, 363))
    y = np.minimum(CAP, T - 1 - np.arange(T, dtype=float))
    e = np.empty(T)
    e[0] = rng.normal(0, p["sigma"])
    z = rng.normal(0, p["sigma"] * np.sqrt(1 - p["phi"] ** 2), T)
    for t in range(1, T):
        e[t] = p["phi"] * e[t - 1] + z[t]
    base = y.copy()
    if p.get("rate"):
        base = CAP - (1 + rng.uniform(-p["rate"], p["rate"])) * (CAP - y)
    if p.get("abrupt"):
        base = np.where(y > rng.uniform(30, 90), CAP, y)
    if p.get("regimes"):
        levels = np.array([-12.0, -6.0, 0.0, 6.0, 12.0, 18.0])
        r, off = int(rng.integers(6)), np.empty(T)
        for t in range(T):
            if rng.random() < 0.05:
                r = int(rng.integers(6))
            off[t] = levels[r]
        base = base + off
    f = base + e
    start = 29
    end = T if not truncated else int(rng.integers(start + 31, T))
    return {"f": f[start:end], "y": y[start:end]}


def fleet(rng: np.random.Generator, n: int, pattern: str = "smooth", truncated: bool = True) -> list[dict]:
    return [trajectory(rng, pattern, truncated) for _ in range(n)]

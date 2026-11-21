"""Exact one-shot covert attacker against the clock monitor.

Every score is monotone in the prediction, so the monitor admits one interval
[lo, hi]. An attacker who must stay undetected and can move the prediction by at
most `budget` cycles does best at an end of that interval. A harmful class is
therefore reachable exactly when that end crosses the decision boundary.
"""

from __future__ import annotations

from ..clock import ClockMonitor


def false_alarm_reachable(mon: ClockMonitor, f_clean: float, t: float,
                          theta_danger: float, budget: float) -> bool:
    """Can a covert attacker push a healthy asset into MAINTAIN_NOW (l < theta_danger)?"""
    lo, _ = mon.admissible(t)
    return max(lo, f_clean - budget) < theta_danger + mon.q


def suppression_reachable(mon: ClockMonitor, f_clean: float, t: float,
                          theta_safe: float, budget: float) -> bool:
    """Can a covert attacker push a failing asset into NO_ACTION (l > theta_safe)?"""
    _, hi = mon.admissible(t)
    return min(hi, f_clean + budget) > theta_safe + mon.q

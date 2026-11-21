"""ConfGuard: physics-informed guardrails for agentic AI in predictive maintenance."""

from .calibration import risk_control
from .clock import ClockMonitor, Thresholds, calibrate
from .conformal import split_conformal_quantile
from .validator import decide, reference_policy

__version__ = "1.0.0"
__all__ = ["ClockMonitor", "Thresholds", "calibrate", "risk_control",
           "split_conformal_quantile", "reference_policy", "decide"]

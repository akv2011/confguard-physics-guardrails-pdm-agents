# ConfGuard: Physics-Informed Guardrails for Agentic AI in Predictive Maintenance

Reference implementation of ConfGuard, two deterministic checks that sit outside an
LLM maintenance agent. The physics-clock monitor tests every remaining-useful-life
(RUL) prediction against the asset's own degradation physics, with a threshold
calibrated over whole asset trajectories. The physics-policy validator checks the
agent's maintenance-urgency decision against the class a reference policy derives
from the same conformal interval.

## Layout

```
confguard/
├── conformal.py        # split and Mondrian conformal quantiles
├── clock.py            # physics-clock monitor, per-window threshold shape
├── calibration.py      # trajectory-level risk control of false abstention
├── validator.py        # reference policy, urgency projection, bounded retry
├── provenance.py       # capability policy over tool-call arguments
├── sim.py              # simulated fleets with five degradation patterns
├── models/
│   └── lstm_rul.py     # bidirectional LSTM RUL regressor
└── attacks/
    ├── universal.py    # universal sensor perturbation
    └── covert.py       # exact one-shot covert attacker on the clock monitor
```

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

Tested with Python 3.10 to 3.12 and `torch>=2.0`, CPU only.

## Minimal usage

Calibrate the monitor on simulated clean fleets and run it as deployed:

```python
import numpy as np

from confguard import ClockMonitor, calibrate, decide, risk_control
from confguard.calibration import deployed_rate
from confguard.sim import fleet

rng = np.random.default_rng(0)
q_hat, k, rho = 28.0, 30, 0.05

# Threshold shape from per-window quantiles on run-to-failure trajectories.
val = fleet(rng, 10, "smooth", truncated=False)
tau0 = calibrate([t["f"] for t in val], q_hat, k, rho)

# Widen it until the expected false-abstention share per asset is at most rho.
cal = fleet(rng, 50, "smooth")
lam, thr = risk_control(cal, q_hat, k, tau0, rho)

test = fleet(rng, 200, "smooth")
rates = [deployed_rate(t["f"], q_hat, k, thr) for t in test]
print(f"lambda={lam:.2f}  mean false abstention per asset={np.mean(rates):.3f}")

# Deployed use on one asset: a window enters the history only if it passes.
mon = ClockMonitor(q_hat, k, thr)
for t, f in enumerate(test[0]["f"]):
    if not mon.step(float(f), t):
        pass  # escalate this window to a human planner

# Validate an agent's tool call against the reference policy, with one retry.
call = {"name": "generate_work_order", "arguments": {"action": "maintain_now"}}
lower = 70.0 - q_hat
verdict, final = decide(call, lower, retry=lambda: {"arguments": {"action": "inspect"}})
print(verdict, final)
```

"""Physics-policy validator for the agent's maintenance decision.

The validator does not check a number against the interval. It checks the decision:
the urgency class of the agent's tool call must equal the class a deterministic
reference policy derives from the same conformal interval [l, u]. The policy reads
the pessimistic end l:

    MAINTAIN_NOW  if l < theta_danger
    NO_ACTION     if l > theta_safe
    INSPECT       otherwise

A rejected call gets one bounded retry; a retried call is validated again and goes
to a human if it still disagrees.
"""

from __future__ import annotations

import json
import re
from typing import Callable

MAINTAIN_NOW, INSPECT, NO_ACTION = "MAINTAIN_NOW", "INSPECT", "NO_ACTION"
MALFORMED, ESCALATE, DISPATCH = "MALFORMED", "ESCALATE", "DISPATCH"

URGENCY = {"maintain_now": MAINTAIN_NOW, "replace_bearing": MAINTAIN_NOW,
           "inspect": INSPECT, "schedule_overhaul": INSPECT,
           "no_action": NO_ACTION, "defer": NO_ACTION}


def reference_policy(lower: float, theta_danger: float = 25.0, theta_safe: float = 60.0) -> str:
    """Urgency class warranted by an interval with lower end `lower`."""
    return MAINTAIN_NOW if lower < theta_danger else NO_ACTION if lower > theta_safe else INSPECT


def parse_call(content: str) -> dict | None:
    """Parse a JSON tool call from model output, tolerating code fences."""
    if not content:
        return None
    for cand in (content, re.sub(r"^```[a-z]*|```$", "", content.strip(), flags=re.M)):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", content, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def projection(call: dict | None) -> str:
    """Urgency class of a tool call, or MALFORMED when none can be read."""
    if not isinstance(call, dict):
        return MALFORMED
    args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    action = str(args.get("action", call.get("action", ""))).strip().lower()
    return URGENCY.get(action, MALFORMED)


def validate(call: dict | None, lower: float, theta_danger: float = 25.0, theta_safe: float = 60.0) -> bool:
    return projection(call) == reference_policy(lower, theta_danger, theta_safe)


def decide(call: dict | None, lower: float, retry: Callable[[], dict | None] | None = None,
           theta_danger: float = 25.0, theta_safe: float = 60.0) -> tuple[str, dict | None]:
    """DISPATCH the call if it matches the reference policy, after at most one retry.

    `retry` is called once when the first call is rejected; it should re-ask the
    planner, for example with the warranted class stated. Otherwise ESCALATE.
    """
    if validate(call, lower, theta_danger, theta_safe):
        return DISPATCH, call
    if retry is not None:
        call = retry()
        if validate(call, lower, theta_danger, theta_safe):
            return DISPATCH, call
    return ESCALATE, call

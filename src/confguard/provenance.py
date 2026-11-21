"""Provenance monitor: a capability policy over tool-call arguments.

A deterministic check at the point the action takes effect: every part,
technician and service window in the call must be in the asset's trusted
maintenance record. It knows nothing about physics, so it cannot tell that a
remaining-life estimate is impossible; it complements the clock monitor and the
validator on the fields they do not cover.
"""

from __future__ import annotations

import random

FIELDS = ("parts", "technician", "service_window")


def build_catalog(asset_ids, seed: int = 0) -> dict:
    """A synthetic trusted record per asset: allowed parts, technicians and windows."""
    rng = random.Random(seed)
    catalog = {}
    for a in sorted(int(x) for x in asset_ids):
        catalog[str(a)] = {
            "parts": sorted(f"P-{rng.randrange(1000, 9999)}" for _ in range(2)),
            "technician": sorted(f"T-{rng.randrange(100, 999)}" for _ in range(2)),
            "service_window": sorted(rng.sample(
                ["2026-W41-A", "2026-W41-B", "2026-W42-A", "2026-W42-B", "2026-W43-A"], 2)),
        }
    return catalog


def check(call: dict, asset_id: str, catalog: dict) -> tuple[bool, str | None]:
    """Reject any argument outside the asset's trusted record."""
    rec = catalog.get(str(asset_id))
    if rec is None:
        return False, f"unknown asset {asset_id}"
    args = call.get("arguments", {}) if isinstance(call, dict) else {}
    parts = args.get("parts") or []
    if isinstance(parts, str):
        parts = [parts]
    for p in parts:
        if str(p) not in rec["parts"]:
            return False, f"part {p} not in catalogue"
    for field in ("technician", "service_window"):
        v = args.get(field)
        if v not in (None, "") and str(v) not in rec[field]:
            return False, f"{field} {v} not authorised"
    return True, None

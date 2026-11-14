"""Hazard index + training filter for the real PSI-159 CFI-G tests.

Copied from the Flame-in-Freefall data pipeline (pipeline/validate.py) so the ML
package is self-contained inside rag/ml.
hazard index: none = 0; self_extinguished = 15 + 30*min(burn,120)/120;
sustained (still burning at fuel cut-off) = 55 + 45*min(burn,300)/300; +10 for a
cool flame; capped at 100.  Levels: low < 20 <= medium < 45 <= high.
"""
from __future__ import annotations

import json

SE_BASE, SE_SPAN, SE_CAP_S = 15.0, 30.0, 120.0
FT_BASE, FT_SPAN, FT_CAP_S = 55.0, 45.0, 300.0
COOL_BONUS = 10.0


def risk_bucket(score: float) -> str:
    if score < 20:
        return "low"
    if score < 45:
        return "medium"
    return "high"


def hazard_score(outcome: str, burn_s: float | None, cool: bool) -> float:
    b = burn_s or 0.0
    if outcome == "sustained":
        s = FT_BASE + FT_SPAN * min(b, FT_CAP_S) / FT_CAP_S
    elif outcome == "self_extinguished":
        s = SE_BASE + SE_SPAN * min(b, SE_CAP_S) / SE_CAP_S
    else:
        s = 0.0
    if cool:
        s += COOL_BONUS
    return round(min(100.0, s), 1)


def ml_usable(rec: dict) -> bool:
    """Rows excluded from model training (kept in the knowledge base, just not learned from)."""
    flags = rec.get("data_flags") or []
    if isinstance(flags, str):
        flags = json.loads(flags)
    return not ({"zero_fuel_flow", "ignition_flag_conflict", "procedure_error"} & set(flags))

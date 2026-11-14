"""Shared feature + target definitions for the CFI-G flame models.

train.py and predict.py both import from here, so the saved artifact and the
inference code can never drift apart.

Data: NASA PSI-159 CFI-G "Normal Flames" (real ISS tests). Three questions:
  1. ignition   — will a hot flame ignite?                      (classifier)
  2. sustained  — once lit, will it still be burning when the    (classifier)
                  fuel flow is cut, instead of self-extinguishing?
  3. burn_time  — how long does an ignited flame burn (s)?        (regressor)
"""
from __future__ import annotations

import re

# Pre-test conditions only. Anything measured during/after the burn
# (burn_time_s, peak_burner_temp_c, termination, cool_flame, o2_after_pct) and
# the derived labels (outcome, risk_score) are excluded — they leak the answer.
# Tad/Zst are excluded too: they are computed from these inputs and are not
# available from a cabin sensor.
NUMERIC_FEATURES = ["o2_pct", "pressure_bar", "xf", "fuel_flow_mg_s", "fuel_bottle_pct"]
CATEGORICAL_FEATURES = ["fuel"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
FUELS = ("ethane", "propane", "butane")
LEAKY = {"burn_time_s", "peak_burner_temp_c", "termination", "cool_flame", "o2_after_pct",
         "outcome", "risk_score", "risk_level", "hot_ignition", "tad_k", "zst"}

# Range actually covered by the tests (used to flag extrapolation honestly).
DOMAIN = {
    "o2_pct": (17.9, 40.0),
    "pressure_bar": (0.5, 3.04),
    "xf": (0.018, 1.0),
    "fuel_flow_mg_s": (0.1, 3.3),
}
O2_GAP = (23.5, 33.0)   # no tests between these O2 levels

# Cabin-nominal defaults when a caller omits a field. fuel=None means
# "unknown" -> predictions are averaged over the three tested fuels.
DEFAULTS = {
    "o2_pct": 20.9,
    "pressure_bar": 1.0133,
    "xf": 0.3,
    "fuel_flow_mg_s": 1.0,
    "fuel_bottle_pct": 30.0,
    "fuel": None,
}

ALIASES = {
    "o2": "o2_pct", "oxygen": "o2_pct", "oxygen_pct": "o2_pct", "o2_pct": "o2_pct",
    "oxygen_concentration_pct": "o2_pct", "oxygen_before_pct": "o2_pct",
    "pressure_bar": "pressure_bar", "p_bar": "pressure_bar",
    "pressure_kpa": "pressure_kpa", "pressure": "pressure",
    "xf": "xf", "fuel_mole_fraction": "xf",
    "fuel_flow": "fuel_flow_mg_s", "fuel_flow_mg_s": "fuel_flow_mg_s", "mhc": "fuel_flow_mg_s",
    "fuel_bottle_pct": "fuel_bottle_pct", "bottle_pct": "fuel_bottle_pct",
    "fuel": "fuel", "material": "fuel", "fuel_mix": "fuel",
}


def parse_fuel(value) -> tuple[str | None, float | None]:
    """'30% butane' -> ('butane', 30); 'Propane' -> ('propane', None); other -> (None, None)."""
    if value is None:
        return None, None
    s = str(value).strip().lower()
    m = re.search(r"(?:(\d+)\s*%\s*)?(ethane|propane|butane)", s)
    if not m:
        return None, None
    return m.group(2), (float(m.group(1)) if m.group(1) else None)


def normalize_input(conditions: dict) -> tuple[dict, list[str]]:
    """Map aliases, apply defaults, validate numbers. Returns (row, warnings)."""
    row = dict(DEFAULTS)
    warnings: list[str] = []
    bottle_given = False
    for key, value in (conditions or {}).items():
        target = ALIASES.get(str(key).lower())
        if target is None or value is None or value == "":
            continue
        if target == "fuel":
            fuel, pct = parse_fuel(value)
            if fuel is None:
                warnings.append(f"'{value}' is not a tested fuel (ethane/propane/butane); "
                                "prediction averaged over the three tested fuels.")
            row["fuel"] = fuel
            if pct is not None:
                row["fuel_bottle_pct"] = pct
                bottle_given = True
            elif fuel is not None and not bottle_given and "fuel_bottle_pct" not in conditions:
                row["fuel_bottle_pct"] = 100.0
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number, got {value!r}")
        if target == "pressure_kpa":
            row["pressure_bar"] = num / 100.0
        elif target == "pressure":            # ambiguous: <= 10 -> bar, else kPa
            row["pressure_bar"] = num if num <= 10 else num / 100.0
        elif target == "fuel_bottle_pct":
            row["fuel_bottle_pct"] = num
            bottle_given = True
        else:
            row[target] = num
    row["pressure_bar"] = round(float(row["pressure_bar"]), 4)
    for col, (lo, hi) in DOMAIN.items():
        v = row[col]
        if v < lo or v > hi:
            warnings.append(f"{col}={v:g} is outside the tested range {lo:g}-{hi:g}; "
                            "this is an extrapolation.")
    if O2_GAP[0] < row["o2_pct"] < O2_GAP[1]:
        warnings.append(f"o2_pct={row['o2_pct']:g} falls in the untested gap "
                        f"{O2_GAP[0]:g}-{O2_GAP[1]:g}% (the tests cluster near 21% and 35-40%); "
                        "the prediction interpolates between those clusters.")
    return row, warnings

"""Load the trained CFI-G models and score conditions.

    from ml.predict import predict
    predict({"fuel": "30% butane", "o2": 34, "pressure_kpa": 56.5})
    # -> {"ignition": {...}, "sustained": {...}, "burn_time_s": ..., "hazard_index": ...,
    #     "level": "HIGH", "fire": True, "probability": 0.41, "riskBand": "HIGH", ...}

`probability` = P(hot ignition) x P(still burning when fuel is cut | ignited)
               = probability of a SUSTAINED flame.
`hazard_index` = expected value of the per-test hazard index used in the corpus
                 (see ml/scoring.py) under the predicted outcome mix.

CLI:
    python ml/predict.py --fuel "30% butane" --o2 34 --pressure-kpa 56.5
    python ml/predict.py --json '{"fuel":"propane","o2":21}'
    python ml/predict.py --info
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402

from ml.features import FEATURES, FUELS, normalize_input  # noqa: E402
from ml.scoring import hazard_score, risk_bucket  # noqa: E402

MODEL_PATH = ROOT / "ml" / "artifacts" / "fire_model.joblib"
NOTE = ("Trained on 175 real ISS tests (NASA PSI-159 CFI-G, gaseous fuel jets). "
        "Research/educational — not certified safety guidance.")


class ModelNotTrained(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load() -> dict:
    if not MODEL_PATH.exists():
        raise ModelNotTrained(f"{MODEL_PATH.relative_to(ROOT)} not found — run `python ml/train.py`.")
    bundle = joblib.load(MODEL_PATH)
    if "ignition" not in bundle:
        raise ModelNotTrained("Old model artifact (simulated data) — run `python ml/train.py`.")
    return bundle


def reload() -> None:
    _load.cache_clear()


def _score_row(bundle: dict, row: dict) -> dict:
    """Pure scoring of one normalized row. fuel=None -> mean over tested fuels."""
    fuels = [row["fuel"]] if row.get("fuel") in FUELS else list(FUELS)
    xs = np.array([[{**row, "fuel": f}[c] for c in FEATURES] for f in fuels], dtype=object)
    p_ign = float(np.mean(bundle["ignition"].predict_proba(xs)[:, 1]))
    p_sus = float(np.mean(bundle["sustained"].predict_proba(xs)[:, 1]))
    burn = float(np.mean(np.expm1(bundle["burn_time"].predict(xs))))
    burn = max(0.0, burn)
    hz = p_ign * (p_sus * hazard_score("sustained", burn, False)
                  + (1 - p_sus) * hazard_score("self_extinguished", burn, False))
    return {"p_ignition": round(p_ign, 4), "p_sustained": round(p_sus, 4),
            "burn_time_s": round(burn, 1), "hazard_index": round(hz, 1),
            "fuels_used": fuels}


O2_CABIN_CAVEAT = ("O2 levels above ~23% in these tests (33-40%) are ISS test-chamber conditions, "
                   "not a spacecraft cabin atmosphere (ISS air is ~21%).")


def predict(conditions: dict) -> dict:
    """Primary label: `predicted_end_mode` in the data-engineering vocabulary
    (no_ignition | self_extinguished | flow_terminated), with its probabilities.
    `hot_ignition`, `end_mode` and `burn_time_s` blocks use the same column names
    as psi159_clean.csv. Cool flames are NOT predicted (see model_info)."""
    bundle = _load()
    row, warnings = normalize_input(conditions)
    s = _score_row(bundle, row)
    p_i, p_s = s["p_ignition"], s["p_sustained"]
    probs = {"no_ignition": round(1 - p_i, 4),
             "self_extinguished": round(p_i * (1 - p_s), 4),
             "flow_terminated": round(p_i * p_s, 4)}
    label = max(probs, key=probs.get)
    p_fire = probs["flow_terminated"]
    level = risk_bucket(s["hazard_index"])
    band = {"low": "LOW", "medium": "MED", "high": "HIGH"}[level]
    caveats = []
    if row["o2_pct"] > 23.5:
        caveats.append(O2_CABIN_CAVEAT)
    return {
        "predicted_end_mode": label,
        "end_mode_probabilities": probs,
        "hot_ignition": {"probability": p_i, "predicted": p_i >= 0.5},
        "end_mode": {"p_flow_terminated_given_ignition": p_s,
                     "predicted_given_ignition": "flow_terminated" if p_s >= 0.5 else "self_extinguished"},
        "burn_time_s": s["burn_time_s"],
        "hazard_index": s["hazard_index"],
        "level": band,
        "cool_flame": {"predicted": None,
                       "note": "Not predicted: cool flames were only examined on 2 test days (9 positives)."},
        # legacy keys (v2 contract) kept so older clients don't break
        "ignition": {"probability": p_i, "label": "ignites" if p_i >= 0.5 else "no_ignition"},
        "sustained": {"probability_given_ignition": p_s,
                      "label": "keeps_burning" if p_s >= 0.5 else "self_extinguishes"},
        "fire": bool(p_fire >= 0.5),
        "label": "fire" if p_fire >= 0.5 else "no_fire",
        "probability": round(p_fire, 4),
        "riskBand": band,
        "threshold": 0.5,
        "model": bundle["model_names"],
        "inputs": row,
        "fuels_used": s["fuels_used"],
        "warnings": warnings,
        "caveats": caveats,
        "in_domain": not any("outside the tested range" in w or "untested gap" in w for w in warnings),
        "note": NOTE,
    }


def model_info() -> dict:
    b = _load()
    return {"targets": {
                "hot_ignition": "classifier — will a hot flame ignite? (column hot_ignition)",
                "end_mode": "classifier — given ignition: flow_terminated (still burning when the fuel was cut) "
                            "vs self_extinguished (column end_mode)",
                "burn_time_s": "regressor — burn time of an ignited flame, seconds (column burn_time_s)",
            },
            "primary_label": "predicted_end_mode: no_ignition | self_extinguished | flow_terminated",
            "not_predicted": {"cool_flame": "only examined on test days 21174 and 21291 (9 positives) — "
                                            "an empty cell elsewhere means 'not examined', so no valid label",
                              "peak_burner_temp_c": "measured during the burn (would leak the outcome)"},
            "models": b["model_names"], "features": b["features"], "fuels": b["fuels"],
            "trained_at": b["trained_at"], "sklearn_version": b["sklearn_version"],
            "metrics_headline": b["metrics_headline"],
            "evaluation": "leave-one-day-out cross-validation", "note": NOTE}


def predict_batch(items: list[dict]) -> list[dict]:
    return [predict(it) for it in items]


def _cli() -> int:
    ap = argparse.ArgumentParser(description="CFI-G flame model inference")
    ap.add_argument("--json")
    ap.add_argument("--fuel")
    ap.add_argument("--o2", type=float, help="ambient oxygen percent")
    ap.add_argument("--pressure-kpa", dest="pressure_kpa", type=float)
    ap.add_argument("--pressure-bar", dest="pressure_bar", type=float)
    ap.add_argument("--xf", type=float)
    ap.add_argument("--fuel-flow", dest="fuel_flow_mg_s", type=float)
    ap.add_argument("--info", action="store_true")
    a = ap.parse_args()
    if a.info:
        print(json.dumps(model_info(), indent=2))
        return 0
    cond = json.loads(a.json) if a.json else {k: v for k, v in vars(a).items()
                                              if v is not None and k not in ("json", "info")}
    print(json.dumps(predict(cond), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())

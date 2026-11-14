"""Train the three CFI-G flame models on the REAL ISS data and save one artifact.

    python ml/train.py

Honest evaluation: tests run in consecutive sequences on the same day (same gas
bottle, chamber O2 drifting test to test), so a random split would leak. The
headline metrics use LEAVE-ONE-DAY-OUT cross-validation (11 days -> 11 folds):
every prediction is made on a day the model never saw. A random 5-fold score is
reported next to it only to show how optimistic a naive split would be.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import sklearn  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,  # noqa: E402
                             roc_auc_score, brier_score_loss)
from sklearn.model_selection import (KFold, LeaveOneGroupOut, StratifiedKFold,  # noqa: E402
                                     cross_val_predict)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from ml.features import CATEGORICAL_FEATURES, FEATURES, FUELS, NUMERIC_FEATURES  # noqa: E402
from ml.scoring import ml_usable  # noqa: E402

# rag/ml/train.py -> project root is two levels above rag/
DATA_FILE = ROOT.parent / "Data" / "raw" / "psi159_experiments.json"
ART_DIR = ROOT / "ml" / "artifacts"
MODEL_PATH = ART_DIR / "fire_model.joblib"
METRICS_PATH = ART_DIR / "metrics.json"
SEED = 42


def load_records() -> list[dict]:
    """Real PSI-159 CFI-G tests, in the field names the models are trained on."""
    rows = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    for r in rows:
        r["hot_ignition"] = int(bool(r["hot_ignition"]))
        r["cool_flame"] = int(bool(r.get("cool_flame")))
    return sorted(rows, key=lambda r: r["experiment_id"])


def _X(recs: list[dict]) -> np.ndarray:
    return np.array([[r[f] for f in FEATURES] for r in recs], dtype=object)


def _pre(scale: bool) -> ColumnTransformer:
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", num, list(range(len(NUMERIC_FEATURES)))),
        ("cat", OneHotEncoder(categories=[list(FUELS)], handle_unknown="ignore"),
         list(range(len(NUMERIC_FEATURES), len(FEATURES)))),
    ])


def classifiers() -> dict:
    return {
        "logreg": Pipeline([("pre", _pre(True)),
                            ("clf", LogisticRegression(max_iter=2000, C=1.0,
                                                       class_weight="balanced"))]),
        "random_forest": Pipeline([("pre", _pre(False)),
                                   ("clf", RandomForestClassifier(
                                       n_estimators=400, min_samples_leaf=3, max_depth=6,
                                       class_weight="balanced", random_state=SEED))]),
    }


def regressors() -> dict:
    return {
        "ridge": Pipeline([("pre", _pre(True)), ("reg", Ridge(alpha=1.0))]),
        "random_forest": Pipeline([("pre", _pre(False)),
                                   ("reg", RandomForestRegressor(
                                       n_estimators=400, min_samples_leaf=3, max_depth=6,
                                       random_state=SEED))]),
    }


def _cls_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= 0.5).astype(int)
    out = {"n": int(len(y)), "positive_rate": round(float(y.mean()), 3),
           "accuracy": round(float(accuracy_score(y, pred)), 3),
           "f1": round(float(f1_score(y, pred, zero_division=0)), 3),
           "brier": round(float(brier_score_loss(y, p)), 3),
           "majority_baseline_accuracy": round(float(max(y.mean(), 1 - y.mean())), 3)}
    out["roc_auc"] = round(float(roc_auc_score(y, p)), 3) if len(set(y)) > 1 else None
    return out


def _eval_classifier(name: str, model, X, y, groups) -> dict:
    lodo = cross_val_predict(model, X, y, groups=groups, cv=LeaveOneGroupOut(),
                             method="predict_proba")[:, 1]
    rnd = cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            method="predict_proba")[:, 1]
    return {"model": name, "leave_one_day_out": _cls_metrics(y, lodo),
            "random_5fold_optimistic": _cls_metrics(y, rnd)}


def _eval_regressor(name: str, model, X, y_log, groups) -> dict:
    lodo = cross_val_predict(model, X, y_log, groups=groups, cv=LeaveOneGroupOut())
    rnd = cross_val_predict(model, X, y_log, cv=KFold(5, shuffle=True, random_state=SEED))
    y = np.expm1(y_log)

    def m(pred_log):
        pred = np.expm1(pred_log)
        return {"n": int(len(y)), "mae_s": round(float(mean_absolute_error(y, pred)), 1),
                "median_abs_error_s": round(float(np.median(np.abs(y - pred))), 1),
                "baseline_mae_s_predict_median": round(float(np.mean(np.abs(y - np.median(y)))), 1)}

    return {"model": name, "leave_one_day_out": m(lodo), "random_5fold_optimistic": m(rnd)}


def _pick(results: list[dict], key: str, higher_better: bool) -> dict:
    def score(r):
        v = r["leave_one_day_out"][key]
        return -math.inf if v is None else (v if higher_better else -v)
    return max(results, key=score)


def train() -> dict:
    recs = [r for r in load_records() if ml_usable(r)]
    groups_all = np.array([r["campaign_day"] for r in recs])

    # ---- 1) ignition ------------------------------------------------------
    X_ign = _X(recs)
    y_ign = np.array([int(r["hot_ignition"]) for r in recs])
    ign_results = [_eval_classifier(n, m, X_ign, y_ign, groups_all) for n, m in classifiers().items()]
    ign_best = _pick(ign_results, "roc_auc", True)

    # ---- 2) sustained | ignited ------------------------------------------
    lit = [r for r in recs if r["hot_ignition"] == 1 and r["termination"]]
    X_sus = _X(lit)
    y_sus = np.array([1 if r["outcome"] == "sustained" else 0 for r in lit])
    g_sus = np.array([r["campaign_day"] for r in lit])
    sus_results = [_eval_classifier(n, m, X_sus, y_sus, g_sus) for n, m in classifiers().items()]
    sus_best = _pick(sus_results, "roc_auc", True)

    # ---- 3) burn time | ignited ------------------------------------------
    timed = [r for r in lit if r["burn_time_s"] is not None and r["burn_time_s"] > 0]
    X_bt = _X(timed)
    y_bt = np.log1p(np.array([float(r["burn_time_s"]) for r in timed]))
    g_bt = np.array([r["campaign_day"] for r in timed])
    bt_results = [_eval_regressor(n, m, X_bt, y_bt, g_bt) for n, m in regressors().items()]
    bt_best = min(bt_results, key=lambda r: r["leave_one_day_out"]["mae_s"])

    # ---- fit winners on all usable data -----------------------------------
    ign_model = classifiers()[ign_best["model"]].fit(X_ign, y_ign)
    sus_model = classifiers()[sus_best["model"]].fit(X_sus, y_sus)
    bt_model = regressors()[bt_best["model"]].fit(X_bt, y_bt)

    # O2 sweep sanity check (pooled fuels, 1 atm, Xf 0.3, 1 mg/s) — logged in metrics
    from ml.predict import _score_row  # noqa: E402  (pure function, no artifact needed)
    bundle_tmp = {"ignition": ign_model, "sustained": sus_model, "burn_time": bt_model}
    sweep = []
    for o2 in (18, 21, 25, 30, 35, 40):
        row = {"o2_pct": float(o2), "pressure_bar": 1.0133, "xf": 0.3,
               "fuel_flow_mg_s": 1.0, "fuel_bottle_pct": 30.0, "fuel": None}
        s = _score_row(bundle_tmp, row)
        sweep.append({"o2_pct": o2, **{k: s[k] for k in ("p_ignition", "p_sustained",
                                                         "burn_time_s", "hazard_index")}})

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data": {"source": "NASA PSI-159 CFI-G Normal Flames (real ISS tests)",
                 "rows_total": len(load_records()), "rows_usable": len(recs),
                 "ignition_rows": len(recs), "sustained_rows": len(lit), "burn_time_rows": len(timed),
                 "days": int(len(set(groups_all)))},
        "evaluation": "leave-one-day-out CV (headline) + random 5-fold (optimistic reference)",
        "ignition": {"chosen": ign_best["model"], "candidates": ign_results},
        "sustained": {"chosen": sus_best["model"], "candidates": sus_results},
        "burn_time": {"chosen": bt_best["model"], "candidates": bt_results},
        "o2_sweep_pooled_fuels_1atm": sweep,
        "features": FEATURES,
    }

    ART_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "ignition": ign_model, "sustained": sus_model, "burn_time": bt_model,
        "model_names": {"ignition": ign_best["model"], "sustained": sus_best["model"],
                        "burn_time": bt_best["model"]},
        "features": FEATURES, "fuels": list(FUELS),
        "sklearn_version": sklearn.__version__, "trained_at": metrics["trained_at"],
        "metrics_headline": {
            "ignition_roc_auc_lodo": ign_best["leave_one_day_out"]["roc_auc"],
            "sustained_roc_auc_lodo": sus_best["leave_one_day_out"]["roc_auc"],
            "burn_time_mae_s_lodo": bt_best["leave_one_day_out"]["mae_s"],
        },
    }, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def line(task, best):
        l = best["leave_one_day_out"]
        r = best["random_5fold_optimistic"]
        if "roc_auc" in l:
            return (f"{task:10s} {best['model']:14s} LODO ROC-AUC={l['roc_auc']} acc={l['accuracy']} "
                    f"(baseline {l['majority_baseline_accuracy']}) | random-5fold ROC-AUC={r['roc_auc']}")
        return (f"{task:10s} {best['model']:14s} LODO MAE={l['mae_s']}s "
                f"(baseline {l['baseline_mae_s_predict_median']}s) | random-5fold MAE={r['mae_s']}s")

    print(f"usable rows: {len(recs)} / {metrics['data']['rows_total']}  "
          f"(ignited+terminated: {len(lit)}, with burn time: {len(timed)})")
    print(line("ignition", ign_best))
    print(line("sustained", sus_best))
    print(line("burn_time", bt_best))
    print("O2 sweep (pooled fuels, 1 atm):")
    for s in sweep:
        print(f"  O2 {s['o2_pct']:>2}%  P(ignite)={s['p_ignition']:.2f}  P(sustained|lit)={s['p_sustained']:.2f}"
              f"  burn~{s['burn_time_s']:.0f}s  hazard={s['hazard_index']:.0f}")
    print(f"saved -> {MODEL_PATH.relative_to(ROOT)} , {METRICS_PATH.relative_to(ROOT)}")
    return metrics


if __name__ == "__main__":
    train()

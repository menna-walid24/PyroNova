# MODEL_CARD — CFI-G flame models

## What it predicts
For a small gaseous fuel jet in microgravity under given conditions:
1. **P(ignition)** — will a hot flame ignite?
2. **P(sustained | lit)** — will it still be burning when the fuel flow is cut (instead of self-extinguishing)?
3. **Burn time (s)** of a lit flame.

Combined into `probability` = P(ignition) × P(sustained) and a **hazard index** (expected value of the corpus hazard
index, see DATA_QUALITY.md) with level LOW < 20 ≤ MED < 45 ≤ HIGH.

## Outputs & label names (v3.1, aligned with psi159_clean.csv columns)
| Key | Type | Meaning |
|---|---|---|
| `predicted_end_mode` | **primary label** | `no_ignition` \| `self_extinguished` \| `flow_terminated` (argmax of the three probabilities) |
| `end_mode_probabilities` | object | P(no_ignition)=1−P(ign); P(self_extinguished)=P(ign)·(1−P(FT\|ign)); P(flow_terminated)=P(ign)·P(FT\|ign) |
| `hot_ignition.probability` | 0–1 | classifier for column `hot_ignition` |
| `end_mode.p_flow_terminated_given_ignition` | 0–1 | classifier for column `end_mode` among ignited tests |
| `burn_time_s` | seconds | regressor for column `burn_time_s` (ignited tests) |
| `hazard_index` / `level` | 0–100 / LOW·MED·HIGH | expected hazard index |
| `cool_flame` | `predicted: null` | **not predicted** — only examined on 2 test days (9 positives) |
| `caveats` | list | O₂ > 23.5% → "test-chamber conditions, not cabin atmosphere" |

Legacy keys (`label` fire/no_fire, `probability`, `riskBand`, `ignition`, `sustained`) are still returned.
Inputs accept the psi159 column names: `fuel`, `oxygen_before_pct` (or `o2`), `pressure_kpa`, `fuel_flow_mg_s`, `xf`.

## Data
NASA PSI-159, CFI-G "Normal Flames" — 179 real ISS tests, 11 days. 175 usable (4 excluded: zero fuel flow, a
conflicting ignition flag, a procedural igniter error). Ignited + terminated: 132.

## Features (pre-test only — no leakage)
`o2_pct`, `pressure_bar`, `xf` (fuel mole fraction), `fuel_flow_mg_s`, `fuel_bottle_pct`, `fuel` (ethane/propane/butane, one-hot).
Excluded: anything measured during/after the burn (burn time, peak burner T, termination, cool flame, O₂ after) and
Tad/Zst (derived from inputs, not available from cabin sensors). Unknown gas → prediction averaged over the three fuels.

## Evaluation — leave-one-day-out (honest), random 5-fold shown only for contrast

| Task | Chosen | LODO | Random 5-fold |
|---|---|---|---|
| Ignition | random forest | ROC-AUC 0.77, accuracy 0.77 vs majority 0.75 | ROC-AUC 0.89 |
| Sustained \| lit | logistic regression | ROC-AUC 0.80, accuracy 0.75 vs majority 0.77 | ROC-AUC 0.81 |
| Burn time | random forest (log target) | MAE 29.6 s vs median-baseline 43.4 s | MAE 25.4 s |

The gap between LODO and random splits shows how much a naive split would overstate performance on new test days.
**Ignition prediction is weak** (barely above always-"yes"); sustained-burning and burn-time are the useful outputs.

## Sanity checks (`python ml/evaluate.py`)
30% butane at 1 atm: P(keeps burning) 0.03 at 21% O₂ → 0.62 at 38% O₂; hazard LOW → HIGH.
ISS air (21%, 101 kPa) hazard ≈ 15 LOW; exploration habitat (34%, 56.5 kPa) ≈ 49 HIGH.

## Limits
- 175 tests on one burner geometry, gaseous fuels only — not solids, not liquids.
- **No data between 23% and 33% O₂**: predictions there interpolate (a warning is returned).
- Outside 17.9–40% O₂, 0.5–3.04 bar, the model extrapolates (warning returned).
- Microgravity only. Not certified safety guidance.

## Use
```python
from ml.predict import predict
predict({"fuel": "30% butane", "o2": 34, "pressure_kpa": 56.5})
```
Retrain: `python ml/train.py` → `ml/artifacts/fire_model.joblib` + `metrics.json`.

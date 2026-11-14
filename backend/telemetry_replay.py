"""
Fire-risk status from REAL flight data: a replay of NASA SAFFIRE-III (PSI-100).

There is no live sensor feed on a spacecraft we can connect to, so the mobile
"FIRE RISK STATUS" card replays the 1,405 active seconds of the SAFFIRE-III
flight telemetry (data-engineering table saffire3_telemetry, is_baseline = 0).
The fire is visible in the radiometers: ~-16 (idle) rising to ~160 between
t = 5960 s and 6310 s of the flight. Units are not stated in the source, so
values are reported as raw signals.

Risk index (0-100) = radiometer rise above the idle level, scaled so the
flight's peak = 100; igniter energized -> at least 25. Levels: LOW < 20,
MEDIUM < 45, HIGH >= 45 (same thresholds as the experiment hazard index).
This is a demo derived from archived data, NOT live telemetry and NOT a
certified fire detector.

Environment:
  RISK_REPLAY_SPEED  replay speed multiplier (default 4 -> one loop ~6 min)
"""
from __future__ import annotations

import os
import sqlite3
import time
from functools import lru_cache
from pathlib import Path

DB_FILE = Path(__file__).resolve().parents[1] / "Data" / "raw" / "pyronova" / "pyronova_data.sqlite"
SOURCE = "NASA SAFFIRE-III (PSI-100) flight telemetry - replay of archived data, not a live feed"
DOI = "10.60555/040q-4h36"
_START = time.time()


@lru_cache(maxsize=1)
def _rows() -> list[dict]:
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT t_s, radiometer1, radiometer2, radiometer3, radiometer4, airflow1, "
        "fan_rpm_raw, igniter_on FROM saffire3_telemetry "
        "WHERE is_baseline = 0 ORDER BY t_s")]
    con.close()
    if not rows:
        raise RuntimeError("saffire3_telemetry has no active rows")
    for r in rows:
        r["radiant"] = max(r["radiometer1"], r["radiometer2"], r["radiometer3"], r["radiometer4"])
    radiants = sorted(r["radiant"] for r in rows)
    idle = radiants[len(radiants) // 2]          # median = instrument idle level
    peak = radiants[-1]
    for r in rows:
        idx = round(100 * max(0.0, r["radiant"] - idle) / max(peak - idle, 1e-9))
        if r["igniter_on"]:
            idx = max(idx, 25)
        r["index"] = int(min(100, idx))
    return rows


def _level(index: int) -> str:
    return "HIGH" if index >= 45 else "MEDIUM" if index >= 20 else "LOW"


def _trend_text(rows, i) -> str:
    now = rows[i]["index"]
    past = rows[max(0, i - 20)]["index"]
    if rows[i]["igniter_on"]:
        return "Igniter energized in the replayed flight."
    if now >= 45:
        return "Burning: strong radiant heat in the replayed flight."
    if now - past >= 10:
        return "Rising radiant heat."
    if past - now >= 10:
        return "Fire dying down."
    return "Stable, no flame detected." if now < 20 else "Elevated radiant heat."


def current(speed: float | None = None) -> dict:
    rows = _rows()
    speed = speed or float(os.environ.get("RISK_REPLAY_SPEED", "4"))
    n = len(rows)
    i = int((time.time() - _START) * speed) % n
    r = rows[i]
    trend = [rows[(i - k * 10) % n]["index"] / 100 for k in range(9, -1, -1)]
    index = r["index"]
    return {
        "index": index,
        "level": _level(index),
        "radiant_signal": round(r["radiant"], 1),
        "airflow": round(r["airflow1"], 2),
        "fan_rpm": round(r["fan_rpm_raw"]),
        "igniter_on": bool(r["igniter_on"]),
        "o2": None,                      # not measured by SAFFIRE-III
        "pressure": None,                # pressure_signal is empty in this flight
        "watch": "SAFFIRE-III",
        "trend": [round(x, 3) for x in trend],
        "trend_text": _trend_text(rows, i),
        "updated_seconds_ago": 0,
        "flight_time_s": r["t_s"],
        "replay": {"speed": speed, "position": i, "length": n},
        "units_note": "Raw signals; units are not stated in the NASA source.",
        "source": SOURCE,
        "doi": DOI,
        "is_live": False,
    }


def timeline(step: int = 10) -> list[dict]:
    return [{"flight_time_s": r["t_s"], "index": r["index"], "radiant_signal": round(r["radiant"], 1),
             "igniter_on": bool(r["igniter_on"])} for r in _rows()[::max(1, step)]]

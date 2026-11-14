"""
Experiment explorer (web page /experiments) — AI service side.

Everything comes from the REAL NASA PSI data prepared by the data-engineering team
(Data/raw/pyronova/pyronova_data.sqlite): 24 investigations with their official
objective / approach / hypothesis / Earth-benefits text, 352 publications, and the
flight-test tables (BASS, BASS-II, SAFFIRE-III) plus our PSI-159 CFI-G tests.

The relevance score is a TRANSPARENT HEURISTIC (every point is listed in
`score_breakdown`), not an official NASA ranking.

AI part: a short plain-language summary per investigation, written by the LLM ONLY
from the official NASA text, cached in Data/cache/experiment_summaries.json
(build it once with `python -m backend.experiments --build-summaries`).
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_FILE = ROOT / "Data" / "raw" / "pyronova" / "pyronova_data.sqlite"
PSI159_FILE = ROOT / "Data" / "raw" / "psi159_experiments.json"
CACHE_FILE = ROOT / "Data" / "cache" / "experiment_summaries.json"

ENVIRONMENTS = {
    "cabin": "Spacecraft cabin",
    "lunar": "Lunar base",
    "mars": "Mars habitat",
}
GOALS = ("flammability", "suppression", "detection", "fundamentals")

# Goal keywords searched in acronym + title + sub_area + objective (lower case).
_HIGH_O2 = r"oxygen[- ]enriched|elevated oxygen|high oxygen|enriched oxygen|\b(3\d|4\d)\s*%\s*(o2|oxygen)"
_PARTIAL_G = r"partial[- ]gravity|lunar|martian|mars|electric field|buoyan"
_LOW_P = r"reduced pressure|low pressure|sub-?atmospheric"

# Flight-test tables loaded in this system (counts are computed from the data).
_DATA_TABLES = {"PSI-26": "bass_tests", "PSI-25": "bass2_tests", "PSI-100": "saffire3_telemetry"}


def _con():
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    return con


@lru_cache(maxsize=1)
def _psi159() -> list[dict]:
    if PSI159_FILE.exists():
        return json.loads(PSI159_FILE.read_text(encoding="utf-8"))
    return []


@lru_cache(maxsize=1)
def investigations() -> list[dict]:
    con = _con()
    rows = [dict(r) for r in con.execute("SELECT * FROM investigations ORDER BY id")]
    pubs: dict[str, list[dict]] = {}
    for p in con.execute("SELECT id, investigation_id, title, authors, doi FROM publications ORDER BY id"):
        pubs.setdefault(p["investigation_id"], []).append(dict(p))
    con.close()
    for r in rows:
        r["publications"] = pubs.get(r["id"], [])
        r["name"] = r["acronym"] or r["id"]
        for k in ("total_gb", "files_listed", "tabular_files", "publications_count"):
            try:
                r[k] = float(r[k]) if k == "total_gb" else int(float(r[k]))
            except (TypeError, ValueError):
                r[k] = None
    return rows


def _text(inv: dict) -> str:
    return " ".join(str(inv.get(k) or "") for k in
                    ("acronym", "title", "sub_area", "objective", "approach", "hypothesis")).lower()


def goals_of(inv: dict) -> list[str]:
    head = " ".join(str(inv.get(k) or "") for k in ("acronym", "title", "sub_area")).lower()
    obj = str(inv.get("objective") or "").lower()
    found = []
    if re.search(r"flammab|flame spread|\bsolids?\b|saffire|burning rate|surface combustion", head) \
            or "flammab" in obj:
        found.append("flammability")
    if re.search(r"suppress|extinguish", head + " " + obj):
        found.append("suppression")
    if re.search(r"smoke|aerosol|dust", head) or re.search(r"detector|smoke detection|fire detection", obj):
        found.append("detection")
    return found or ["fundamentals"]


def _first_sentence(text: str | None, limit: int = 220) -> str:
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text).strip()
    m = re.match(r"(.+?[.!?])(\s|$)", t)
    s = m.group(1) if m else t
    return s if len(s) <= limit else s[:limit].rsplit(" ", 1)[0] + "…"


@lru_cache(maxsize=1)
def _data_stats() -> dict[str, dict]:
    """Real numbers for investigations whose test tables are loaded."""
    con = _con()
    out = {}
    b = con.execute("SELECT COUNT(*) n, SUM(ignited) ig, SUM(spread) sp, SUM(extinguished) ex, "
                    "MIN(o2_pct) o2min, MAX(o2_pct) o2max, COUNT(DISTINCT material) mats FROM bass_tests").fetchone()
    out["PSI-26"] = {"table": "bass_tests", "tests": b["n"], "ignited": b["ig"], "spread": b["sp"],
                     "extinguished": b["ex"], "o2_range_pct": [b["o2min"], b["o2max"]], "materials": b["mats"],
                     "note": "ignited/spread/extinguished are keywords found in the crew observation text, not NASA labels"}
    b2 = con.execute("SELECT COUNT(*) n, COUNT(DISTINCT material) mats, MIN(o2_initial_pct) o2min, "
                     "MAX(o2_initial_pct) o2max FROM bass2_tests").fetchone()
    out["PSI-25"] = {"table": "bass2_tests", "tests": b2["n"], "materials": b2["mats"],
                     "o2_range_pct": [b2["o2min"], b2["o2max"]]}
    s3 = con.execute("SELECT COUNT(*) n, SUM(is_baseline = 0) active FROM saffire3_telemetry").fetchone()
    out["PSI-100"] = {"table": "saffire3_telemetry", "rows": s3["n"], "active_seconds": s3["active"],
                      "note": "one real flight; radiometers show the fire between t=5960 s and 6310 s"}
    con.close()
    rows = _psi159()
    if rows:
        lit = [r for r in rows if r.get("hot_ignition")]
        kept = [r for r in lit if r.get("end_mode") == "flow_terminated"]
        o2 = [r.get("o2_pct", r.get("oxygen_before_pct")) for r in rows]
        out["PSI-159"] = {"table": "psi159_experiments", "tests": len(rows), "ignited": len(lit),
                          "still_burning_at_fuel_cutoff": len(kept), "o2_range_pct": [min(o2), max(o2)],
                          "fuels": sorted({r["fuel"] for r in rows}),
                          "note": "33-40% O2 are ISS test-chamber conditions, not cabin air (~21%)"}
    return out


def score(inv: dict, environment: str = "cabin") -> tuple[int, list[dict]]:
    """Transparent heuristic. Returns (score 0-100, list of {reason, points})."""
    t = _text(inv)
    goals = goals_of(inv)
    parts: list[dict] = []

    def add(reason: str, pts: float):
        if pts:
            parts.append({"reason": reason, "points": round(pts)})

    fire_safety = bool(re.search(r"fire safety|spacecraft fire|smoke", str(inv.get("sub_area") or "").lower()))
    add("Base score", 35)
    add("Listed under spacecraft fire safety / smoke by NASA PSI", 20 if fire_safety else 0)
    best_goal = max((15 if g == "flammability" else 12 if g in ("suppression", "detection") else 0) for g in goals)
    add(f"Directly informs crew fire safety ({', '.join(g for g in goals if g != 'fundamentals') or 'fundamentals'})",
        best_goal)
    add("Real flight-test data loaded in this system", 10 if inv["id"] in _data_stats() else 0)
    add("Publications on record", min(8, (inv.get("publications_count") or 0) / 6))

    high_o2 = bool(re.search(_HIGH_O2, t)) or inv["id"] == "PSI-159"
    partial_g = bool(re.search(_PARTIAL_G, t))
    low_p = bool(re.search(_LOW_P, t))
    platform = str(inv.get("platform") or "").lower()
    if environment == "cabin":
        add("Cabin: smoke / detection work for enclosed stations", 8 if "detection" in goals else 0)
        add("Cabin: flown in a crewed-vehicle environment (ISS / Cygnus)", 2 if ("iss" in platform or "cygnus" in platform) else 0)
    elif environment == "lunar":
        add("Lunar: high-oxygen test data (exploration atmospheres are oxygen-enriched)", 10 if high_o2 else 0)
        add("Lunar: partial-gravity / buoyancy proxy", 6 if partial_g else 0)
        add("Lunar: reduced-pressure conditions", 4 if low_p else 0)
    elif environment == "mars":
        add("Mars: partial-gravity / buoyancy proxy", 8 if partial_g else 0)
        add("Mars: high-oxygen test data", 6 if high_o2 else 0)
        add("Mars: reduced-pressure conditions", 4 if low_p else 0)
    total = int(min(100, round(sum(p["points"] for p in parts))))
    return total, parts


def _summaries() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _card(inv: dict, environment: str) -> dict:
    s, breakdown = score(inv, environment)
    ai = _summaries().get(inv["id"])
    tags = [g.capitalize() for g in goals_of(inv)]
    if inv.get("sub_area"):
        tags.append(inv["sub_area"])
    if inv.get("hardware"):
        tags.append(inv["hardware"].split(",")[0].strip())
    return {
        "id": inv["id"],
        "title": inv["name"],
        "subtitle": inv.get("title"),
        "desc": (ai or {}).get("summary") or _first_sentence(inv.get("objective")),
        "desc_source": "ai_summary_of_nasa_text" if ai else "nasa_objective",
        "relevance": s,
        "score_breakdown": breakdown,
        "goals": goals_of(inv),
        "tags": tags[:4],
        "platform": inv.get("platform"),
        "start_date": inv.get("start_date"),
        "end_date": inv.get("end_date"),
        "publications_count": inv.get("publications_count"),
        "doi": inv.get("doi"),
        "url": inv.get("url"),
        "has_test_data": inv["id"] in _data_stats(),
    }


def list_experiments(environment: str = "cabin", goal: str = "all", q: str = "") -> dict:
    environment = environment if environment in ENVIRONMENTS else "cabin"
    q = (q or "").strip().lower()
    cards = []
    for inv in investigations():
        if goal and goal != "all" and goal not in goals_of(inv):
            continue
        if q and q not in _text(inv) and q not in inv["id"].lower():
            continue
        cards.append(_card(inv, environment))
    cards.sort(key=lambda c: (-c["relevance"], c["title"]))
    for i, c in enumerate(cards, 1):
        c["rank"] = i
    return {
        "environment": environment,
        "environment_label": ENVIRONMENTS[environment],
        "goal": goal or "all",
        "count": len(cards),
        "total": len(investigations()),
        "items": cards,
        "ranking_note": ("Transparent heuristic, not an official NASA ranking: base score + fire-safety "
                         "relevance + real test data + publications + environment adjustments "
                         "(see score_breakdown on each item)."),
        "source": "NASA Physical Sciences Informatics (psi.nasa.gov), CC0",
    }


def get_experiment(inv_id: str, environment: str = "cabin") -> dict | None:
    inv = next((i for i in investigations() if i["id"].upper() == inv_id.upper()), None)
    if not inv:
        return None
    card = _card(inv, environment)
    card.update({
        "objective": inv.get("objective"),
        "approach": inv.get("approach"),
        "hypothesis": inv.get("hypothesis"),
        "earth_benefits": inv.get("earth_benefits"),
        "hardware": inv.get("hardware"),
        "center": inv.get("center"),
        "related": [r.strip() for r in str(inv.get("related") or "").split(",") if r.strip()],
        "publications": inv["publications"],
        "data": _data_stats().get(inv["id"]),
        "ai_summary": _summaries().get(inv["id"]),
    })
    return card


def compare(ids: list[str], environment: str = "cabin") -> dict:
    items = [get_experiment(i, environment) for i in ids[:3]]
    items = [i for i in items if i]
    fields = ["relevance", "goals", "platform", "start_date", "end_date", "publications_count", "has_test_data", "doi"]
    return {
        "environment": environment,
        "items": [{k: it.get(k) for k in ["id", "title", "subtitle", "desc", "data", "score_breakdown", *fields]}
                  for it in items],
        "rows": [{"field": f, "values": [it.get(f) for it in items]} for f in fields],
        "note": "Investigations differ in hardware, fuel and conditions; compare them as research programs, "
                "not as controlled experiments.",
    }


def earth_impact() -> dict:
    items = []
    for inv in investigations():
        if inv.get("earth_benefits"):
            ai = _summaries().get(inv["id"]) or {}
            items.append({
                "id": inv["id"], "title": inv["name"], "subtitle": inv.get("title"),
                "earth_benefits": inv["earth_benefits"],
                "short": ai.get("earth_impact") or _first_sentence(inv["earth_benefits"], 260),
                "doi": inv.get("doi"), "url": inv.get("url"),
            })
    return {"count": len(items), "items": items,
            "source": "Official 'Earth benefits' text of each NASA PSI investigation"}


# ------------------------------------------------------------------ AI summaries
def build_summaries(force: bool = False) -> dict:
    """LLM writes a 1-2 sentence plain summary + Earth impact line from the NASA text only."""
    sys.path.insert(0, str(ROOT / "rag"))
    from openai import OpenAI          # noqa: E402
    from llm.llm_client import LLMClient  # noqa: E402  (reads rag/.env)

    llm = LLMClient()
    cache = {} if force else _summaries()
    schema = {"type": "object", "additionalProperties": False,
              "required": ["summary", "crew_safety_relevance", "earth_impact"],
              "properties": {"summary": {"type": "string"}, "crew_safety_relevance": {"type": "string"},
                             "earth_impact": {"type": "string"}}}
    for inv in investigations():
        if inv["id"] in cache:
            continue
        source = "\n".join(f"{k.upper()}: {inv.get(k)}" for k in
                           ("title", "sub_area", "objective", "approach", "earth_benefits") if inv.get(k))
        prompt = ("Using ONLY the NASA text below, write JSON with: summary (max 2 short sentences, plain "
                  "English, what was studied), crew_safety_relevance (1 sentence, only if the text supports it, "
                  "else 'Not stated in the source.'), earth_impact (1 sentence from the Earth-benefits text, else "
                  "'Not stated in the source.'). Do not add facts or numbers that are not in the text.\n\n" + source)
        for model in (llm.model, llm.fallback_model):
            try:
                r = llm.client.chat.completions.create(
                    model=model, temperature=0.1, messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_schema",
                                     "json_schema": {"name": "inv_summary", "strict": True, "schema": schema}})
                out = json.loads(r.choices[0].message.content)
                out["model"] = model
                out["generated_from"] = "NASA PSI investigation text"
                cache[inv["id"]] = out
                print("ok", inv["id"], model)
                break
            except Exception as exc:  # try the fallback model
                print("fail", inv["id"], model, str(exc)[:120])
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
    return cache


if __name__ == "__main__":
    if "--build-summaries" in sys.argv:
        print(len(build_summaries(force="--force" in sys.argv)), "summaries cached in", CACHE_FILE)
    else:
        for env in ENVIRONMENTS:
            top = list_experiments(env)["items"][:5]
            print(env, [(c["title"], c["relevance"]) for c in top])

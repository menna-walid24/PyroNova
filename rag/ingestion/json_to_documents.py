"""
Build the knowledge base from the REAL NASA data.

Input : Data/raw/psi159_experiments.json   (179 ISS tests, NASA PSI-159 CFI-G Normal Flames,
                                            cleaned by the data-engineering team: psi159_clean.csv)
Output: knowledge_base/experiments/CFI-xxxxx-yy.md   (one Markdown document per test)

Old synthetic files (EXP-0001 ... EXP-0300) are deleted from the output folder first,
so the retriever can never mix old and new data.

Run:  python rag/ingestion/json_to_documents.py      (from any folder)
"""

import json
from pathlib import Path


# ============================================================
# Paths (relative to the project root, works on every machine)
# ============================================================

def find_project_root(start=None):
    start = Path(start or __file__).resolve()
    for p in [start, *start.parents]:
        if (p / "rag" / "pipeline.py").exists():
            return p
    raise FileNotFoundError("Project root (folder containing rag/pipeline.py) not found.")


PROJECT_ROOT = find_project_root()
INPUT_FILE = PROJECT_ROOT / "Data" / "raw" / "psi159_experiments.json"
OUTPUT_DIR = PROJECT_ROOT / "knowledge_base" / "experiments"

COOL_FLAME_DAYS = {"21174", "21291"}
O2_CAVEAT = ("The 33-40% oxygen levels are ISS test-chamber conditions, "
             "not a spacecraft cabin atmosphere (ISS cabin air is about 21%).")


# ============================================================
# Helpers
# ============================================================

def _fmt(value, unit="", digits=None):
    if value is None or value == "":
        return "not recorded"
    if digits is not None and isinstance(value, (int, float)):
        value = round(float(value), digits)
    return f"{value}{(' ' + unit) if unit else ''}"


def _day(exp):
    return exp["experiment_id"].split("-")[1]


def _cool_flame_text(exp):
    if exp.get("cool_flame"):
        return "yes"
    if _day(exp) in COOL_FLAME_DAYS:
        return "no"
    return "not examined (cool flames were only examined on test days 21174 and 21291)"


def _outcome_text(exp):
    if not exp.get("hot_ignition"):
        return "the hot flame did not ignite"
    if exp.get("end_mode") == "flow_terminated":
        return (f"the hot flame ignited and was still burning when the fuel flow was cut "
                f"after {_fmt(exp.get('burn_time_s'), 's')}")
    return (f"the hot flame ignited and self-extinguished after "
            f"{_fmt(exp.get('burn_time_s'), 's')}")


# ============================================================
# One experiment -> Markdown
# ============================================================

def experiment_to_markdown(exp):
    o2 = exp["o2_pct"]
    enriched = o2 >= 25
    flags = ", ".join(exp.get("data_flags") or []) or "none"
    comments = exp.get("comments") or "none"
    end_mode = exp.get("end_mode") or "none"

    return f"""---
experiment_id: {exp["experiment_id"]}
psi159_id: {exp["psi159_id"]}
source_investigation: {exp["source_investigation"]}
test_date: {exp["test_date"]}
fuel: {exp["fuel"]}
fuel_mix: {exp["fuel_mix"]}
oxygen_before_pct: {o2}
oxygen_band: {"enriched" if enriched else "air_like"}
pressure_kpa: {exp["pressure_kpa"]}
fuel_flow_mg_s: {exp["fuel_flow_mg_s"]}
xf: {exp["xf"]}
fuel_bottle_pct: {exp["fuel_bottle_pct"]}
hot_ignition: {str(bool(exp["hot_ignition"])).lower()}
burn_time_s: {exp["burn_time_s"] if exp["burn_time_s"] is not None else ""}
end_mode: {end_mode}
peak_burner_temp_c: {exp["peak_burner_temp_c"] if exp["peak_burner_temp_c"] is not None else ""}
cool_flame: {str(bool(exp["cool_flame"])).lower()}
outcome: {exp["outcome"]}
risk_score: {exp["risk_score"]}
risk_level: {exp["risk_level"]}
---

# Experiment {exp["experiment_id"]}

## Investigation

- **Source Investigation:** {exp["source_investigation"]} — NASA Cool Flames Investigation with Gases, normal (fuel-into-oxidizer) gaseous diffusion flames
- **Platform:** International Space Station (microgravity)
- **PSI-159 Test ID:** {exp["psi159_id"]}
- **Test Date:** {exp["test_date"]}

## Fuel

- **Fuel:** {exp["fuel"]}
- **Fuel Mix:** {exp["fuel_mix"]}
- **Fuel Mole Fraction (Xf):** {exp["xf"]}
- **Fuel Bottle Concentration:** {exp["fuel_bottle_pct"]} %

## Experimental Conditions

- **Oxygen Concentration (before test):** {o2} %
- **Oxygen Concentration (after test):** {_fmt(exp.get("o2_after_pct"), "%")}
- **Oxygen Band:** {"enriched (33-40% test-chamber condition)" if enriched else "air-like (18-23%)"}
- **Pressure:** {exp["pressure_kpa"]} kPa ({exp["pressure_bar"]} bar)
- **Fuel Flow:** {exp["fuel_flow_mg_s"]} mg/s
- **Adiabatic Flame Temperature:** {_fmt(exp.get("tad_k"), "K")}

## Observed Results

- **Hot Ignition:** {"yes" if exp["hot_ignition"] else "no"}
- **Burn Time:** {_fmt(exp.get("burn_time_s"), "s")}
- **End Mode:** {end_mode}
- **Peak Burner Temperature:** {_fmt(exp.get("peak_burner_temp_c"), "°C")}
- **Cool Flame:** {_cool_flame_text(exp)}
- **Outcome:** {exp["outcome"]}
- **Data Quality Flags:** {flags}
- **Crew Comments:** {comments}

## Fire Risk

- **Hazard Index (Risk Score):** {exp["risk_score"]} of 100
- **Risk Level:** {exp["risk_level"]}

## Experimental Summary

Experiment {exp["experiment_id"]} is a real ISS test from {exp["source_investigation"]}
run on {exp["test_date"]} with {exp["fuel_mix"]} as fuel.

The test used an oxygen concentration of {o2}%, a pressure of
{exp["pressure_kpa"]} kPa and a fuel flow of {exp["fuel_flow_mg_s"]} mg/s.

Result: {_outcome_text(exp)}. Peak burner temperature was
{_fmt(exp.get("peak_burner_temp_c"), "°C")}. Cool flame: {_cool_flame_text(exp)}.

The hazard index is {exp["risk_score"]} of 100, a **{exp["risk_level"]}** risk level.
{("Note: " + O2_CAVEAT) if enriched else ""}
"""



# ============================================================
# Dataset-level summary documents
# ============================================================
# Single-test documents cannot answer broad questions ("which fuel is most
# dangerous?", "what happens at 21% oxygen?"), so a few summary documents with
# statistics computed from ALL 179 tests are added to the knowledge base.

def _stats(rows):
    lit = [r for r in rows if r["hot_ignition"]]
    kept = [r for r in lit if r.get("end_mode") == "flow_terminated"]
    burns = sorted(r["burn_time_s"] for r in lit if r.get("burn_time_s") is not None)
    med = burns[len(burns) // 2] if burns else None
    temps = [r["peak_burner_temp_c"] for r in rows if r.get("peak_burner_temp_c") is not None]
    hz = [r["risk_score"] for r in rows]
    return {
        "n": len(rows), "ignited": len(lit), "still_burning_at_cutoff": len(kept),
        "self_extinguished": len(lit) - len(kept),
        "median_burn_s": med, "max_burn_s": burns[-1] if burns else None,
        "max_burner_c": max(temps) if temps else None,
        "mean_hazard": round(sum(hz) / len(hz), 1) if hz else None,
        "o2_min": min(r["o2_pct"] for r in rows), "o2_max": max(r["o2_pct"] for r in rows),
    }


def _line(label, rows):
    st = _stats(rows)
    top = sorted(rows, key=lambda r: -(r.get("burn_time_s") or 0))[:3]
    return (f"- **{label}** ({st['n']} tests, O2 {st['o2_min']:.1f}-{st['o2_max']:.1f}%): "
            f"{st['ignited']} ignited, {st['self_extinguished']} self-extinguished, "
            f"{st['still_burning_at_cutoff']} still burning when the fuel flow was cut; "
            f"median burn time {_fmt(st['median_burn_s'], 's')}, longest {_fmt(st['max_burn_s'], 's')}; "
            f"mean hazard index {st['mean_hazard']}/100. "
            f"Longest-burning tests: {', '.join(r['experiment_id'] for r in top)}.")


def _summary_doc(doc_id, title, body):
    return f"""---
experiment_id: {doc_id}
doc_type: summary
source_investigation: PSI-159 (CFI-G)
---

# {title}

## Summary

{body.strip()}
"""


def summary_documents(experiments):
    E = experiments
    air = [r for r in E if r["o2_pct"] < 25]
    enr = [r for r in E if r["o2_pct"] >= 25]
    docs = {}

    allst = _stats(E)
    days = sorted({r["experiment_id"].split("-")[1] for r in E})
    docs["SUMMARY-DATASET"] = _summary_doc("SUMMARY-DATASET", "Dataset overview — NASA PSI-159 CFI-G Normal Flames", f"""
The knowledge base contains {len(E)} real tests from the NASA PSI-159 CFI-G (Cool Flames
Investigation with Gases) experiment on the International Space Station (microgravity), run on
{len(days)} test days from {min(r['test_date'] for r in E)} to {max(r['test_date'] for r in E)}.
Each test burns a small gaseous fuel jet (ethane, propane or butane, pure or diluted with
nitrogen) in an oxygen/nitrogen atmosphere.

- Oxygen: {allst['o2_min']:.1f}-{allst['o2_max']:.1f}%. There are NO tests between 23.5% and 33% oxygen.
- Pressure: {min(r['pressure_kpa'] for r in E):.0f}-{max(r['pressure_kpa'] for r in E):.0f} kPa.
- Fuel flow: {min(r['fuel_flow_mg_s'] for r in E)}-{max(r['fuel_flow_mg_s'] for r in E)} mg/s.
- Outcomes: {allst['ignited']} of {len(E)} tests ignited a hot flame; {allst['self_extinguished']} self-extinguished
  and {allst['still_burning_at_cutoff']} were still burning when the crew cut the fuel flow
  ({len(E) - allst['ignited']} did not ignite).
- Median burn time of lit flames {_fmt(allst['median_burn_s'], 's')}, longest {_fmt(allst['max_burn_s'], 's')}.
- Recorded fields: hot_ignition, burn_time_s, end_mode (self_extinguished / flow_terminated / none),
  peak_burner_temp_c, cool_flame, plus a hazard index (0-100) computed from the outcome.

Note: {O2_CAVEAT}
""")

    docs["SUMMARY-OXYGEN"] = _summary_doc("SUMMARY-OXYGEN", "Effect of oxygen concentration — all tests", f"""
{_line("Air-like oxygen (18-23%)", air)}
{_line("Enriched oxygen (33-40%)", enr)}

At air-like oxygen every lit flame went out by itself; flames that kept burning until the fuel
was cut only occurred in enriched oxygen. No tests exist between 23.5% and 33% oxygen, so the
data cannot say exactly where this change happens.

Note: {O2_CAVEAT}
""")

    by_fuel = "\n".join(_line(f, [r for r in E if r["fuel"] == f]) for f in ("ethane", "propane", "butane"))
    mixes = sorted({r["fuel_mix"] for r in E})
    by_mix = "\n".join(_line(m, [r for r in E if r["fuel_mix"] == m]) for m in mixes)
    docs["SUMMARY-FUELS"] = _summary_doc("SUMMARY-FUELS", "Comparison of fuels — ethane, propane, butane", f"""
By fuel:
{by_fuel}

By fuel mix (diluted mixes contain nitrogen):
{by_mix}

Caution: each fuel was run on different days and at different oxygen/pressure conditions, so
this is not a controlled fuel-versus-fuel comparison.
""")

    plevels = sorted({round(r["pressure_kpa"]) for r in E})
    by_p = "\n".join(_line(f"{p} kPa", [r for r in E if round(r["pressure_kpa"]) == p]) for p in plevels)
    docs["SUMMARY-PRESSURE"] = _summary_doc("SUMMARY-PRESSURE", "Effect of pressure — all tests", f"""
{by_p}

Pressure and oxygen were changed together across test days, so pressure effects are mixed with
oxygen and fuel effects.
""")

    cool = [r for r in E if r.get("cool_flame")]
    docs["SUMMARY-COOL-FLAMES"] = _summary_doc("SUMMARY-COOL-FLAMES", "Cool flames", f"""
Cool flames were only examined on test days 21174 and 21291 (30% butane, about 2 bar, 39-40%
oxygen). {len(cool)} tests recorded a cool flame: {', '.join(r['experiment_id'] for r in cool)}.
On all other days the cool-flame field means "not examined", not "no cool flame", so the data
cannot say whether ethane or propane produce cool flames.
""")

    top_burn = sorted([r for r in E if r.get("burn_time_s")], key=lambda r: -r["burn_time_s"])[:10]
    top_hz = sorted(E, key=lambda r: -r["risk_score"])[:10]
    docs["SUMMARY-EXTREMES"] = _summary_doc("SUMMARY-EXTREMES", "Longest burns and highest hazard tests", f"""
Longest burn times:
""" + "\n".join(f"- {r['experiment_id']}: {r['fuel_mix']}, O2 {r['o2_pct']}%, {r['pressure_kpa']} kPa, "
                f"burn {r['burn_time_s']} s, end mode {r['end_mode']}" for r in top_burn) + """

Highest hazard index:
""" + "\n".join(f"- {r['experiment_id']}: hazard {r['risk_score']}/100 ({r['risk_level']}), {r['fuel_mix']}, "
                f"O2 {r['o2_pct']}%, end mode {r['end_mode']}" for r in top_hz))
    return docs


# ============================================================
# NASA PSI investigation documents (data-engineering rag_documents.jsonl)
# ============================================================
# 24 investigation summaries (objective, platform, hardware, DOI), 23 publication
# lists and 2 dataset summaries (BASS, SAFFIRE-III), all from psi.nasa.gov (CC0).

PSI_DOCS_FILE = PROJECT_ROOT / "Data" / "raw" / "pyronova" / "rag_documents.jsonl"


def psi_documents():
    docs = {}
    if not PSI_DOCS_FILE.exists():
        return docs
    for line in PSI_DOCS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        doc_id = "PSI-DOC-" + d["id"].replace("/", "-")
        title = str(d.get("title") or "").replace("\n", " ")
        docs[doc_id] = f"""---
experiment_id: {doc_id}
doc_type: nasa_psi
psi_source_type: {d.get("source_type") or ""}
investigation_id: {d.get("investigation_id") or ""}
source_url: {d.get("source_url") or ""}
doi: {d.get("doi") or ""}
---

# {title}

## Summary

{d["text"].strip()}

Source: {d.get("source_url") or "NASA PSI"}{(" · doi:" + d["doi"]) if d.get("doi") else ""}
"""
    return docs


# ============================================================
# Main
# ============================================================

def main():
    experiments = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    removed = 0
    for old in OUTPUT_DIR.glob("*.md"):
        old.unlink()
        removed += 1

    for exp in experiments:
        (OUTPUT_DIR / f"{exp['experiment_id']}.md").write_text(
            experiment_to_markdown(exp), encoding="utf-8")

    summaries = summary_documents(experiments)
    for doc_id, text in summaries.items():
        (OUTPUT_DIR / f"{doc_id}.md").write_text(text, encoding="utf-8")

    psi_docs = psi_documents()
    for doc_id, text in psi_docs.items():
        (OUTPUT_DIR / f"{doc_id}.md").write_text(text, encoding="utf-8")

    print(f"Input file      : {INPUT_FILE}")
    print(f"Removed old docs: {removed}")
    print(f"Created         : {len(experiments)} experiment documents (real PSI-159 data)")
    print(f"                + {len(summaries)} dataset summary documents")
    print(f"                + {len(psi_docs)} NASA PSI investigation documents")
    print(f"Output directory: {OUTPUT_DIR}")
    return len(experiments)


if __name__ == "__main__":
    main()

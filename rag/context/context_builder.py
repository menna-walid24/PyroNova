# ============================================================
# CONTEXT BUILDER
# ============================================================

class ContextBuilder:

    def __init__(self):
        pass

    def build(self, selected_results):
        """
        Build an LLM-ready context from selected
        retrieval results.

        Parameters
        ----------
        selected_results : list
            Results selected by ContextSelector.

        Returns
        -------
        str
            Formatted context.
        """

        if not selected_results:
            return (
                "No relevant experimental evidence "
                "was retrieved."
            )

        context_parts = []

        # ----------------------------------------------------
        # Build each evidence block
        # ----------------------------------------------------

        for result in selected_results:

            metadata = result.get(
                "metadata",
                {}
            )

            if metadata.get("doc_type") in ("summary", "nasa_psi"):
                label = (
                    "Dataset summary (statistics over all 179 real PSI-159 ISS tests)"
                    if metadata.get("doc_type") == "summary"
                    else f"NASA PSI document — {metadata.get('source_url') or 'psi.nasa.gov'}"
                         + (f" (doi:{metadata.get('doi')})" if metadata.get("doi") else "")
                )
                context_parts.append(f"""
[EVIDENCE {result.get("context_rank", "Unknown")}]

Source ID: {metadata.get("experiment_id")} · {label}

{result.get("text", "").strip()}
""".strip())
                continue

            def g(key):
                value = metadata.get(key)
                return "not recorded" if value in (None, "") else value

            context_rank = result.get("context_rank", "Unknown")
            text = result.get("text", "").strip()

            o2 = metadata.get("oxygen_before_pct")
            o2_note = (
                " (ISS test-chamber condition, not cabin air)"
                if isinstance(o2, (int, float)) and o2 >= 25 else ""
            )

            # ------------------------------------------------
            # Evidence block (real NASA PSI-159 CFI-G test)
            # ------------------------------------------------

            evidence = f"""
[EVIDENCE {context_rank}]

Experiment ID: {g("experiment_id")}
Investigation: {g("source_investigation")} (ISS microgravity)
Test date: {g("test_date")}
Fuel: {g("fuel_mix")}

Experimental Conditions:
- Oxygen concentration: {g("oxygen_before_pct")}%{o2_note}
- Pressure: {g("pressure_kpa")} kPa
- Fuel flow: {g("fuel_flow_mg_s")} mg/s
- Fuel mole fraction (Xf): {g("xf")}

Observed Results:
- Hot ignition: {g("hot_ignition")}
- Burn time: {g("burn_time_s")} s
- End mode: {g("end_mode")}
- Peak burner temperature: {g("peak_burner_temp_c")} °C
- Cool flame: {g("cool_flame")}

Outcome: {g("outcome")}
Hazard index: {g("risk_score")} / 100 ({g("risk_level")})

Experimental Evidence:
{text}
""".strip()

            context_parts.append(
                evidence
            )

        # ----------------------------------------------------
        # Combine evidence
        # ----------------------------------------------------

        context = "\n\n".join(
            context_parts
        )

        return context
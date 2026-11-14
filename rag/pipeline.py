# ============================================================
# RAG PIPELINE ORCHESTRATOR
# ============================================================


import re

try:                                   # imported as the package module rag.pipeline
    from .ml.predict import predict as predict_fire
except ImportError:                    # imported as a top-level module (notebooks)
    from ml.predict import predict as predict_fire


# Real test IDs look like CFI-21088-A1. Also accept PSI159-21088A1 and "CFI 21088 A1".
_ID_RE = re.compile(r"\bCFI[-_ ]?(\d{5})[-_ ]?([A-Z]\d{1,2})\b", re.IGNORECASE)
_PSI_RE = re.compile(r"\bPSI[-_ ]?159[-_ ]?(\d{5})[-_ ]?([A-Z]\d{1,2})\b", re.IGNORECASE)


def find_experiment_id(text):
    text = str(text)
    m = _ID_RE.search(text) or _PSI_RE.search(text)
    return f"CFI-{m.group(1)}-{m.group(2).upper()}" if m else None


O2_CAVEAT = (
    "The 33-40% oxygen levels in these tests are ISS test-chamber conditions, "
    "not a spacecraft cabin atmosphere (ISS cabin air is about 21%)."
)
COOL_FLAME_SCOPE = (
    "Cool flames were only examined on test days 21174 and 21291 (30% butane, ~2 bar, "
    "~39% O2); for other tests 'no cool flame' means 'not examined'."
)


# Broad questions are answered better from dataset-level summaries than from
# a few individual tests, so these summaries are always added to the context.
_SUMMARY_RULES = [
    ("SUMMARY-FUELS", r"\bfuel|ethane|propane|butane|compare|dangerous|riskiest|safest|وقود|غاز|بيوتان|بروبان|ايثان|إيثان|قارن"),
    ("SUMMARY-OXYGEN", r"oxygen|\bo2\b|%|أكسجين|اكسجين|اوكسجين"),
    ("SUMMARY-PRESSURE", r"pressure|\bkpa\b|\bbar\b|ضغط"),
    ("SUMMARY-COOL-FLAMES", r"cool flame|لهب بارد"),
    ("SUMMARY-EXTREMES", r"longest|highest|most|top|maximum|max\b|worst|اطول|أطول|اعلى|أعلى|اكتر|أكثر"),
    ("SUMMARY-DATASET", r"how many|dataset|overview|what (data|tests)|which tests|summary|كام|عدد|الداتا|البيانات"),
]


def summary_ids_for(question):
    q = str(question).lower()
    return [doc_id for doc_id, pattern in _SUMMARY_RULES if re.search(pattern, q)][:3]


def _mentions_enriched_oxygen(question, results):
    for r in results or []:
        o2 = (r.get("metadata") or {}).get("oxygen_before_pct")
        if isinstance(o2, (int, float)) and o2 >= 25:
            return True
    return bool(re.search(r"\b(2[5-9]|3\d|4\d)(\.\d+)?\s*%|enriched|oxygen-rich|high oxygen",
                          str(question).lower()))


def add_data_caveats(response, question, results):
    """Deterministic caveats, so they never depend on the LLM remembering them."""
    if not isinstance(response, dict):
        return response
    limitations = response.get("limitations")
    if not isinstance(limitations, list):
        limitations = [] if limitations in (None, "") else [str(limitations)]
    if _mentions_enriched_oxygen(question, results) and not any(
            "test-chamber" in str(x) for x in limitations):
        limitations.append(O2_CAVEAT)
    if "cool flame" in str(question).lower() and not any(
            "examined" in str(x) for x in limitations):
        limitations.append(COOL_FLAME_SCOPE)
    response["limitations"] = limitations

    # Never return an evidence-based answer without its sources.
    if not response.get("sources"):
        response["sources"] = list(dict.fromkeys(
            (r.get("metadata") or {}).get("experiment_id")
            for r in results or []
            if (r.get("metadata") or {}).get("experiment_id")
        ))
    return response



def enforce_prediction_schema(response, ml_prediction, question):
    """Keep real ML outputs separate from experimental measurements."""

    if not isinstance(response, dict):
        return response

    # Never trust a prediction invented by the LLM.
    response["prediction"] = None

    if ml_prediction is None:
        return response

    records = (
        [ml_prediction]
        if isinstance(ml_prediction, dict)
        else ml_prediction
    )

    if not isinstance(records, list):
        return response

    requested_id = find_experiment_id(question)

    valid_predictions = []

    for record in records:
        if not isinstance(record, dict):
            continue

        experiment_id = record.get("experiment_id")

        # Exact experiment questions must match that experiment.
        if requested_id and str(experiment_id).upper() != requested_id:
            continue

        actual = record.get("prediction")

        # Support a direct model output as well.
        if not isinstance(actual, dict):
            if "label" in record and "probability" in record:
                actual = record
            else:
                continue

        if "label" not in actual or "probability" not in actual:
            continue

        valid_predictions.append({
            "experiment_id": experiment_id,
            **actual,
        })

    if not valid_predictions:
        response["prediction"] = None

    elif requested_id:
        # One matching prediction for an exact experiment question.
        response["prediction"] = valid_predictions[0]

    else:
        # Preserve predictions for broad questions, without
        # arbitrarily selecting one experiment.
        response["prediction"] = valid_predictions

    return response


class RAGPipeline:

    def __init__(
        self,
        retriever,
        reranker,
        metadata_filter,
        context_selector,
        context_builder,
        answerability_gate,
        response_generator,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.metadata_filter = metadata_filter
        self.context_selector = context_selector
        self.context_builder = context_builder
        self.answerability_gate = answerability_gate
        self.response_generator = response_generator

    def _experiment_id(self, candidate):
        metadata = candidate.get("metadata", {})

        value = metadata.get("experiment_id")
        if value:
            return str(value).upper()

        return find_experiment_id(candidate.get("text", ""))


    def _field_value(self, text, aliases):
        clean_text = str(text).lower()

        # Normalize Markdown formatting and line breaks.
        clean_text = re.sub(r"[*_`]", "", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text)

        for alias in aliases:
            pattern = (
                r"\b"
                + re.escape(alias)
                + r"\b\s*"
                + r"(?:(?:was|is|of|:)\s*)*"
                + r"([-+]?\d+(?:\.\d+)?)"
                + r"\s*"
                + r"([a-zA-Z%/³^.-]*)"
              )

            match = re.search(pattern, clean_text)

            if match:
                value = float(match.group(1))
                unit = match.group(2).strip()
                return value, unit

        return None, None


    def _knowledge_base_response(self, question, candidates):
        q = question.lower()
        requested_id = self._experiment_id({
             "text": question,
             "metadata": {}
                    })
        requested_fields = {
            "burn time": ["burn time"],
            "peak burner temperature": ["peak burner temperature"],
            "adiabatic flame temperature": ["adiabatic flame temperature"],
            "fuel flow": ["fuel flow"],
            "oxygen concentration": ["oxygen concentration (before test)"],
            "pressure": ["pressure"],
            "hazard index": ["hazard index (risk score)"],
            "risk score": ["hazard index (risk score)"],
        }

        requested_field = next(
            (
                field for field in sorted(
                    requested_fields, key=len, reverse=True
                )
                if field in q
            ),
            None,
        )

        evidence = []
        sources = []
        answer_parts = []

        for candidate in candidates:
            text = str(candidate.get("text", ""))
            metadata = candidate.get("metadata", {})
            experiment_id = self._experiment_id(candidate)

            if requested_id and experiment_id != requested_id:
                    continue

            # Preserve evidence only when the requested field
            # and its numeric value are actually present.
            if requested_field:
                value, unit = self._field_value(
                    text, requested_fields[requested_field]
                )

                if value is None:
                    continue

                finding = (
                    f"Observed {requested_field}: {value:g}"
                    f"{(' ' + unit) if unit else ''}."
                )

                item = {
                    "experiment_id": experiment_id,
                    "measurement": requested_field,
                    "value": value,
                    "unit": unit,
                    "finding": finding,
                }

                evidence.append(item)
                answer_parts.append(
                    f"{experiment_id}: {finding}"
                )

                if experiment_id not in sources:
                    sources.append(experiment_id)

            else:
                # Don't invent an answer to a broad question
                # using a generic chunk summary.
                continue

        if not evidence:
            return None

        return {
            "answer": " ".join(answer_parts),
            "experimental_evidence": evidence,
            "prediction": None,
            "observed_trends": [],
            "interpretation": (
                "Values are reported experimental observations, "
                "not machine-learning predictions."
            ),
            "safety_insight": (
                "Interpret the measurement within the specific "
                "experimental conditions; do not generalize it "
                "to other conditions without supporting evidence."
            ),
            "limitations": [
                "This response reports retrieved measurements only.",
                "No prediction was generated."
            ],
            "sources": sources,
            "source": "knowledge_base",
        }


    def _predict_from_retrieved_experiments(self, results):
        """Predict fire risk using conditions from retrieved experiments."""
        predictions = []
        seen_experiments = set()

        # Pre-test conditions of a real CFI-G test (the model's inputs).
        required_fields = [
            "fuel_mix",
            "oxygen_before_pct",
            "pressure_kpa",
            "fuel_flow_mg_s",
            "xf",
            "fuel_bottle_pct",
        ]

        for item in results:
            # Support metadata stored in different result structures
            metadata = item.get("metadata", {})

            if not metadata and isinstance(item.get("document"), dict):
                metadata = item["document"].get("metadata", {})

            if not metadata:
                metadata = item

            experiment_id = metadata.get("experiment_id")

            # Dataset summaries describe many tests: no single set of conditions.
            if metadata.get("doc_type") in ("summary", "nasa_psi"):
                continue

            # Avoid predicting repeatedly for multiple chunks of one experiment
            if not experiment_id or experiment_id in seen_experiments:
                continue

            seen_experiments.add(experiment_id)

            # Never silently use model defaults for missing experiment data
            missing = [
                field for field in required_fields
                if metadata.get(field) is None
                or metadata.get(field) == ""
            ]

            if missing:
                predictions.append({
                    "experiment_id": experiment_id,
                    "status": "insufficient_input",
                    "missing_fields": missing,
                })
                continue

            conditions = {
                field: metadata[field]
                for field in required_fields
            }

            try:
                full = predict_fire({
                    "fuel": conditions["fuel_mix"],
                    "oxygen_before_pct": conditions["oxygen_before_pct"],
                    "pressure_kpa": conditions["pressure_kpa"],
                    "fuel_flow_mg_s": conditions["fuel_flow_mg_s"],
                    "xf": conditions["xf"],
                    "fuel_bottle_pct": conditions["fuel_bottle_pct"],
                })
                # Compact view for the LLM prompt and the API response.
                prediction = {
                    "label": full.get("label"),
                    "probability": full.get("probability"),
                    "riskBand": full.get("riskBand"),
                    "predicted_end_mode": full.get("predicted_end_mode"),
                    "end_mode_probabilities": full.get("end_mode_probabilities"),
                    "p_hot_ignition": (full.get("hot_ignition") or {}).get("probability"),
                    "predicted_burn_time_s": full.get("burn_time_s"),
                    "hazard_index": full.get("hazard_index"),
                    "level": full.get("level"),
                    "model": full.get("model"),
                    "caveats": full.get("caveats", []),
                    "observed_end_mode": metadata.get("end_mode"),
                    "observed_burn_time_s": metadata.get("burn_time_s"),
                }

                predictions.append({
                    "experiment_id": experiment_id,
                    "status": "predicted",
                    "conditions": conditions,
                    "prediction": prediction,
                })

            except (ValueError, TypeError) as exc:
                predictions.append({
                    "experiment_id": experiment_id,
                    "status": "prediction_error",
                    "reason": str(exc),
                })

        return predictions or None

    
    def answer(self, question, conditions=None, ml_prediction=None):
        # 1. Hybrid retrieval
        hybrid_results = self.retriever.search(
            query=question,
            dense_top_k=20,
            bm25_top_k=20,
            top_k=20,
        )

        # 2. Enrich retrieved chunks with metadata
        enriched_results = self.metadata_filter.enrich_results(
            hybrid_results
        )

        # 2b. A question naming a test ID gets that test's own chunks
        #     (exact lookup — semantic search is unreliable for IDs).
        requested_id = find_experiment_id(question)
        if requested_id:
            exact = [
                {"chunk_id": c.get("chunk_id", i), "score": 1.0,
                 "text": c.get("text", ""), "metadata": c.get("metadata", {})}
                for i, c in enumerate(self.metadata_filter.chunks)
                if str(c.get("metadata", {}).get("experiment_id", "")).upper() == requested_id
            ]
            if exact:
                enriched_results = exact

        # 3. Cross-encoder reranking
        reranked_results = self.reranker.rerank(
            query=question,
            candidates=enriched_results,
            top_k=6,
        )

        # 3b. Broad questions: put the matching dataset summaries first.
        if not requested_id:
            wanted = summary_ids_for(question)
            if wanted:
                summaries = [
                    {"chunk_id": c.get("chunk_id", i), "score": 1.0,
                     "reranker_score": 10.0,
                     "text": c.get("text", ""), "metadata": c.get("metadata", {})}
                    for i, c in enumerate(self.metadata_filter.chunks)
                    if c.get("metadata", {}).get("experiment_id") in wanted
                ]
                summaries.sort(key=lambda c: wanted.index(c["metadata"]["experiment_id"]))
                have = {c["chunk_id"] for c in summaries}
                reranked_results = summaries + [
                    r for r in reranked_results if r.get("chunk_id") not in have
                ]

        # 4. Check whether the evidence can answer the question
        gate_result = self.answerability_gate.check(
            question,
            reranked_results,
        )
        
        if ml_prediction is None:
            ml_prediction = self._predict_from_retrieved_experiments(
            reranked_results
        )
        
        # 5. Select evidence and build the LLM context
        selected_results = self.context_selector.select(
            reranked_results
        )

            
        context = self.context_builder.build(
            selected_results
        )

        
        
        # 6. Generate the LLM response (fall back to the knowledge base
        #    if the LLM provider is unreachable, so the API never 500s).
        try:
            response = self.response_generator.generate(
                question=question,
                context=context,
                ml_prediction=ml_prediction,
            )
        except Exception as exc:
            print("LLM unavailable, using knowledge-base fallback:", exc)
            response = self._knowledge_base_response(question, reranked_results) or {
                "answer": (
                    "The language model is unavailable right now. The most relevant "
                    "real ISS tests are listed in the sources."
                ),
                "experimental_evidence": [],
                "prediction": None,
                "observed_trends": [],
                "interpretation": "",
                "safety_insight": "",
                "limitations": ["Answer generated without the language model."],
                "sources": [
                    r.get("metadata", {}).get("experiment_id")
                    for r in selected_results
                    if r.get("metadata", {}).get("experiment_id")
                ],
                "source": "knowledge_base",
            }
            response["model_used"] = "knowledge_base_fallback"

        # 7. Enforce the prediction schema
        response = enforce_prediction_schema(
            response=response,
            ml_prediction=ml_prediction,
            question=question,
        )
        response = add_data_caveats(response, question, selected_results)
        # 7. Return the complete result
        return {
            "response": response,
            "answerability": gate_result,
            "retrieved_results": reranked_results,
        }

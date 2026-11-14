import sys

# Windows consoles default to cp1252: printing Arabic answers would crash a request.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rag.agent import ResearchAgent
from typing import Any
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Replace this import with the module where you move
# your existing, working pipeline initialization.
from rag.service import get_pipeline
from backend import telemetry_replay, experiments

app = FastAPI(
    title="Flame in Freefall API",
    description="Evidence-backed microgravity combustion research",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://192.168.1.18:3000",
    ],
    # Any local port + public tunnels used for team testing
    # (Cloudflare quick tunnels, tunnl.gg).
    allow_origin_regex=(
        r"^https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+)(:\d+)?$"
        r"|^https://[a-z0-9-]+\.(trycloudflare\.com|tunnl\.gg)$"
    ),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
pipeline = None
agent = None

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.on_event("startup")
def startup():
    global pipeline, agent

    pipeline = get_pipeline()
    agent = ResearchAgent(pipeline)

    print("Research agent initialized successfully.")


@app.get("/health")
def health():
    chunks = None
    try:
        chunks = len(pipeline.metadata_filter.chunks) if pipeline else None
    except Exception:
        pass
    return {
        "status": "ok",
        "pipeline_loaded": pipeline is not None,
        "dataset": "NASA PSI-159 CFI-G (179 real ISS tests)",
        "chunks": chunks,
    }


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not initialized.",
        )

    try:
        if agent is None:
            raise HTTPException(
                status_code=503,
                detail="Research agent is not initialized."
            )
        result = agent.answer(request.question)
        # Your pipeline returns a wrapper containing "response".
        response = result.get("response", result)

        if not isinstance(response, dict):
            raise ValueError("Pipeline response must be a dictionary.")

        return {
            "question": request.question,
            "answer": response.get("answer", ""),
            "experimental_evidence": response.get(
                "experimental_evidence", []
            ),
            "prediction": response.get("prediction"),
            "observed_trends": response.get("observed_trends", []),
            "interpretation": response.get("interpretation", ""),
            "safety_insight": response.get("safety_insight", ""),
            "limitations": response.get("limitations", []),
            "sources": response.get("sources", []),
            "model_used": response.get("model_used"),
            "answerability": result.get("answerability", {}),
            "prediction_notice": (
                "ML predictions are trained on 175 real ISS tests "
                "(NASA PSI-159 CFI-G) and are not certified "
                "fire-safety guidance."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(
              status_code=500,
              detail=f"{type(exc).__name__}: {exc}",
        ) from exc


# ------------------------------------------------------------------
# Fire-risk status for the mobile Home card: replay of REAL archived
# SAFFIRE-III flight telemetry (not a live feed — see telemetry_replay.py)
# ------------------------------------------------------------------
@app.get("/risk/current")
def risk_current() -> dict[str, Any]:
    try:
        return telemetry_replay.current()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Telemetry replay unavailable: {exc}") from exc


@app.get("/telemetry/saffire3")
def telemetry_saffire3(step: int = 10) -> dict[str, Any]:
    return {
        "source": telemetry_replay.SOURCE,
        "doi": telemetry_replay.DOI,
        "is_live": False,
        "points": telemetry_replay.timeline(step),
    }


# ------------------------------------------------------------------
# Experiment explorer (web page /experiments) — real NASA PSI investigations
# ------------------------------------------------------------------
@app.get("/experiments")
def experiments_list(
    environment: str = Query("cabin", description="cabin | lunar | mars"),
    goal: str = Query("all", description="all | flammability | suppression | detection | fundamentals"),
    q: str = Query("", description="free-text search"),
) -> dict[str, Any]:
    return experiments.list_experiments(environment, goal, q)


@app.get("/experiments/compare")
def experiments_compare(
    ids: str = Query(..., description="comma-separated PSI ids, max 3, e.g. PSI-26,PSI-100"),
    environment: str = "cabin",
) -> dict[str, Any]:
    return experiments.compare([i.strip() for i in ids.split(",") if i.strip()], environment)


@app.get("/experiments/earth-impact")
def experiments_earth_impact() -> dict[str, Any]:
    return experiments.earth_impact()


@app.get("/experiments/{inv_id}")
def experiments_detail(inv_id: str, environment: str = "cabin") -> dict[str, Any]:
    item = experiments.get_experiment(inv_id, environment)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Unknown investigation {inv_id}")
    return item


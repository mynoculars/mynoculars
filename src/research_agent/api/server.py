"""
api/server.py — Minimal HTTP interface.

Purpose:
    One POST endpoint running the same graph as the CLI — demonstrates that
    the assembly function is interface-agnostic.

Responsibilities:
    - POST /research {"query": "..."} -> {"report": ..., "telemetry": ...}
    - GET /health for liveness.

Run:
    uvicorn research_agent.api.server:app --reload
"""

import uuid

from fastapi import FastAPI
from pydantic import BaseModel

from research_agent.cli import build_app_and_settings
from research_agent.logging_setup import run_id_var
from research_agent.state import ResearchState

app = FastAPI(title="Agentic Research Agent (core build)")
_graph, _settings = build_app_and_settings()


class ResearchRequest(BaseModel):
    """Request body for /research."""

    query: str
    thread_id: str | None = None


@app.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok", "llm_mode": _settings.llm_mode}


@app.post("/research")
def research(req: ResearchRequest) -> dict:
    """Run one research query end to end and return report + telemetry."""
    thread_id = req.thread_id or f"api-{uuid.uuid4().hex[:12]}"
    run_id_var.set(thread_id)
    result = _graph.invoke(
        ResearchState(raw_query=req.query),
        config={"configurable": {"thread_id": thread_id},
                "recursion_limit": _settings.recursion_limit},
    )
    return {"thread_id": thread_id,
            "report": result["final_report"],
            "telemetry": result["telemetry"]}

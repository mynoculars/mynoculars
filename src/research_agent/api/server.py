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
from langgraph.types import Command
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


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id},
            "recursion_limit": _settings.recursion_limit}


def _respond(thread_id: str, result: dict) -> dict:
    """Shared shape for both endpoints: finished vs interrupted (D-23)."""
    if "__interrupt__" in result:
        return {"thread_id": thread_id, "status": "interrupted",
                "review": result["__interrupt__"][0].value}
    return {"thread_id": thread_id, "status": "done",
            "report": result["final_report"],
            "telemetry": result["telemetry"]}


class ResumeRequest(BaseModel):
    """Request body for /resume — the human's escalation decision."""

    thread_id: str
    action: str            # approve | redirect | abort
    guidance: str = ""


@app.post("/research")
def research(req: ResearchRequest) -> dict:
    """Run a query; returns done, or interrupted with a review payload."""
    thread_id = req.thread_id or f"api-{uuid.uuid4().hex[:12]}"
    run_id_var.set(thread_id)
    result = _graph.invoke(ResearchState(raw_query=req.query),
                           config=_config(thread_id))
    return _respond(thread_id, result)


@app.post("/resume")
def resume(req: ResumeRequest) -> dict:
    """Resume an interrupted run under its thread_id (D-20/D-23)."""
    run_id_var.set(req.thread_id)
    result = _graph.invoke(
        Command(resume={"action": req.action, "guidance": req.guidance}),
        config=_config(req.thread_id))
    return _respond(req.thread_id, result)

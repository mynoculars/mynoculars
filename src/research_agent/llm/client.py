"""
llm/client.py — Minimal chat clients: one real, one stub.

Purpose:
    Isolate ALL model I/O behind a two-method protocol so the rest of the
    codebase never touches HTTP, SDKs, or provider quirks.

Responsibilities:
    - ChatClient protocol: complete() and complete_json().
    - OpenAICompatibleClient: works for BOTH the local Qwen Cogito server
      (llama-server exposes /v1/chat/completions) and Gemini Flash (Google
      publishes an OpenAI-compatibility endpoint). One class, two configs.
    - StubClient: deterministic canned responses so the entire graph runs
      offline — used by tests and by LLM_MODE=stub.

Design decisions:
    - Why one client class instead of per-provider classes: both targets
      speak the same wire protocol; a second class would be duplicate code
      pretending to be abstraction. Alternative considered: LiteLLM (heavy
      dependency for a learning repo). Adding a genuinely different provider
      later = implement ChatClient, register in router — nothing else moves.
    - Why raw httpx instead of the openai SDK: ~40 lines of transparent HTTP
      teaches more than an SDK call, and drops a dependency.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Protocol

import httpx

from research_agent.logging_setup import log_event

logger = logging.getLogger(__name__)

Message = Dict[str, str]  # {"role": ..., "content": ...}


class ChatClient(Protocol):
    """The only surface the agent uses to talk to a model."""

    name: str

    def complete(self, messages: List[Message], temperature: float = 0.2) -> str:
        """Return the model's text response for a chat transcript."""
        ...

    def complete_json(self, messages: List[Message], temperature: float = 0.0) -> Dict[str, Any]:
        """Return the model's response parsed as a JSON object.

        Raises ValueError if the response is not parseable JSON — the router
        treats that as a quality failure and may fall back (see router.py).
        """
        ...


def _extract_json(text: str) -> Dict[str, Any]:
    """Parse JSON out of a model reply, tolerating ```json fences.

    Why: small local models frequently wrap JSON in markdown fences even
    when told not to. Stripping fences before parsing avoids a fallback
    round-trip for a purely cosmetic failure.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    obj = json.loads(cleaned)
    if not isinstance(obj, dict):
        raise ValueError("model returned JSON that is not an object")
    return obj


class OpenAICompatibleClient:
    """Chat client for any /v1/chat/completions endpoint."""

    def __init__(self, name: str, base_url: str, api_key: str, model: str,
                 timeout: float = 60.0, tracer: Any = None, display_label: str = ""):
        """Parameters map directly to Settings fields; see config.py.

        `tracer` (a Tracer, optional) receives the exact prompt/response/tokens/
        latency for every call when debug tracing is on. `display_label` is the
        human name shown in the trace banner (e.g. "GOOGLE GEMINI FLASH"); it
        falls back to the model id. `_trace_node` is set by the router before
        each call so the trace shows which graph node issued it."""
        self.name = name
        self._model = model
        self._tracer = tracer
        self._label = display_label or model
        self._trace_node: Optional[str] = None
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def set_trace_node(self, node: Optional[str]) -> None:
        """Router sets the current node name so traces attribute each call."""
        self._trace_node = node

    def complete(self, messages: List[Message], temperature: float = 0.2) -> str:
        """POST the transcript; return assistant text. Raises httpx errors
        upward — the router owns retry/fallback policy, not this class."""
        started = time.perf_counter()
        resp = self._http.post(
            "/chat/completions",
            json={"model": self._model, "messages": messages, "temperature": temperature},
        )
        resp.raise_for_status()
        data = resp.json()
        latency = time.perf_counter() - started
        text: str = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
        log_event(logger, "llm.call", provider=self.name, model=self._model,
                  label=self._label, node=self._trace_node,
                  latency_s=round(latency, 3),
                  prompt_tokens=pt, completion_tokens=ct)
        if self._tracer is not None:
            self._tracer.record_llm(self._label, self._trace_node, messages,
                                    text, pt, ct, latency)
        return text

    def complete_json(self, messages: List[Message], temperature: float = 0.0) -> Dict[str, Any]:
        """complete() then strict-parse. ValueError on non-JSON output."""
        return _extract_json(self.complete(messages, temperature))


class StubClient:
    """Offline deterministic client.

    Looks at the LAST user message for a routing hint the prompt templates
    embed (e.g. "TASK=classify") and returns a canned, schema-valid answer.
    This keeps stub behavior honest: the same prompts and the same JSON
    schemas as live mode, just fixed content.
    """

    name = "stub"

    def __init__(self, tracer: Any = None):
        self._tracer = tracer
        self._trace_node: Optional[str] = None

    def set_trace_node(self, node: Optional[str]) -> None:
        self._trace_node = node

    _CANNED: Dict[str, Dict[str, Any]] = {
        "classify": {"intent": "Comparison", "confidence": 0.9},
        "goals": {"goals": [
            {"goal_id": "g1", "description": "Identify the key differences"},
            {"goal_id": "g2", "description": "Summarize practical tradeoffs"},
        ]},
        "expand": {"tasks": [
            {"query": "key differences", "goal_id": "g1", "priority": 2},
            {"query": "practical tradeoffs", "goal_id": "g2", "priority": 1},
        ]},
        "gaps": {"tasks": []},
        "critique": {"passed": True, "score": 0.9, "notes": []},
        "quality": {"score": 0.9},
    }

    def complete(self, messages: List[Message], temperature: float = 0.2) -> str:
        """Return canned JSON (as text) for known tasks, else a fixed report."""
        last = messages[-1]["content"]
        answer = None
        for tag, payload in self._CANNED.items():
            if f"TASK={tag}" in last:
                answer = json.dumps(payload)
                break
        if answer is None:
            answer = ("# Research Report (stub mode)\n\n"
                      "This deterministic report proves the full graph executed "
                      "offline. Switch LLM_MODE=live for real model output.")
        if self._tracer is not None:
            self._tracer.record_llm("STUB (offline)", self._trace_node,
                                    messages, answer, None, None, 0.0)
        return answer

    def complete_json(self, messages: List[Message], temperature: float = 0.0) -> Dict[str, Any]:
        """Parse the canned reply."""
        return _extract_json(self.complete(messages, temperature))

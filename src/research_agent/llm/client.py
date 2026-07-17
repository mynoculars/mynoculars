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
from typing import Any, Dict, List, Protocol

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

    def __init__(self, name: str, base_url: str, api_key: str, model: str, timeout: float = 60.0):
        """Parameters map directly to Settings fields; see config.py."""
        self.name = name
        self._model = model
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def complete(self, messages: List[Message], temperature: float = 0.2) -> str:
        """POST the transcript; return assistant text. Raises httpx errors
        upward — the router owns retry/fallback policy, not this class."""
        resp = self._http.post(
            "/chat/completions",
            json={"model": self._model, "messages": messages, "temperature": temperature},
        )
        resp.raise_for_status()
        data = resp.json()
        text: str = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        log_event(logger, "llm.call", provider=self.name, model=self._model,
                  prompt_tokens=usage.get("prompt_tokens"),
                  completion_tokens=usage.get("completion_tokens"))
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
        for tag, payload in self._CANNED.items():
            if f"TASK={tag}" in last:
                return json.dumps(payload)
        return ("# Research Report (stub mode)\n\n"
                "This deterministic report proves the full graph executed "
                "offline. Switch LLM_MODE=live for real model output.")

    def complete_json(self, messages: List[Message], temperature: float = 0.0) -> Dict[str, Any]:
        """Parse the canned reply."""
        return _extract_json(self.complete(messages, temperature))

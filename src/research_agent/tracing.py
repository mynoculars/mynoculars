"""
tracing.py — Optional per-run debug tracer.

Purpose:
    When enabled (--debug on the CLI, or DEBUG_TRACE=true in .env), capture the
    exact inputs/outputs at every external boundary of a run — each LLM call
    (which provider served it, the prompt sent, the raw response, tokens,
    latency) and each retrieval call (which engine, what it returned) — and
    write them to a single human-readable file per run:  logs/trace-<run_id>.txt

Responsibilities:
    - Tracer: a small sink that accumulates banner-delimited entries and flushes
      them to a file. Threaded explicitly into the LLM clients and the storage
      wrappers (the two boundaries worth tracing) so there is no hidden global
      state — a disabled tracer is a cheap no-op passed by reference.

Design decision (explicit object over a global singleton):
    The tracer is injected the same way every other dependency is (see cli.py),
    so it is testable, has no import-time side effects, and a run with tracing
    off carries a NullTracer whose methods return immediately. This keeps the
    default (non-debug) path free of file I/O and formatting cost.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Dict, List, Optional

_BANNER = "=" * 78


class Tracer:
    """Accumulates debug entries for one run and flushes to a file.

    A single Tracer instance is created per invoke (keyed by run_id) and passed
    to every component that touches an external boundary. Call record_llm() /
    record_retrieval() as those boundaries are crossed, then flush() once at the
    end of the run.
    """

    def __init__(self, run_id: str, log_dir: str = "logs"):
        self.run_id = run_id
        self._dir = pathlib.Path(log_dir)
        self._entries: List[str] = []

    @property
    def enabled(self) -> bool:
        return True

    def record_llm(self, source_label: str, node: Optional[str],
                   prompt_messages: List[Dict[str, str]], response: str,
                   prompt_tokens: Optional[int], completion_tokens: Optional[int],
                   latency_s: float) -> None:
        """Record one LLM call: which provider, the exact prompt, raw response."""
        header = (f"RETRIEVED FROM {source_label.upper()}"
                  f"{f'  |  node={node}' if node else ''}"
                  f"  |  {latency_s:.2f}s"
                  f"  |  prompt_tok={prompt_tokens}  completion_tok={completion_tokens}")
        prompt_text = "\n".join(
            f"[{m.get('role','?')}]\n{m.get('content','')}" for m in prompt_messages)
        self._entries.append(
            f"{_BANNER}\n{header}\n{_BANNER}\n"
            f"PROMPT:\n{prompt_text}\n"
            f"{'-'*78}\n"
            f"RESPONSE:\n{response}\n")

    def record_retrieval(self, source_label: str, query: str,
                         hits: List[Dict[str, Any]]) -> None:
        """Record one retrieval call: which engine, the query, the raw hits."""
        header = (f"RETRIEVED FROM {source_label.upper()}"
                  f"  |  query={query!r}  |  {len(hits)} hits")
        body_lines: List[str] = []
        for i, h in enumerate(hits, 1):
            slim = {k: v for k, v in h.items() if k not in ("vector", "embedding")}
            body_lines.append(f"[hit {i}] " + json.dumps(slim, default=str, ensure_ascii=False))
        body = "\n".join(body_lines) or "(no hits)"
        self._entries.append(f"{_BANNER}\n{header}\n{_BANNER}\n{body}\n")

    def note(self, text: str) -> None:
        """Free-form marker (e.g. run header/footer)."""
        self._entries.append(text)

    def flush(self) -> Optional[str]:
        """Write all accumulated entries to logs/trace-<run_id>.txt. Returns
        the path written, or None if there was nothing to write."""
        if not self._entries:
            return None
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"trace-{self.run_id}.txt"
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"DEBUG TRACE  |  run_id={self.run_id}  |  {stamp}\n\n")
            f.write("\n".join(self._entries))
        return str(path)


class NullTracer(Tracer):
    """No-op tracer used when tracing is disabled. Every recorder returns
    immediately; nothing is stored or written, so the non-debug path pays no
    formatting or I/O cost."""

    def __init__(self) -> None:  # noqa: D107 — intentionally skips base init
        pass

    @property
    def enabled(self) -> bool:
        return False

    def record_llm(self, *a: Any, **k: Any) -> None:
        return None

    def record_retrieval(self, *a: Any, **k: Any) -> None:
        return None

    def note(self, text: str) -> None:
        return None

    def flush(self) -> Optional[str]:
        return None

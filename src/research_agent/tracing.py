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

Python mechanics used in this file, if any of this is new to you:
    from __future__ import annotations
        Makes every type hint in this file (e.g. "-> Optional[str]") get
        treated as plain text rather than being evaluated immediately when
        the module loads. This has no effect on how the code below actually
        runs — it only matters for forward references in type hints and for
        slightly faster module import — and is unrelated to any behaviour
        described elsewhere in this file.
    class NullTracer(Tracer):
        NullTracer INHERITS from Tracer (the parenthesized name after the
        class name is its "base class" / "parent class"). This means
        NullTracer automatically has every method Tracer defines UNLESS it
        explicitly overrides that method with its own version — which it
        does for every single method below (__init__, record_llm,
        record_retrieval, note, flush), replacing each one with a version
        that immediately returns and does nothing. The benefit: any code
        elsewhere that expects "an object with a .record_llm() method" (a
        Tracer) works identically whether it was actually handed a real
        Tracer or a NullTracer — the caller never needs an `if tracing_is_on`
        check anywhere.
    @property
        A decorator (see agents/gathering.py's docstring for what a
        decorator is) that lets you call a method WITHOUT parentheses, as if
        it were a plain attribute: `tracer.enabled` instead of
        `tracer.enabled()`. It's used here so callers can write
        `if tracer.enabled:` and get real, computed behaviour (in
        NullTracer's case, always False) rather than exposing a raw
        attribute that some code path could accidentally overwrite.
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
        """
        CALLED BY   cli.py::main (once per CLI invocation, only when
                    --debug or DEBUG_TRACE is set — otherwise a NullTracer
                    is built instead and this __init__ never runs).
        """
        self.run_id = run_id
        # pathlib.Path wraps a filesystem path string in an object with
        # convenient methods (like .mkdir() and the "/" operator used in
        # flush() below to join a directory and a filename) — it's the
        # modern, cross-platform-safe alternative to manually concatenating
        # path strings.
        self._dir = pathlib.Path(log_dir)
        self._entries: List[str] = []

    @property
    def enabled(self) -> bool:
        return True

    def record_llm(self, source_label: str, node: Optional[str],
                   prompt_messages: List[Dict[str, str]], response: str,
                   prompt_tokens: Optional[int], completion_tokens: Optional[int],
                   latency_s: float) -> None:
        """Record one LLM call: which provider, the exact prompt, raw response.

        CALLED BY   llm/client.py::OpenAICompatibleClient.complete (and
                    StubClient.complete in stub mode) — right after each
                    provider call returns, success or not.
        Nothing is written to disk here — this only APPENDS a formatted
        string to the in-memory self._entries list. The actual file write
        happens once, in flush(), at the very end of the run.
        """
        header = (f"RETRIEVED FROM {source_label.upper()}"
                  f"{f'  |  node={node}' if node else ''}"
                  f"  |  {latency_s:.2f}s"
                  f"  |  prompt_tok={prompt_tokens}  completion_tok={completion_tokens}")
        # This builds one line per message in the prompt transcript, each
        # prefixed with its role (e.g. "[system]" or "[user]"). The
        # generator expression inside join(...) below is like a list
        # comprehension but never builds an intermediate list in memory —
        # join() consumes the values one at a time as it stitches them
        # together with "\n" between each.
        prompt_text = "\n".join(
            f"[{m.get('role','?')}]\n{m.get('content','')}" for m in prompt_messages)
        self._entries.append(
            f"{_BANNER}\n{header}\n{_BANNER}\n"
            f"PROMPT:\n{prompt_text}\n"
            f"{'-'*78}\n"
            f"RESPONSE:\n{response}\n")

    def record_retrieval(self, source_label: str, query: str,
                         hits: List[Dict[str, Any]]) -> None:
        """Record one retrieval call: which engine, the query, the raw hits.

        CALLED BY   storage/qdrant_store.py::QdrantStore.search and
                    storage/opensearch_store.py::OpenSearchStore.search —
                    right after each returns its raw hit list.
        """
        header = (f"RETRIEVED FROM {source_label.upper()}"
                  f"  |  query={query!r}  |  {len(hits)} hits")
        body_lines: List[str] = []
        for i, h in enumerate(hits, 1):
            # {k: v for k, v in h.items() if k not in (...)} is a DICT
            # COMPREHENSION: it builds a brand-new dict by looping over
            # h.items() (every key/value pair in h) and keeping only the
            # pairs whose key is NOT "vector" or "embedding" — those two
            # fields would be long lists of floating-point numbers and
            # would make the trace file huge and unreadable for no benefit.
            slim = {k: v for k, v in h.items() if k not in ("vector", "embedding")}
            body_lines.append(f"[hit {i}] " + json.dumps(slim, default=str, ensure_ascii=False))
        # "\n".join(body_lines) or "(no hits)":  if body_lines is an empty
        # list, "\n".join([]) evaluates to "" (an empty string), which is
        # FALSY in Python, so the `or` falls through to the string
        # "(no hits)" instead. This is a common Python idiom for "use this
        # value unless it's empty/None/zero/False, in which case use this
        # fallback instead."
        body = "\n".join(body_lines) or "(no hits)"
        self._entries.append(f"{_BANNER}\n{header}\n{_BANNER}\n{body}\n")

    def flush(self) -> Optional[str]:
        """Write all accumulated entries to logs/trace-<run_id>.txt. Returns
        the path written, or None if there was nothing to write.

        CALLED BY   cli.py::main, exactly once, right after the graph
                    invoke loop finishes (whether the run completed
                    normally or went through a human-escalation pause/
                    resume cycle).
        """
        if not self._entries:
            return None
        # mkdir(parents=True, exist_ok=True): create the "logs" directory
        # (and any missing parent directories) if it doesn't already exist;
        # exist_ok=True means "don't raise an error if it's already there."
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"trace-{self.run_id}.txt"
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        # "with open(...) as f:" is a CONTEXT MANAGER — it opens the file,
        # runs the indented block below with `f` bound to the open file
        # object, and GUARANTEES the file gets closed afterward even if an
        # exception is raised inside the block. This is the standard,
        # safe way to work with files in Python; writing open()/close()
        # by hand risks leaving the file open if something goes wrong
        # in between.
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"DEBUG TRACE  |  run_id={self.run_id}  |  {stamp}\n\n")
            f.write("\n".join(self._entries))
        return str(path)


class NullTracer(Tracer):
    """No-op tracer used when tracing is disabled. Every recorder returns
    immediately; nothing is stored or written, so the non-debug path pays no
    formatting or I/O cost.

    See the module docstring above for what it means that this class
    INHERITS from Tracer and overrides every one of its methods.
    """

    def __init__(self) -> None:  # noqa: D107 — intentionally skips base init
        # Deliberately does NOT call Tracer.__init__(self, ...) (there is no
        # super().__init__() here) — NullTracer never needs self.run_id or
        # self._entries because none of its methods below ever read or
        # write them. Skipping the parent's __init__ entirely is safe only
        # because every method is also overridden; if a future change added
        # a new Tracer method without overriding it here too, NullTracer
        # would break when that inherited method tried to use attributes
        # that were never set.
        pass

    @property
    def enabled(self) -> bool:
        return False

    def record_llm(self, *a: Any, **k: Any) -> None:
        # *a collects any number of positional arguments, **k collects any
        # number of keyword arguments (see logging_setup.py's docstring for
        # the **kwargs explanation) — together they mean "accept absolutely
        # anything the caller passes, and ignore all of it." This lets
        # NullTracer's method signatures stay in sync with Tracer's real
        # ones without having to repeat every parameter name here.
        return None

    def record_retrieval(self, *a: Any, **k: Any) -> None:
        return None

    def flush(self) -> Optional[str]:
        return None

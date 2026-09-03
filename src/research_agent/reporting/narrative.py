"""
reporting/narrative.py -- the human-readable execution-narrative report
generator (S-2), split out of logging_setup.py.

Purpose:
    Render ONE run's buffered log_event() calls as prose --
    logs/run-<run_id>.txt -- when narrative capture is enabled (--debug /
    DEBUG_TRACE=true; see tracing.py::Tracer, which is the on/off switch
    for this module, not a second recorder). This is presentation only:
    every fact it renders was already decided by whichever node or
    routing function called log_event() -- this module groups, labels,
    and formats, it never decides what happened or why.

Why split from logging_setup.py:
    logging_setup.py is imported by every module in this codebase for
    log_event()/run_id_var alone. Before this split, importing it also
    pulled in ~800 lines of narrative report-generation code that only
    tracing.py ever actually used -- an import-order coupling on the
    project's most fundamental utility for a feature most call sites
    never touch. Nothing else changed: this module reads the exact same
    LogRecords log_event() already produces, via the same
    NarrativeBufferHandler mechanism.

CALLED BY   tracing.py::Tracer, via enable_narrative_logging() (turns the
            handler on, once per process) and flush_narrative() (renders
            and writes one run's buffer, at the end of that run).
"""

import logging
import pathlib
import time

from research_agent.logging_setup import run_id_var


class _Event:
    """One buffered narrative event: (timestamp, provenance, event name,
    fields, level). This is what NarrativeBufferHandler now stores instead
    of pre-rendered text — see its docstring for why."""

    __slots__ = ("ts", "where", "msg", "fields", "level")

    def __init__(self, ts: float, where: str, msg: str, fields: dict,
                level: str = "INFO") -> None:
        self.ts = ts
        self.where = where
        self.msg = msg
        self.fields = fields
        self.level = level


class NarrativeFormatter(logging.Formatter):
    """Render buffered _Event objects as prose — the "presentation, not
    instrumentation" half of this file's design (see module docstring).

    This formatter owns presentation ONLY: it groups, labels, and
    summarizes events that business code already recorded via log_event();
    it never decides what happened or why — that decision is always
    something a node or routing function already wrote into a field
    (route.decision's `reason`, node.progress's `recall`, etc.). Nothing
    here should ever need to change because a NODE's behavior changed;
    only because the PRESENTATION of an already-recorded fact should
    change. If a new architectural decision point is added to the graph,
    it becomes visible here automatically the moment it emits a
    "route.decision" or one of the _DECISION_EVENTS below — no node name
    needs to be added to this file.

    Grouping logic (single pass over one run's buffered events, done once
    in render_all() rather than per-line, which is why NarrativeBufferHandler
    buffers raw events instead of pre-rendered text — see its docstring):
      - Events are grouped into "spans," one per node.enter, EXCEPT that
        consecutive node.enter events for the SAME node (the parallel
        search_worker fan-out) collapse into one "node x N" group.
      - A span gets the richer INPUT/DECISION/TRANSITION rendering ONLY if
        it contains one of _DECISION_EVENTS (a node's own "here's what I
        decided" event) or a "route.decision" — i.e. detected from WHICH
        events are present, never from a hardcoded node-name list.
      - "Gather Loop N" / "Critique attempt N" headings and phase-exit
        summaries are inserted at the route.decision events that are
        already the graph's own record of a loop continuing or exiting
        (dispatch_tasks's from_node/route_convergence's reason) — nothing
        new is inferred, only surfaced more visibly.
      - An EXECUTION TIMELINE + SUMMARY closes the file, built from the
        same per-span labels and already-existing run.telemetry counters.
    """

    _BANNER = "=" * 78
    _DIVIDER = "-" * 78
    _THIN = "-" * 62

    # Events that ARE a node's own "here's what I decided" record — a span
    # containing one of these is treated as an architectural decision
    # point, regardless of which node happened to emit it. This is the
    # "generic, not a 6-node list" detection the design calls for: today
    # this covers classify/task_expander/progress_checker/gap_generator/
    # critic/compiler because those are the six nodes that emit one of
    # these; a future node that emits its own summary event this way gets
    # the same treatment automatically.
    _DECISION_EVENTS = {"node.classify", "node.expand", "node.progress",
                        "node.gaps", "node.critique", "node.compiled"}

    # Context fields worth carrying forward into a later decision span's
    # INPUT section — e.g. goal_manager's INPUT is "what memory_retrieve
    # and classify already found," not anything goal_manager itself
    # produced. Updated as render_all() scans forward; each span's INPUT
    # section is whatever of these is non-empty at the moment that span
    # starts.
    _CONTEXT_KEYS = ("intent", "memory_hits", "recall", "depth",
                     "tasks_produced", "revision_count")

    # Nodes LangGraph can genuinely dispatch as a parallel batch (via Send —
    # see orchestration/graph.py::dispatch_tasks). A node NOT in this set
    # that happens to enter twice in a row (human_escalation: pause, then
    # resume after a human decision) is a sequential re-entry, not a
    # fan-out, and must render as two separate spans, not one "x N" group.
    _FANOUT_NODES = {"search_worker", "mcp_search_worker"}

    # ---- single-event rendering (used both standalone and inside spans) --

    def render_event(self, ev: "_Event") -> str:
        if ev.level in ("WARNING", "ERROR", "CRITICAL"):
            return self._render_alert(ev)
        if ev.msg == "llm.call" and "prompt_messages" in ev.fields:
            return self._render_llm_call(ev.fields, ev.where, ev.ts)
        if ev.msg == "retrieval.raw" and "hits" in ev.fields:
            return self._render_retrieval(ev.fields, ev.where)
        if ev.msg == "route.decision":
            return self._render_route(ev.fields)
        if ev.msg.startswith("graph."):
            return self._render_graph(ev.msg, ev.fields)
        if ev.msg == "run.telemetry":
            return self._render_telemetry(ev.fields)
        rest = "  ".join(f"{k}={v}" for k, v in ev.fields.items())
        label = self._PROSE.get(ev.msg, ev.msg)
        return f"[{ev.where}] {label}" + (f"  {rest}" if rest else "")

    def _render_alert(self, ev: "_Event") -> str:
        bang = "!" * 30
        label = self._PROSE.get(ev.msg, ev.msg)
        shown = {k: v for k, v in ev.fields.items() if k not in ("run_id",)}
        width = max([len("Event"), len("Where")]
                    + [len(k.replace("_", " ")) for k in shown], default=5)
        lines = [bang, ev.level, bang,
                f"{'Event':<{width}} : {label}",
                f"{'Where':<{width}} : {ev.where}"]
        if ev.msg == "llm.truncated_runaway_generation":
            lines.append(f"{'Action':<{width}} : Output truncated from "
                         f"{shown.get('raw_chars')} to {shown.get('kept_chars')} chars")
        for k, v in shown.items():
            lines.append(f"{k.replace('_', ' ').title():<{width}} : {v}")
        return "\n".join(lines)

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        # Kept for direct single-record use (e.g. a future handler that
        # wants one line at a time); render_all() below is the path
        # flush_narrative() actually uses.
        fields = dict(getattr(record, "event_fields", None) or {})
        where = f"{record.name}::{record.funcName}:{record.lineno}"
        return self.render_event(_Event(record.created, where, record.getMessage(), fields))

    def _render_telemetry(self, f: dict) -> str:
        lines = [self._BANNER, "TELEMETRY", self._BANNER]
        # D-145: the composed verdict opens the block, with EVERY reason --
        # this is the artifact someone reads after the fact, so unlike
        # cli.py's one-line summary there is no need to truncate the list.
        confidence = f.get("confidence")
        if confidence:
            lines.append("\nConfidence")
            lines.append(self._THIN)
            lines.append(f"Verdict          : {confidence.get('band')} "
                         f"({confidence.get('score')}%)")
            for reason in confidence.get("reasons") or []:
                lines.append(f"  - {reason}")
        lines.append("\nAttribution")
        lines.append(self._THIN)
        lines.append(f"Goals cited      : {f.get('evidence_cited')}")
        if f.get("citations_attached"):
            lines.append(f"  attached deterministically (D-144): "
                         f"{f.get('citations_attached')}")
        lines.append(f"Sources listed   : {f.get('web_sources_listed')} "
                     f"({f.get('web_sources_suppressed')} suppressed)")
        lines.append("\nResearch")
        lines.append(self._THIN)
        lines.append(f"Intent           : {f.get('intent')}")
        lines.append(f"Goals            : {f.get('goals')}")
        lines.append(f"Evidence items   : {f.get('evidence_items')}")
        lines.append(f"Grounding ratio  : {f.get('grounding_ratio')}")
        if f.get("goals_without_evidence"):
            lines.append(f"Goals w/o evidence: {f.get('goals_without_evidence')}")
        lines.append("\nSearch")
        lines.append(self._THIN)
        lines.append(f"Search calls     : {f.get('search_calls')} "
                     f"({f.get('search_failures')} failed)")
        lines.append(f"Dense retrievals : {f.get('retrieval_dense_calls')}")
        lines.append(f"Keyword retrievals: {f.get('retrieval_keyword_calls')}")
        lines.append(f"Gather laps      : {f.get('iterations')}")
        lines.append("\nMemory")
        lines.append(self._THIN)
        lines.append(f"Hits             : {f.get('memory_hits')}")
        lines.append(f"Writes           : {f.get('memory_writes')}")
        lines.append("\nLLM")
        lines.append(self._THIN)
        lines.append(f"Provider calls   : {f.get('llm_provider_calls')} "
                     f"({f.get('llm_fallback_hops')} fallback hops)")
        lines.append(f"Quality checks   : {f.get('llm_quality_calls')} "
                     f"({f.get('llm_quality_calls_failed')} failed)")
        # D-108: what the judge DECIDED, not just how often it was asked.
        # Imported at call time rather than at module import: cli.py
        # already imports this module (for Tracer -> narrative logging),
        # and a module-level import back into cli would be circular.
        from research_agent.cli import _fmt_judge_line
        lines.append(_fmt_judge_line(f))
        lines.append("\nCritique")
        lines.append(self._THIN)
        status = "PASSED" if f.get("critique_passed") else "FAILED"
        lines.append(f"Result           : {status}")
        lines.append(f"Revision cycles  : {f.get('revision_cycles')}")
        if f.get("escalations"):
            lines.append("\nEscalations")
            lines.append(self._THIN)
            for esc in f["escalations"]:
                lines.append(f"{esc.get('trigger')}: {esc.get('action')}")
        if f.get("planning_error"):
            lines.append(f"\nPlanning error   : {f.get('planning_error')}")
        return "\n".join(lines)

    def _fmt_ts(self, ts: float) -> str:
        base = time.strftime("%H:%M:%S", time.localtime(ts))
        millis = int((ts - int(ts)) * 1000)
        return f"{base}.{millis:03d}"

    def _render_llm_call(self, fields: dict, where: str, ts: float) -> str:
        self._llm_call_counter += 1
        latency = fields.get("latency_s") or 0.0
        finished = self._fmt_ts(ts)
        started = self._fmt_ts(ts - latency)
        prompt_text = "\n".join(
            f"[{m.get('role', '?')}]\n{m.get('content', '')}"
            for m in fields.get("prompt_messages", []))
        header = (f"{self._BANNER}\nLLM REQUEST  (Call #{self._llm_call_counter})  [{where}]\n"
                 f"{self._BANNER}\n"
                 f"Metadata\n{self._THIN}\n"
                 f"Model      : {fields.get('provider')} / {fields.get('model')}\n"
                 f"Node       : {fields.get('node')}\n"
                 f"Started    : {started}\n"
                 f"Finished   : {finished}\n"
                 f"Elapsed    : {latency:.2f}s\n"
                 f"Tokens     : {fields.get('prompt_tokens')} -> {fields.get('completion_tokens')}\n"
                 f"Prompt size: {len(prompt_text)} chars")
        response = fields.get("response", "")
        return (f"{header}\n\n"
                f"Prompt\n{self._THIN}\n{prompt_text}\n{self._DIVIDER}\n"
                f"Response ({len(response)} chars)\n{self._THIN}\n{response}")

    def _render_retrieval(self, fields: dict, where: str) -> str:
        hits = fields.get("hits") or []
        meta = (f"Source     : {fields.get('source')}\n"
               f"Query      : {fields.get('query')!r}\n"
               f"Hits       : {len(hits)}")
        body = "\n".join(self._render_hit(i, h) for i, h in enumerate(hits, 1)) or "(no hits)"
        return (f"{self._BANNER}\nSEARCH RESULTS  [{where}]\n{self._BANNER}\n"
                f"Metadata\n{self._THIN}\n{meta}\n\n"
                f"Results\n{self._THIN}\n{body}")

    def _render_hit(self, i: int, h: dict) -> str:
        # Structured rendering ONLY when a hit has at least one field this
        # design recognizes (similarity/bm25_score/age_days/title/topic/
        # content) — the SAME keys every retrieval call site already
        # produces (storage/qdrant_store.py, storage/opensearch_store.py,
        # memory/semantic_memory.py). Any hit shape this doesn't recognize
        # falls back to the raw dict, unchanged — nothing is ever hidden.
        if not isinstance(h, dict) or not ({"similarity", "bm25_score", "age_days",
                                            "title", "topic", "content"} & h.keys()):
            return f"[hit {i}] {h}"
        parts = [f"[hit {i}]"]
        if "similarity" in h:
            parts.append(f"similarity={h['similarity']:.2f}" if isinstance(
                h["similarity"], (int, float)) else f"similarity={h['similarity']}")
        if "bm25_score" in h:
            parts.append(f"bm25={h['bm25_score']:.2f}" if isinstance(
                h["bm25_score"], (int, float)) else f"bm25={h['bm25_score']}")
        if "age_days" in h:
            parts.append(f"age={h['age_days']:.1f}d" if isinstance(
                h["age_days"], (int, float)) else f"age={h['age_days']}")
        if "topic" in h:
            parts.append(f"topic={h['topic']}")
        header = "  ".join(parts)
        content = (h.get("content") or h.get("title") or "")
        content = content[:200] + ("..." if len(content) > 200 else "")
        return f"{header}\n    {content}" if content else header

    def _render_route(self, fields: dict) -> str:
        state_bits = "  ".join(
            f"{k}={v}" for k, v in fields.items()
            if k not in ("from_node", "to_node", "reason"))
        return (f"{self._BANNER}\nROUTING DECISION\n\n"
                f"{fields.get('from_node')}\n      |\n      v\n{fields.get('to_node')}\n\n"
                f"Reason:\n{fields.get('reason')}\n"
                + (f"\nstate: {state_bits}\n" if state_bits else "")
                + self._BANNER)

    def _render_graph(self, msg: str, fields: dict) -> str:
        # Fallback for a stray graph.* event outside the grouped block below
        # (shouldn't normally happen — render_all groups all of them).
        rest = "  ".join(f"{k}={v}" for k, v in fields.items())
        return f"[GRAPH BUILD] {msg}" + (f"  {rest}" if rest else "")

    def _render_graph_build(self, gevents: list) -> str:
        """Collapse the whole graph-construction block (state creation,
        every node registered, every edge registered, final compile) into
        grouped bullet summaries instead of one line per registration —
        same information, far less repetition for what is otherwise 14+
        near-identical lines.
        """
        lines = [self._BANNER, "GRAPH CONSTRUCTION", self._BANNER, ""]
        state_ev = next((e for e in gevents if e.msg == "graph.state_created"), None)
        if state_ev is not None:
            lines.append(f"Graph state created ({state_ev.fields.get('state_model')})")
            lines.append("")
        node_evs = [e for e in gevents if e.msg == "graph.node_registered"]
        if node_evs:
            lines.append(f"Nodes registered ({len(node_evs)}):")
            lines.extend(f"  - {e.fields.get('node')}" for e in node_evs)
            lines.append("")
        edge_evs = [e for e in gevents if e.msg == "graph.edge_registered"]
        if edge_evs:
            cond = [e for e in edge_evs if e.fields.get("edge_type") == "conditional"]
            lines.append(f"Edges registered ({len(edge_evs)} total, {len(cond)} conditional):")
            for e in edge_evs:
                if e.fields.get("edge_type") == "conditional":
                    dests = ", ".join(e.fields.get("destinations") or [])
                    lines.append(f"  - {e.fields.get('from_node')} -> [{dests}]"
                                 f"  (router: {e.fields.get('router')})")
                else:
                    lines.append(f"  - {e.fields.get('from_node')} -> {e.fields.get('to_node')}")
            lines.append("")
        compiled_ev = next((e for e in gevents if e.msg == "graph.compiled"), None)
        if compiled_ev is not None:
            f = compiled_ev.fields
            lines.append(f"Graph compiled: {f.get('nodes')} nodes, {f.get('edges')} edges "
                        f"({f.get('conditional_edges')} conditional)")
        return "\n".join(lines)

    # Prose labels for event names that otherwise read as raw internal
    # identifiers — purely a presentation lookup, no new semantics: each
    # entry names the SAME event log_event() already recorded, just in
    # words instead of a dotted code. Events not listed here fall back to
    # their raw name unchanged (better an honest raw name than a guessed
    # translation for something this table wasn't specifically written for).
    _PROSE = {
        "node.enter": "Entering node",
        "llm.fallback": "Provider fallback",
        "llm.served_by_fallback": "Served by fallback provider",
        "llm.truncated_runaway_generation": "Truncated a runaway generation",
        "llm.truncated_by_token_limit": "Generation cut off at a token limit",
        "llm.http_error": "Provider rejected the request (HTTP error)",
        "llm.quality_reject": "Quality gate rejected the response",
        "llm.quality_scored": "Quality gate scored the response",
        "llm.last_provider_worse": "Last provider scored worse, kept earlier answer",
        "llm.skipped_for_context": "Provider skipped (prompt exceeds its context window)",
        "llm.context_overflow": "Provider refused the prompt (context overflow)",
        "quality.judge_unreliable": "Quality judge failed on every attempt",
        "run_history.skipped": "Run-history row skipped (Postgres unreachable)",
        "llm.chain_exhausted_low_quality": "Fallback chain exhausted (low quality)",
        "llm.chain_built": "LLM fallback chain built",
        "escalation.raised": "Escalation raised",
        "escalation.suppressed": "Escalation suppressed",
        "escalation.stub": "Escalation stub (HITL disabled)",
        "escalation.resumed": "Escalation resumed",
        "escalation.unknown_trigger": "Unknown escalation trigger",
        "memory.retrieved": "Memory retrieved",
        "memory.stored": "Memory stored",
        "memory.below_quality_floor": "Memory items below quality floor",
        "retrieval.hybrid": "Hybrid retrieval",
        "retrieval.no_results": "No retrieval results",
        "retrieval.below_floor": "Retrieval below similarity floor",
        "retrieval.leg_failed": "Retrieval leg failed",
        "chain.answered": "Retrieval chain answered",
        "chain.attempt": "Retrieval attempt started",
        "chain.tier_failed": "Retrieval tier failed",
        "worker.done": "Search worker finished",
        "worker.failed": "Search worker failed",
        "worker.contract_violation": "Worker contract violation",
        "producer.reject": "Producer rejected a malformed item",
        "run.telemetry": "Run telemetry",
        "app.degraded_checkpointing": "Degraded checkpointing (Postgres unreachable)",
        "checkpointer.pool_active": "Checkpointer pool active",
        "checkpointer.postgres_active": "Postgres checkpointer active",
        "checkpointer.memory_fallback": "Checkpointer fell back to in-memory",
        "checkpointer.closed": "Checkpointer closed",
        "langfuse.client_active": "Langfuse client active",
        "mcp.call_tool_task_started": "MCP tool call started",
        "mcp.call_tool_task_finished": "MCP tool call finished",
        "quality.score_failed": "Quality scoring failed",
    }

    def _decision_text(self, msg: str, fields: dict) -> str:
        if msg == "node.classify":
            return f"Intent classified: {fields.get('intent')}"
        if msg == "node.expand":
            extra = f", {fields['rejected']} rejected" if fields.get("rejected") else ""
            return f"{fields.get('produced')} search task(s) queued{extra}"
        if msg == "node.progress":
            recall = fields.get("recall")
            pct = f"{recall * 100:.0f}%" if isinstance(recall, (int, float)) else recall
            return f"Coverage {pct} at depth {fields.get('depth')}"
        if msg == "node.gaps":
            extra = f", {fields['rejected']} rejected" if fields.get("rejected") else ""
            return f"{fields.get('produced')} new search task(s) queued{extra}"
        if msg == "node.critique":
            outcome = "PASSED" if fields.get("passed") else "FAILED"
            return f"Critique {outcome} (revision {fields.get('revision')})"
        if msg == "node.compiled":
            cited = fields.get("evidence_cited")
            cited_text = (f"{cited} goal(s) cited (no [gN] markers found in report)"
                         if cited == 0 else f"{cited} goal(s) cited")
            return (f"Report compiled: {fields.get('sections')} section(s), "
                    f"{cited_text}, {fields.get('output_chars')} chars")
        return msg

    # ---- whole-run rendering (single pass, this is what flush_narrative calls) --

    def render_problems(self, events: list) -> str:
        """Every WARNING and above in one block, for the top of the file.

        D-117. D-116 put the warnings back into the narrative, but in
        their chronological place -- which for run p205.265-check meant a
        403 saying "this team has no credits" sitting on line ~900 of a
        2,300-line file. Being present is not the same as being found.

        This is a SUMMARY, not a second copy of the truth: every entry
        here also appears in full, in order, in the body below. It exists
        so an administrator opening the file sees what went wrong before
        deciding whether to read the rest.

        Renders positively when there is nothing to report. "No warnings"
        is a real result, and its absence would be indistinguishable from
        a section that failed to render.
        """
        alerts = [e for e in events
                  if e.level in ("WARNING", "ERROR", "CRITICAL")]
        lines = [self._BANNER, f"PROBLEMS ({len(alerts)})", self._BANNER]
        if not alerts:
            lines.append("")
            lines.append("None. No WARNING or ERROR was logged during this run.")
            return "\n".join(lines)
        lines.append("")
        lines.append("Every entry below also appears in full, in order, in the")
        lines.append("narrative body. Listed here so it is not missed.")
        # Grouped by event name and kept in first-seen order: three
        # identical context skips are one problem seen three times, and
        # collapsing them stops a repeated event from burying a singular
        # one further down.
        seen: dict = {}
        for e in alerts:
            seen.setdefault(e.msg, []).append(e)
        for msg, group in seen.items():
            first = group[0]
            label = self._PROSE.get(msg, msg)
            count = f"  (x{len(group)})" if len(group) > 1 else ""
            lines.append("")
            lines.append(f"  [{first.level}] {label}{count}")
            # The fields an operator acts on, named first and in a fixed
            # order, then everything else. `body` last because it is the
            # long one and reads better as the tail of the entry.
            shown = {k: v for k, v in first.fields.items() if k != "run_id"}
            # Width from the widest key actually present, the same way
            # _render_alert sizes its own column -- a fixed width lines up
            # "model" and misaligns "provider", which is the sort of thing
            # that makes a summary look untrustworthy.
            width = max([len("event")] + [len(k) for k in shown])
            lines.append(f"      {'event':<{width}} : {msg}")
            for key in ("provider", "model", "node", "status", "kind",
                        "hint", "effect"):
                if shown.get(key) not in (None, ""):
                    lines.append(f"      {key:<{width}} : {shown.pop(key)}")
            body = shown.pop("body", None)
            for k, v in shown.items():
                lines.append(f"      {k:<{width}} : {v}")
            if body:
                lines.append(f"      {'body':<{width}} : {body}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # render_all, in three phases (S-12)
    #
    # This used to be ONE 247-line method at cyclomatic complexity 74 and
    # nesting depth 7 -- by a factor of 1.7 the most complex block in the
    # codebase, and the only depth-7 one. It did six jobs at once:
    # partition the event stream, group fan-out spans, choose and inline
    # three different renderers, keep gather-lap/critique-attempt phase
    # bookkeeping, maintain a rolling `context` so a span's INPUT shows the
    # PREVIOUS span's output, and build the `timeline` two later renderers
    # consume. Six mutable variables were threaded through one loop, and
    # the plan preview had to be written as a `None` placeholder into `out`
    # and back-patched at the end -- the tell that the loop needed a value
    # it could only compute after itself.
    #
    # Split into partition -> plan -> render. The first two are pure and
    # independently testable; the back-patch is gone because planning
    # finishes before rendering starts. NOTHING about the output changed --
    # that is the point, and tests/unit/test_reporting_narrative.py is what
    # holds it to that.
    # ------------------------------------------------------------------

    def _partition(self, events: list) -> tuple:
        """Phase 1 (pure): split a run's events into their three regions.

        RETURNS (prelude, graph_blocks, orphans, spans) where
                prelude      events before the first node.enter/graph.*
                graph_blocks consecutive runs of graph.* events
                orphans      events in the span region before any node.enter
                             -- surfaced rather than silently dropped, which
                             is what the original's `current is None` arm did
                spans        [{"node", "count", "events"}], consecutive
                             node.enter's for the SAME fan-out node folded
                             into one span with count > 1
        """
        prelude, graph_blocks, orphans, spans = [], [], [], []
        i, n = 0, len(events)

        while i < n and not (events[i].msg == "node.enter"
                             or events[i].msg.startswith("graph.")):
            prelude.append(events[i])
            i += 1

        while i < n and events[i].msg.startswith("graph."):
            graph_start = i
            while i < n and events[i].msg.startswith("graph."):
                i += 1
            graph_blocks.append(events[graph_start:i])

        current = None
        while i < n:
            ev = events[i]
            if ev.msg == "node.enter":
                node = ev.fields.get("node")
                if (current is not None and current["node"] == node
                        and node in self._FANOUT_NODES):
                    current["count"] += 1
                    current["events"].append(ev)
                else:
                    if current is not None:
                        spans.append(current)
                    current = {"node": node, "count": 1, "events": [ev]}
            elif current is not None:
                current["events"].append(ev)
            else:
                # An event before any node.enter fired in this segment
                # (shouldn't normally happen once past startup).
                orphans.append(ev)
            i += 1
        if current is not None:
            spans.append(current)
        return prelude, graph_blocks, orphans, spans

    @staticmethod
    def _fold_context(gevents: list, context: dict) -> int:
        """Fold one span's events into the rolling context. Returns how many
        tasks this span's producer emitted, for the run-level total.

        CALLED BY _plan, AFTER a span is planned, so the NEXT span's INPUT
        shows this span's output and never itself. That ordering is the
        whole reason the context is threaded rather than recomputed.
        """
        produced = 0
        for e in gevents:
            if e.msg == "node.classify":
                context["intent"] = e.fields.get("intent")
            elif e.msg == "memory.retrieved":
                context["memory_hits"] = e.fields.get("count")
            elif e.msg == "node.progress":
                context["recall"] = e.fields.get("recall")
                context["depth"] = e.fields.get("depth")
            elif e.msg in ("node.expand", "node.gaps"):
                context["tasks_produced"] = e.fields.get("produced")
                produced += e.fields.get("produced") or 0
            elif e.msg == "node.critique":
                context["revision_count"] = e.fields.get("revision")
        return produced

    def _decision_text_for(self, decision_ev, route_ev) -> "str | None":
        """The DECISION line for a span, or None if it has no story to tell.

        Two sources, in priority order: a real decision event gets its
        prose form (_decision_text / _PROSE); failing that, a bare
        route.decision is summarised from whatever fields it carries BEYOND
        the four the TRANSITION block already prints. When that leaves
        nothing, the caller falls back to printing the route's reason in
        full -- see _render_decision_span.
        """
        if decision_ev is not None:
            return self._decision_text(decision_ev.msg, decision_ev.fields)
        if route_ev is None:
            return None
        extra = {k: v for k, v in route_ev.fields.items()
                 if k not in ("from_node", "to_node", "reason",
                              "escalation_trigger")}
        return (", ".join(f"{k}: {v}" for k, v in extra.items())
                if extra else None)

    @staticmethod
    def _phase_boundaries(route_ev, laps: dict) -> tuple:
        """Loop/critique boundary bookkeeping for one span.

        CALLED BY _plan, once per span. MUTATES `laps` -- the gather-lap
        counter, the critique-attempt counter and the "the next fan-out
        starts a new lap" flag -- and RETURNS (trailers, exits): the phase
        summaries that follow this span, and the __EXIT__ markers its
        timeline entry is followed by.

        Driven entirely by the SAME route.decision fields _render_route
        already uses, never a new inference. Pulled out of _plan because
        these seven conditions were the bulk of its complexity while being
        the part least entangled with everything else: they read one event
        and three integers.
        """
        trailers: list = []
        exits: list = []
        if route_ev is None:
            return trailers, exits
        frm = route_ev.fields.get("from_node")
        to = route_ev.fields.get("to_node")
        exit_marker = (f"__EXIT__:{route_ev.fields.get('reason')}",
                       None, route_ev.ts)

        if frm == "gap_generator" and to not in ("compiler",
                                                 "human_escalation"):
            laps["gather"] += 1
            laps["pending_label"] = True
        # Two ways the gather loop can end: progress_checker reaching
        # target/depth, or gap_generator's own dispatch escalating (E2/E3,
        # task supply exhausted). Same summary either way.
        gather_done = (
            (frm == "progress_checker"
             and to in ("compiler", "human_escalation"))
            or (frm == "gap_generator" and to == "human_escalation"))
        if gather_done:
            trailers.append(("GATHER PHASE", laps["gather"], route_ev.fields))
            exits.append(exit_marker)

        if frm == "critic" and to == "compiler":
            laps["critique"] += 1
        if frm == "critic" and to in ("memory_writer", "telemetry",
                                      "human_escalation"):
            trailers.append(("CRITIQUE PHASE", laps["critique"] + 1,
                             route_ev.fields))
            exits.append(exit_marker)
        return trailers, exits

    def _plan(self, spans: list) -> tuple:
        """Phase 2 (pure): everything the renderer needs to know, decided
        before any string is built.

        RETURNS (planned, timeline, tasks_generated_total).

        This is where the cross-span bookkeeping lives, so that phase 3
        carries none of it: the rolling `context` (see _fold_context), the
        gather-lap and critique-attempt counters (see _phase_boundaries),
        and the `timeline` that the plan preview and the final timeline
        both read.

        Each planned span carries its own `trailers` -- the phase summaries
        that follow it -- rather than the renderer re-deriving them from
        route fields it would otherwise have to inspect twice.
        """
        planned: list = []
        timeline: list = []
        context: dict = {}
        laps = {"gather": 1, "critique": 0, "pending_label": True}
        tasks_generated_total = 0

        for gi, g in enumerate(spans):
            node, count, gevents = g["node"], g["count"], g["events"]

            # Context snapshot BEFORE this span's own events are folded in,
            # so a span's decision is never mistaken for its input.
            span_input = {k: context[k] for k in self._CONTEXT_KEYS
                          if k in context}
            enter_ev = next((e for e in gevents if e.msg == "node.enter"), None)
            if enter_ev is not None and enter_ev.fields.get("query"):
                span_input = {"query": enter_ev.fields["query"], **span_input}

            decision_ev = next((e for e in gevents
                                if e.msg in self._DECISION_EVENTS), None)
            route_ev = next((e for e in gevents
                             if e.msg == "route.decision"), None)
            decision_text = self._decision_text_for(decision_ev, route_ev)

            to_node = route_ev.fields.get("to_node") if route_ev else None
            if to_node is None and gi + 1 < len(spans):
                to_node = spans[gi + 1]["node"]

            if count > 1:
                kind = "fanout"
                if laps["pending_label"]:
                    timeline.append((f"__LOOP__:{laps['gather']}", None,
                                     gevents[0].ts))
                    laps["pending_label"] = False
                timeline.append((f"{node} x {count}", None, gevents[0].ts))
            elif decision_ev is not None or route_ev is not None:
                kind = "decision"
                timeline.append((node, decision_text, gevents[0].ts))
            else:
                kind = "plain"
                timeline.append((node, None, gevents[0].ts))

            trailers, exits = self._phase_boundaries(route_ev, laps)
            timeline.extend(exits)

            planned.append({
                "index": gi, "node": node, "count": count, "events": gevents,
                "kind": kind, "span_input": span_input,
                "decision_ev": decision_ev, "route_ev": route_ev,
                "decision_text": decision_text, "to_node": to_node,
                "reason": route_ev.fields.get("reason") if route_ev else None,
                "trailers": trailers,
            })

            tasks_generated_total += self._fold_context(gevents, context)

        return planned, timeline, tasks_generated_total

    def _span_header(self, node: str, index: int, total: int,
                     gevents: list, rule: str, *,
                     rule_under_title: bool) -> list:
        """The NODE:/Started/Finished/Elapsed block, defined ONCE.

        It used to be written out twice, ninety lines apart, and the two
        copies disagreed in BOTH ways -- which is exactly why one
        definition is worth having:

          decision span   rule / NODE / rule / Started / Finished / Elapsed
          plain span      rule / NODE / Started / Finished / Elapsed / rule

        and the rule itself is `_BANNER` (=) for one and `_THIN` (-) for
        the other. Neither difference is a bug and both are preserved
        byte-for-byte -- caught by diffing a real rendered narrative
        before and after this extraction, not by the unit tests, which do
        not assert on the header layout. `rule_under_title` is keyword-only
        so a call site can never silently pass the wrong shape positionally.
        """
        duration = gevents[-1].ts - gevents[0].ts
        title = [rule, f"NODE: {node}  [Step {index + 1}/{total}]"]
        stamps = [f"Started : {self._fmt_ts(gevents[0].ts)}",
                  f"Finished: {self._fmt_ts(gevents[-1].ts)}",
                  f"Elapsed : {duration:.2f}s"]
        if rule_under_title:
            return title + [rule] + stamps
        return title + stamps + [rule]

    def _render_decision_span(self, sp: dict, total: int) -> str:
        """An architectural decision point -- detected from event TYPE, not
        node name (see _DECISION_EVENTS)."""
        gevents = sp["events"]
        lines = self._span_header(sp["node"], sp["index"], total, gevents,
                                  self._BANNER, rule_under_title=True)
        if sp["span_input"]:
            lines.append("\nINPUT")
            lines.append(self._THIN)
            for k, v in sp["span_input"].items():
                lines.append(f"{k}: {v}")
        # Sub-events (LLM calls, retrievals) render HERE -- before the
        # Decision/Next lines below, not after -- because the decision is a
        # CONSEQUENCE of what these calls returned, not the other way
        # around. Chronological order, not arrival order in the log stream.
        for e in gevents:
            if (e is sp["decision_ev"] or e is sp["route_ev"]
                    or e.msg == "node.enter"):
                continue
            # D-116: the allowlist here was ("llm.call", "retrieval.raw")
            # plus graph.* -- three event names, and EVERYTHING else inside
            # a node span was silently dropped from the human-readable
            # narrative. That included every WARNING the run produced.
            #
            # Measured on run p205.265-check: 12 warnings were logged, 10 of
            # them never reached logs/run-*.txt -- among them the two
            # llm.http_error lines carrying a 403 whose body said, in plain
            # English, that the provider account had no credits.
            # render_event has routed WARNING and above to _render_alert's
            # banner since the file was written; nothing ever called it for
            # a warning raised inside a node.
            #
            # An allowlist of event NAMES cannot stay correct as events are
            # added -- D-110's llm.http_error did not exist when this list
            # was written. Severity does not have that problem: a warning is
            # a warning whatever it is called.
            if (e.level in ("WARNING", "ERROR", "CRITICAL")
                    or e.msg in ("llm.call", "retrieval.raw")
                    or e.msg.startswith("graph.")):
                lines.append("")
                lines.append(self.render_event(e))
        if sp["decision_text"]:
            lines.append("\nDECISION")
            lines.append(self._THIN)
            lines.append(sp["decision_text"])
            # The DECISION line above already explains WHY -- repeating it
            # as a "Reason:" under TRANSITION was pure duplication (review:
            # "Decision / Transition / Reason" all said the same thing three
            # times). Just name what's next.
            lines.append("\nNEXT")
            lines.append(self._THIN)
            lines.append(sp["to_node"] or "(next node)")
        else:
            # No dedicated DECISION text exists for this span (the generic
            # route.decision fallback had nothing extra to show) -- the
            # reason is the ONLY explanation available, so it stays, in full.
            lines.append("\nTRANSITION")
            lines.append(self._THIN)
            lines.append(f"{sp['node']}\n      |\n      v\n"
                         f"{sp['to_node'] or '(next node)'}")
            lines.append(f"\nReason:\n{sp['reason'] or '(fixed edge)'}")
        return "\n".join(lines)

    def _render_plain_span(self, sp: dict, total: int) -> list:
        """An ordinary node with no decision of its own (memory_retrieve,
        merger, memory_writer, telemetry, human_escalation): a plain
        heading plus whatever it logged. Returns a LIST of `out` entries,
        because its sub-events are separate blocks rather than one string.
        """
        gevents = sp["events"]
        header = self._span_header(sp["node"], sp["index"], total, gevents,
                                   self._THIN, rule_under_title=False)
        out = ["\n".join(header)]
        raw_hit_count = None
        for e in gevents:
            if e.msg == "node.enter":
                continue
            if (e.msg == "retrieval.raw"
                    and "memory" in str(e.fields.get("source", ""))):
                raw_hit_count = e.fields.get("hit_count")
                out.append(self.render_event(e))
                continue
            if e.msg == "memory.retrieved" and raw_hit_count is not None:
                # Clarifies a genuinely confusing pair: the retrieval
                # layer's raw hit count (Qdrant, no filtering) vs
                # semantic_memory.py's own count (after its dedup/quality
                # filtering) -- same two numbers already logged two events
                # apart, just stated as one sentence instead of two easily-
                # misread ones.
                kept = e.fields.get("count")
                out.append(f"[{e.where}] Qdrant returned {raw_hit_count}, "
                           f"kept {kept} after quality/dedup filtering")
                continue
            out.append(self.render_event(e))
        return out

    def render_all(self, events: list) -> str:
        """One pass over a whole run's buffered events -> the full
        narrative file body. See class docstring for the grouping rules,
        and the comment block above _partition for why this is three
        phases rather than one loop.
        """
        self._llm_call_counter = 0
        prelude, graph_blocks, orphans, spans = self._partition(events)
        planned, timeline, tasks_generated = self._plan(spans)

        out: list = [self.render_event(ev) for ev in prelude]
        # D-117: rendered from the WHOLE event list, and placed before the
        # execution plan so it is the first thing after the header.
        out.append(self.render_problems(events))
        # The plan preview reads FIRST, before the detailed graph-
        # construction listing -- and is now computed before it is placed,
        # rather than written as a None placeholder and back-patched.
        out.append(self._render_plan_preview(timeline))
        out.extend(self._render_graph_build(b) for b in graph_blocks)
        out.extend(self.render_event(ev) for ev in orphans)

        total = len(planned)
        for sp in planned:
            if sp["kind"] == "fanout":
                out.append(f"[Step {sp['index'] + 1}/{total}]\n"
                           + self._render_fanout(sp["node"], sp["events"],
                                                 sp["count"]))
            elif sp["kind"] == "decision":
                out.append(self._render_decision_span(sp, total))
            else:
                out.extend(self._render_plain_span(sp, total))
            for title, number, route_fields in sp["trailers"]:
                out.append(self._phase_summary(title, number, route_fields))

        out.append(self._render_timeline(events, timeline, tasks_generated))
        return "\n\n".join(out)

    def _render_plan_preview(self, timeline: list) -> str:
        """A short "table of contents" — the same node sequence the final
        timeline shows, without decision/duration detail — placed right
        after graph construction so a reader can see the shape of the run
        before scrolling through the detailed per-node sections below."""
        lines = [self._BANNER, "EXECUTION PLAN", self._BANNER, "", "START"]
        for label, _decision, _ts in timeline:
            if label.startswith("__LOOP__:"):
                lines.append("")
                lines.append(f"Gather Loop {label.split(':', 1)[1]}")
                continue
            if label.startswith("__EXIT__:"):
                continue  # exit reasons belong in the final summary, not the preview
            lines.append(" |")
            lines.append(" v")
            lines.append(label)
        lines.append(" |")
        lines.append(" v")
        lines.append("END")
        return "\n".join(lines)

    def _render_fanout(self, node: str, gevents: list, count: int) -> str:
        """Serialize a parallel search_worker/mcp_search_worker fan-out
        into one block per task, instead of the flat interleaved stream
        the events actually arrive in (review: "the log should replay
        events... debug logs can stay parallel internally, human logs
        should replay events").

        Correlation, in order of preference:
          1. An event's own "task" field (worker.done/failed,
             chain.answered, chain.tier_failed, chain.attempt all carry
             this — see agents/gathering.py and tools/retrieval_chain.py).
          2. Query-string matching, for the retrieval LAYER itself
             (retrieval/hybrid.py, storage/{qdrant_store,opensearch_store}
             .py), which has no concept of "task" at all — it only ever
             knows a query string. Each task's own primary query seeds the
             match set; chain.attempt's "corpus_reformulated" tier
             additionally reports the REFORMULATED query string BEFORE
             that attempt runs, unconditionally — not only chain.answered,
             which only fires when the attempt is later judged sufficient.
             Without chain.attempt, a reformulated attempt that turned out
             insufficient (fell through to the next tier) would have its
             retrieval.raw/retrieval.hybrid events correlate to nothing —
             the query string was tried, but nothing ever logged it
             anywhere retrievable. This is the exact gap a narrative-log
             review surfaced: those events were landing in "unattributed"
             with no indication why.
          3. Anything matching neither — genuinely rare now, but rendered
             plainly under its own heading rather than silently dropped,
             since this design never hides an event, only reorganizes it.
        """
        tasks = []
        for e in gevents:
            if e.msg == "node.enter":
                task_key = e.fields.get("task") or ""
                goal, _, query = task_key.partition("::")
                tasks.append({"key": task_key, "goal": goal, "query": query})

        query_to_task = {t["query"]: t["key"] for t in tasks}
        for e in gevents:
            if e.msg in ("chain.answered", "chain.attempt") and \
                    e.fields.get("tier") == "corpus_reformulated":
                tk, q = e.fields.get("task"), e.fields.get("query")
                if tk and q:
                    query_to_task[q] = tk

        buckets = {t["key"]: [] for t in tasks}
        unattributed = []
        for e in gevents:
            if e.msg == "node.enter":
                continue
            tk = e.fields.get("task")
            if tk in buckets:
                buckets[tk].append(e)
                continue
            q = e.fields.get("query")
            if q is not None and q in query_to_task:
                buckets[query_to_task[q]].append(e)
                continue
            unattributed.append(e)

        lines = [self._THIN, f"[FAN-OUT] {node} x {count}", self._THIN]
        for i, t in enumerate(tasks, 1):
            result_ev = next((e for e in buckets[t["key"]]
                              if e.msg in ("worker.done", "worker.failed")), None)
            lines.append("")
            lines.append(self._BANNER)
            lines.append(f"SEARCH TASK {i}/{len(tasks)}")
            lines.append(self._BANNER)
            lines.append(f"Goal:  {t['goal']}")
            lines.append(f"Query: {t['query']}")
            for e in buckets[t["key"]]:
                if e is result_ev:
                    continue
                lines.append("")
                lines.append(self.render_event(e))
            lines.append("")
            if result_ev is None:
                lines.append("Result: (no completion event recorded)")
            elif result_ev.msg == "worker.done":
                tier_counts = [(e.fields.get("tier"), e.fields.get("items"))
                              for e in buckets[t["key"]] if e.msg == "chain.answered"]
                breakdown = ", ".join(f"{tier}: {n}" for tier, n in tier_counts)
                lines.append(f"Result: {result_ev.fields.get('items')} item(s) "
                             f"from {result_ev.fields.get('source')}"
                             + (f"  [{breakdown}]" if breakdown else ""))
            else:
                lines.append(f"Result: FAILED "
                             f"({result_ev.fields.get('reason', 'unknown reason')})")
        if unattributed:
            lines.append("")
            lines.append(self._THIN)
            lines.append("Events not attributable to a specific task above")
            lines.append(self._THIN)
            lines.append("(these calls don't carry a task/query field to match against —")
            lines.append(" typically the model-knowledge recall tier; see")
            lines.append(" tools/model_knowledge.py. Not a race condition or a bug in")
            lines.append(" this rendering — the underlying event simply has nothing to")
            lines.append(" correlate it to one task over another.)")
            for e in unattributed:
                lines.append("")
                lines.append(self.render_event(e))
        return "\n".join(lines)

    def _phase_summary(self, title: str, number: int, route_fields: dict) -> str:
        return (f"{self._THIN}\n{title} {number} SUMMARY\n{self._THIN}\n"
                f"Exit reason:\n{route_fields.get('reason')}")

    def _render_timeline(self, events: list, timeline: list, tasks_generated: int) -> str:
        telemetry = next((e for e in events if e.msg == "run.telemetry"), None)
        t0 = events[0].ts if events else 0.0
        duration = (events[-1].ts - t0) if events else 0.0
        lines = [self._BANNER, "REQUEST SUMMARY", self._BANNER, "", "Execution Path", "", "START"]
        for label, decision, ts in timeline:
            elapsed = f"T+{ts - t0:.1f}s"
            if label.startswith("__LOOP__:"):
                lines.append("")
                lines.append(f"Gather Loop {label.split(':', 1)[1]}  ({elapsed})")
                continue
            if label.startswith("__EXIT__:"):
                lines.append("")
                lines.append(f"Exit: {label.split(':', 1)[1]}  ({elapsed})")
                lines.append("")
                continue
            lines.append(" |")
            lines.append(" v")
            lines.append(f"{label}  ({elapsed})")
            if decision:
                lines.append(" |")
                lines.append(" v")
                lines.append(f"Decision: {decision}")
        lines.append(" |")
        lines.append(" v")
        lines.append(f"END  (T+{duration:.1f}s)")
        lines.append("")
        if telemetry is not None:
            tf = telemetry.fields
            lines.append(f"Intent          : {tf.get('intent')}")
            lines.append(f"Goals           : {tf.get('goals')}")
        lines.append(f"Tasks generated : {tasks_generated}")
        lines.append(f"Total duration  : {duration:.1f} s")
        if telemetry is not None:
            tf = telemetry.fields
            lines.append(f"LLM calls       : {tf.get('llm_provider_calls')}")
            retrievals = (tf.get("retrieval_dense_calls", 0) or 0) + \
                (tf.get("retrieval_keyword_calls", 0) or 0)
            lines.append(f"Retrievals      : {retrievals}")
            lines.append(f"Gather laps     : {tf.get('iterations')}")
            lines.append(f"Critique loops  : {tf.get('revision_cycles')}")
            status = "SUCCESS" if tf.get("critique_passed") else "INCOMPLETE"
            if tf.get("escalations"):
                status = "ESCALATED"
            lines.append(f"Final status    : {status}")
        return "\n".join(lines)


class NarrativeBufferHandler(logging.Handler):
    """Buffers structured _Event objects per run_id; nothing is written to
    disk (and nothing is rendered to text) until flush_narrative(run_id)
    is called. Buffering STRUCTURED events rather than pre-rendered text
    is what makes grouping-by-node, loop-boundary detection, and the final
    execution timeline possible — all three need to look across multiple
    events, which a per-line formatter (the earlier design) fundamentally
    cannot do. This is a presentation-layer change only: JsonLineFormatter
    and log_event are completely unaffected.

    run_id is read from the event itself when the call site passed one
    explicitly (a few tests construct a Tracer without ever going through
    cli.py's run_id_var.set(...) — see tests/unit/test_tracing.py), falling
    back to the ambient run_id_var every live call site already sets.
    """

    def __init__(self) -> None:
        super().__init__()
        self._buffers: dict[str, list] = {}

    def emit(self, record: logging.LogRecord) -> None:
        fields = getattr(record, "event_fields", None) or {}
        run_id = fields.get("run_id") or run_id_var.get()
        if not run_id:
            return  # nothing to correlate this line to — drop it, like the
                     # old Tracer's per-instance run_id already implied.
        where = f"{record.name}::{record.funcName}:{record.lineno}"
        self._buffers.setdefault(run_id, []).append(
            _Event(record.created, where, record.getMessage(), fields, record.levelname))

    def pop(self, run_id: str) -> list:
        return self._buffers.pop(run_id, [])


# Module-level singleton: at most one NarrativeBufferHandler is ever
# attached to the root logger, shared across every Tracer instance in the
# process (an API server may have several concurrent runs; they each get
# their own buffer, keyed by run_id, inside this one handler — see
# NarrativeBufferHandler.emit above).
_narrative_handler: "NarrativeBufferHandler | None" = None


def enable_narrative_logging() -> None:
    """Attach the shared NarrativeBufferHandler to the root logger, once.

    CALLED BY   tracing.py::Tracer.__init__ — this is the ONLY thing a
                Tracer does now: turn this handler on and, later, flush
                one run's buffer to a file. It records nothing itself.
    """
    global _narrative_handler
    if _narrative_handler is not None:
        return
    _narrative_handler = NarrativeBufferHandler()
    _narrative_handler.setFormatter(NarrativeFormatter())
    _narrative_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(_narrative_handler)


def flush_narrative(run_id: str, log_dir: str = "logs") -> "str | None":
    """Render and write one run's buffered narrative events to
    logs/run-<run_id>.txt.

    CALLED BY   tracing.py::Tracer.flush() — same contract the old
                tracing.py::Tracer.flush() had: returns the path written,
                or None if this run produced nothing to write (narrative
                logging was never enabled, or genuinely no events were
                buffered for this run_id).
    """
    if _narrative_handler is None:
        return None
    events = _narrative_handler.pop(run_id)
    if not events:
        return None
    formatter = _narrative_handler.formatter
    body = formatter.render_all(events)
    directory = pathlib.Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"run-{run_id}.txt"
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"EXECUTION NARRATIVE  |  run_id={run_id}  |  {stamp}\n\n")
        f.write(body)
    return str(path)



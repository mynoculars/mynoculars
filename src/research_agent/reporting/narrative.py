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
from typing import Any

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
        "llm.quality_reject": "Quality gate rejected the response",
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

    def render_all(self, events: list) -> str:
        """One pass over a whole run's buffered events -> the full
        narrative file body. See class docstring for the grouping rules.
        """
        out: list = []
        context: dict = {}
        self._llm_call_counter = 0
        groups: list = []  # list of dicts: {node, events, count}

        # Pass 1: split into a leading "startup" section (everything
        # before the first node.enter) and node-grouped spans.
        i, n = 0, len(events)
        while i < n and not (events[i].msg == "node.enter" or events[i].msg.startswith("graph.")):
            out.append(self.render_event(events[i]))
            i += 1

        plan_index = len(out)
        out.append(None)  # placeholder — filled in below once `timeline` exists,
                          # but positioned here so the condensed overview reads
                          # FIRST, before the detailed graph-construction listing.

        while i < n and events[i].msg.startswith("graph."):
            graph_start = i
            while i < n and events[i].msg.startswith("graph."):
                i += 1
            out.append(self._render_graph_build(events[graph_start:i]))

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
                        groups.append(current)
                    current = {"node": node, "count": 1, "events": [ev]}
            elif current is not None:
                current["events"].append(ev)
            else:
                # An event before any node.enter fired in this segment
                # (shouldn't normally happen once past startup) — surface
                # it rather than silently drop it.
                out.append(self.render_event(ev))
            i += 1
        if current is not None:
            groups.append(current)

        gather_lap = 1
        tasks_generated_total = 0
        pending_lap_label = True
        critique_attempt = 0
        timeline: list = []  # (label, decision_text_or_None)

        for gi, g in enumerate(groups):
            node, count, gevents = g["node"], g["count"], g["events"]

            # Update running context from whatever this span's events carry
            # — done BEFORE rendering INPUT so a span's own decision isn't
            # mistaken for its input.
            span_input = {k: context[k] for k in self._CONTEXT_KEYS if k in context}
            enter_ev = next((e for e in gevents if e.msg == "node.enter"), None)
            if enter_ev is not None and enter_ev.fields.get("query"):
                span_input = {"query": enter_ev.fields["query"], **span_input}

            decision_ev = next((e for e in gevents if e.msg in self._DECISION_EVENTS), None)
            route_ev = next((e for e in gevents if e.msg == "route.decision"), None)

            # Fan-out group (search_worker x N): correlate every event back
            # to the task that produced it, then render one SEARCH TASK
            # block per task instead of a flat interleaved stream — see
            # _render_fanout's own docstring for exactly how correlation
            # works and its one honest limitation (retrieval-layer events
            # carry no task field at all, only a query string).
            if count > 1:
                label = f"{node} x {count}"
                if pending_lap_label:
                    timeline.append((f"__LOOP__:{gather_lap}", None, gevents[0].ts))
                    pending_lap_label = False
                timeline.append((label, None, gevents[0].ts))
                out.append(f"[Step {gi + 1}/{len(groups)}]\n"
                          + self._render_fanout(node, gevents, count))
            elif decision_ev is not None or route_ev is not None:
                # Architectural decision point (detected from event TYPE,
                # not node name — see _DECISION_EVENTS).
                duration = gevents[-1].ts - gevents[0].ts
                lines = [self._BANNER, f"NODE: {node}  [Step {gi + 1}/{len(groups)}]",
                        self._BANNER,
                        f"Started : {self._fmt_ts(gevents[0].ts)}",
                        f"Finished: {self._fmt_ts(gevents[-1].ts)}",
                        f"Elapsed : {duration:.2f}s"]
                if span_input:
                    lines.append("\nINPUT")
                    lines.append(self._THIN)
                    for k, v in span_input.items():
                        lines.append(f"{k}: {v}")
                # Sub-events (LLM calls, retrievals) render HERE — before
                # the Decision/Next lines below, not after — because the
                # decision is a CONSEQUENCE of what these calls returned,
                # not the other way around. Chronological order, not
                # arrival order in the log stream.
                for e in gevents:
                    if e is decision_ev or e is route_ev or e.msg == "node.enter":
                        continue
                    if e.msg in ("llm.call", "retrieval.raw") or e.msg.startswith("graph."):
                        lines.append("")
                        lines.append(self.render_event(e))
                decision_text = None
                if decision_ev is not None:
                    decision_text = self._decision_text(decision_ev.msg, decision_ev.fields)
                elif route_ev is not None:
                    extra = {k: v for k, v in route_ev.fields.items()
                             if k not in ("from_node", "to_node", "reason",
                                          "escalation_trigger")}
                    if extra:
                        decision_text = ", ".join(f"{k}: {v}" for k, v in extra.items())
                if decision_text:
                    lines.append("\nDECISION")
                    lines.append(self._THIN)
                    lines.append(decision_text)
                to_node = route_ev.fields.get("to_node") if route_ev else None
                if to_node is None and gi + 1 < len(groups):
                    to_node = groups[gi + 1]["node"]
                reason = route_ev.fields.get("reason") if route_ev else None
                if decision_text:
                    # The DECISION line above already explains WHY — repeating
                    # it as a "Reason:" under TRANSITION was pure duplication
                    # (review: "Decision / Transition / Reason" all said the
                    # same thing three times). Just name what's next.
                    lines.append("\nNEXT")
                    lines.append(self._THIN)
                    lines.append(to_node or "(next node)")
                else:
                    # No dedicated DECISION text exists for this span (the
                    # generic route.decision fallback had nothing extra to
                    # show) — the reason is the ONLY explanation available,
                    # so it stays, in full.
                    lines.append("\nTRANSITION")
                    lines.append(self._THIN)
                    lines.append(f"{node}\n      |\n      v\n{to_node or '(next node)'}")
                    lines.append(f"\nReason:\n{reason or '(fixed edge)'}")
                out.append("\n".join(lines))
                timeline.append((node, decision_text, gevents[0].ts))
            else:
                # Ordinary node, no decision of its own (memory_retrieve,
                # merger, memory_writer, telemetry, human_escalation): a
                # plain heading plus whatever it logged, unchanged.
                duration = gevents[-1].ts - gevents[0].ts
                out.append(f"{self._THIN}\nNODE: {node}  [Step {gi + 1}/{len(groups)}]\n"
                          f"Started : {self._fmt_ts(gevents[0].ts)}\n"
                          f"Finished: {self._fmt_ts(gevents[-1].ts)}\n"
                          f"Elapsed : {duration:.2f}s\n{self._THIN}")
                raw_hit_count = None
                for e in gevents:
                    if e.msg == "node.enter":
                        continue
                    if e.msg == "retrieval.raw" and "memory" in str(e.fields.get("source", "")):
                        raw_hit_count = e.fields.get("hit_count")
                        out.append(self.render_event(e))
                        continue
                    if e.msg == "memory.retrieved" and raw_hit_count is not None:
                        # Clarifies a genuinely confusing pair: the retrieval
                        # layer's raw hit count (Qdrant, no filtering) vs
                        # semantic_memory.py's own count (after its dedup/
                        # quality filtering) — same two numbers already
                        # logged two events apart, just stated as one
                        # sentence instead of two easily-misread ones.
                        kept = e.fields.get("count")
                        out.append(f"[{e.where}] Qdrant returned {raw_hit_count}, "
                                  f"kept {kept} after quality/dedup filtering")
                        continue
                    out.append(self.render_event(e))
                timeline.append((node, None, gevents[0].ts))

            # Loop/critique boundary bookkeeping — driven entirely by the
            # SAME route.decision fields _render_route already uses, never
            # a new inference.
            if route_ev is not None:
                frm, to = route_ev.fields.get("from_node"), route_ev.fields.get("to_node")
                if frm == "gap_generator" and to not in ("compiler", "human_escalation"):
                    gather_lap += 1
                    pending_lap_label = True
                if frm == "progress_checker" and to in ("compiler", "human_escalation"):
                    out.append(self._phase_summary("GATHER PHASE", gather_lap,
                                                    route_ev.fields))
                    timeline.append((f"__EXIT__:{route_ev.fields.get('reason')}", None, route_ev.ts))
                elif frm == "gap_generator" and to == "human_escalation":
                    # The OTHER way the gather loop can end: gap_generator's
                    # own dispatch escalates (E2/E3, task supply exhausted)
                    # instead of progress_checker reaching target/depth.
                    out.append(self._phase_summary("GATHER PHASE", gather_lap,
                                                    route_ev.fields))
                    timeline.append((f"__EXIT__:{route_ev.fields.get('reason')}", None, route_ev.ts))
                if frm == "critic" and to == "compiler":
                    critique_attempt += 1
                if frm == "critic" and to in ("memory_writer", "telemetry", "human_escalation"):
                    out.append(self._phase_summary("CRITIQUE PHASE", critique_attempt + 1,
                                                    route_ev.fields))
                    timeline.append((f"__EXIT__:{route_ev.fields.get('reason')}", None, route_ev.ts))

            # Update context AFTER rendering this span, so the NEXT span's
            # INPUT sees this span's own output, not itself.
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
                    tasks_generated_total += e.fields.get("produced") or 0
                elif e.msg == "node.critique":
                    context["revision_count"] = e.fields.get("revision")

        out[plan_index] = self._render_plan_preview(timeline)
        out.append(self._render_timeline(events, timeline, tasks_generated_total))
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



"""
agents/compilation.py — Phase 3/4 nodes: compile, critique, persist, measure.

Purpose:
    Turn gathered evidence into the final report, judge it, write memory,
    and aggregate telemetry.

Responsibilities:
    - compiler_node: compose the Markdown report; on revision passes it
      receives the critic's notes so the rewrite is grounded (D-22), never
      a blind retry.
    - critic_node: judge faithfulness/completeness ONLY (evidence
      sufficiency is the progress checker's job — one judge per question).
      Bounded by max_revisions; exhaustion triggers the E4 escalation STUB
      (a log line in this core build; interrupt() in the full design).
    - memory_writer_node: persist fresh evidence after a PASSED critique
      (D-24 — bad reports never feed long-term memory).
    - telemetry_node: aggregate node-accumulated counters into the final
      telemetry record (D-12: the logger aggregates, it never invents).
"""

import logging
import re
from collections import Counter
from typing import Any, Dict

from research_agent import langfuse as lf
from research_agent.agents.escalation import raise_or_log
from research_agent.config import Settings
from research_agent.guardrails.citations import (clean_citations,
                                                 normalise_citation_form,
                                                 residual_paste_sites)
from research_agent.guardrails.claims import audit_cited_figures
from research_agent.guardrails.dedup import dedupe_evidence
from research_agent.guardrails.grounding import (annotate_ungrounded_report,
                                                  report_carries_grounding_notice)
from research_agent.guardrails.hedging import enforce_hedging
from research_agent.guardrails.retrieval import has_grounded_evidence
from research_agent.guardrails.sources import (append_web_sources,
                                                cited_goal_ids,
                                                count_listed_sources)
from research_agent.llm.client import strip_code_fence
from research_agent.llm.router import FallbackRouter
from research_agent.logging_setup import log_event, run_id_var
from research_agent.memory.semantic_memory import SemanticMemory
from research_agent.prompts import templates
from research_agent.reporting.metrics import count_sections
from research_agent.retrieval.terms import distinctive_terms
from research_agent.state import ResearchState

logger = logging.getLogger(__name__)


def build_compiler_node(router: FallbackRouter, settings: Settings,
                        debug: bool = False):
    """Build the report compiler.

    `settings` is new here (D-85), and the parameter order deliberately
    matches build_critic_node's existing (router, settings, debug) rather
    than appending it last -- the two nodes are siblings and reading them
    side by side should not require noticing that their signatures
    disagree. Same precedent as build_telemetry_node gaining `settings`
    for Guardrail G1 and build_merger_node gaining it for P2-12.

    Used for exactly three values, all by the D-85 provenance notice:
    settings.llm_mode (the stub gate), settings.min_evidence_score and
    settings.grounded_recall_target (the grounding verdict). Nothing else
    about this node's behaviour reads settings.
    """

    def compiler_node(state: ResearchState) -> Dict[str, Any]:
        """Turns gathered evidence into the deliverable. Reached from THREE
        different places in the graph: normally from route_convergence
        (gather loop finished), and as an error/abort SINK from two other
        paths — this function's first job is figuring out which case it's
        in.

        READS   state.abort_reason      — set only by human_escalation
                state.planning_error    — set only by goal_manager_node
                state.raw_query, state.goals, state.evidence,
                state.critique_notes    — non-empty ONLY on a rewrite pass
        CALLS   router.complete() — the ONLY free-text (non-JSON) LLM call
                anywhere in this codebase, and therefore the only call that
                passes through the router's quality gate (see llm/router.py).
                Skipped entirely on the abort/error short-circuits below.
        WRITES  state.final_report = <markdown string>
                state.counters["llm_calls"] += 1   (normal path only)
        NEXT    graph.py routes unconditionally to critic.

        Three distinct shapes for final_report, in priority order:
          1. abort_reason is set    -> "aborted by human reviewer" report,
                                        NO LLM call.
          2. planning_error is set  -> "planning failed" report (D-21 —
                                        diagnosable beats silent), NO LLM
                                        call.
          3. neither                -> normal path: build the evidence +
                                        goals context (this IS the RAG
                                        "context construction" step — every
                                        piece of evidence, memory and
                                        corpus alike, is inlined with no
                                        truncation or re-ranking), inject
                                        critique_notes if this is a rewrite
                                        (D-22 — grounded, not blind), call
                                        the model.

        Note that critic_node (below) runs unconditionally after this
        regardless of which of the three paths fired — see its
        short-circuit for how it avoids judging an error report.
        """
        if debug:
            log_event(logger, "node.enter", node="compiler")
        if state.abort_reason:
            # Human abort (D-23): terminal, explicit, still reaches END.
            report = (f"# Research Report — aborted by human reviewer\n\n"
                      f"**Query:** {state.raw_query}\n\n"
                      f"**Reason:** {state.abort_reason}\n")
            return {"final_report": report}
        if state.planning_error:
            # D-21 path: an explicit, diagnosable error report beats silence.
            report = (f"# Research Report — planning failed\n\n"
                      f"**Query:** {state.raw_query}\n\n"
                      f"**Problem:** {state.planning_error}\n\n"
                      f"No retrieval was attempted. Rephrase the query and retry.")
            return {"final_report": report}
        router.set_node("compiler")
        # FIX-5: collapse byte-identical evidence per goal BEFORE it goes
        # into the prompt. Tiers 1-3 of the D-38 ladder all resolve to the
        # same ingested documents, so one corpus sentence reliably arrives
        # again tagged "mcp", and again on each gather lap. Live count
        # (run p205.211): one sentence appeared 26 times in a single
        # 10,626-token compiler prompt. See guardrails/dedup.py for the
        # full account and for why this deliberately touches only the
        # PROMPT's copy, never state.evidence -- every telemetry figure
        # below still counts what was genuinely retrieved.
        prompt_evidence, dedup_counters = dedupe_evidence(state.evidence)
        report = router.complete(templates.compile_report(
            state.raw_query, state.goals, prompt_evidence, state.critique_notes))
        # A model under fallback can still wrap its answer in a code fence
        # despite compile_report's explicit "write Markdown, not JSON, no
        # fence" instruction -- observed live from Mistral after a
        # quality-reject bounced the call off the primary provider. See
        # llm/client.py::strip_code_fence for why this exists and what it
        # deliberately does NOT attempt to fix.
        report = strip_code_fence(report)
        # D-99: normalise the citation FORM before anything reads
        # citations. This has to come first: clean_citations below, the
        # Sources block, the D-66 gate and the D-91 audit all read
        # citations through sources.py::cited_goal_ids, which matches
        # `[gN]` and nothing else -- so a report written with `(gN)` is
        # not a report with malformed citations to them, it is a report
        # with NO citations, and they all fail silently together. See
        # guardrails/citations.py::normalise_citation_form for the live
        # run that made this necessary.
        report, form_counters = normalise_citation_form(report, state.goals)
        # D-43: deterministic citation repair. The ATTRIBUTION RULE (D-40)
        # asks the model for correct citations; compliance across live runs
        # was roughly two in three. This enforces the half that can be
        # enforced without reading meaning.
        report, citation_counters = clean_citations(
            report, state.goals, state.evidence)
        citation_counters = {**form_counters, **citation_counters}
        # Guardrail G3 enforcement half (P205.135 follow-up): same call
        # site, same shape of check as clean_citations above -- see
        # guardrails/hedging.py for why this exists (the compiler
        # instruction to hedge UNVERIFIED-SPECIFIC claims is not
        # reliably followed on its own).
        report, hedge_counters = enforce_hedging(report, state.evidence)
        # D-57: deterministic attribution for web evidence. LAST, after
        # both passes above -- each of them searches the report for literal
        # spans of evidence content, and a Sources block full of titles and
        # URLs is exactly what could be mistaken for a paste. Appending
        # afterwards puts it out of their reach entirely.
        #
        # Deterministic rather than a prompt instruction, for D-51's reason
        # verbatim: asking the compiler to carry URLs into prose is the same
        # bet that produced hedge_specific_items 29 with zero visible
        # hedging. This also leaves D-40's [gN]-only prose rule fully
        # intact -- the section sits BELOW the report, so nothing above it
        # changes.
        #
        # No-op returning the report byte-identical whenever there is no
        # cited web evidence, which is every run with WEB_SEARCH_ENABLED
        # false (the default).
        report, source_counters = append_web_sources(
            report, state.evidence, state.goals, state.human_guidance)
        # D-85: the provenance notice, LAST of all -- after the three
        # passes above for the same reason append_web_sources runs after
        # the first two (see guardrails/grounding.py's own docstring):
        # clean_citations and enforce_hedging search the report for
        # literal spans of evidence text, and generated text they were
        # never meant to inspect belongs out of their reach. Running after
        # append_web_sources additionally keeps the notice clear of the
        # Sources block that count_listed_sources (D-59) parses back out.
        #
        # Gated OFF in stub mode, exactly like D-66's zero-citation gate
        # and telemetry_node's report.shipped_with_no_citations backstop:
        # StubClient's fixed placeholder report (llm/client.py) exists to
        # prove the graph executes offline, and models nothing at all
        # about where evidence came from. Annotating it would be noise in
        # the one mode that is deliberately not a real answer.
        if settings.llm_mode != "stub":
            report, grounding_counters = annotate_ungrounded_report(
                report, state.goals, state.evidence,
                settings.min_evidence_score, settings.grounded_recall_target)
        else:
            grounding_counters = {}
        # New: compiler previously had no summary event of its own — only
        # the raw "llm.call" line, which says nothing about the REPORT
        # itself. sections/evidence_cited/output_chars are all cheap,
        # already-computable facts about the report this node just
        # produced; logging_setup.py::NarrativeFormatter renders this as
        # this span's DECISION line (see _decision_text).
        # S-10: shared with cli.py's RESULT block via reporting/metrics.py
        # -- previously two different regexes (this one counted level-1
        # headings too) reported two different counts for the same report.
        sections = count_sections(report)
        evidence_cited = len(cited_goal_ids(report))
        log_event(logger, "node.compiled", sections=sections,
                  evidence_cited=evidence_cited, output_chars=len(report),
                  hedge_markers_inserted=int(hedge_counters.get(
                      "hedge_markers_inserted", 0)))
        # P2-07: renamed from "llm_calls" — see telemetry_node's docstring.
        # complete() (not complete_json) is the only free-text path, so this
        # is the one node whose drained counters can include
        # llm_quality_calls (the self-scoring gate only runs on free text).
        counters = {"llm_node_calls": 1, **router.drain_counters(),
                    **citation_counters, **hedge_counters, **source_counters,
                    **dedup_counters, **grounding_counters}
        # D-88: the SAME guardrail numbers, carried a second way -- scoped
        # to THIS compile pass instead of summed across every revision.
        # See ResearchState.last_compile_guardrails for why both views
        # exist and why this one cannot simply be derived from the shipped
        # report the way D-59 derived web_sources_listed (a citation that
        # was REMOVED leaves nothing behind to count).
        #
        # Deliberately excludes the router's own counters: those are
        # genuinely run-cumulative (provider calls, fallback hops, tokens)
        # and have no per-report meaning.
        last_compile = {**citation_counters, **hedge_counters,
                        **source_counters, **dedup_counters,
                        **grounding_counters}
        return {"final_report": report, "counters": counters,
                "last_compile_guardrails": last_compile}

    return compiler_node


def _confirm_unsupported_figures(router: FallbackRouter, flagged: list,
                                 evidence: list) -> list:
    """Ask one judge which flagged figures are GENUINELY unsupported (D-95).

    CALLED BY   critic_node, only when the gate is enabled and
                audit_cited_figures already flagged something.
    RETURNS     the subset of flagged figures the judge confirms, in the
                order they were flagged. Empty list means "nothing
                confirmed" -- including every failure path.

    FAILS OPEN, in three separate ways, all deliberate:
      - nothing flagged        -> no call, empty result
      - the judge raises       -> empty result, counted, run continues
      - the judge names a
        figure that was never
        flagged                -> ignored

    The third matters as much as the second. The deterministic pass owns
    WHAT MAY BE ACCUSED; the judge may only confirm or clear. Letting it
    add findings would hand an LLM the power to fail a report over
    something no mechanical check ever saw -- the exact inversion of this
    codebase's "deterministic where possible, model only where it cannot"
    rule (guardrails/__init__.py).

    A fail-open judge means the gate can only ever be MORE lenient than
    D-91's raw count, never harsher -- so turning it on cannot make a
    previously-passing report fail for a reason nobody can inspect.
    """
    if not flagged:
        return []
    router.set_node("claim_verifier")
    try:
        result = router.complete_json(
            templates.verify_figures(flagged, evidence))
        named = {str(f) for f in (result.get("unsupported") or [])}
    except Exception as exc:  # noqa: BLE001 -- fail open, never crash a run
        log_event(logger, "critic.claim_verification_failed",
                  level=logging.WARNING, reason=type(exc).__name__,
                  error=str(exc)[:300], flagged=len(flagged))
        return []
    return [f["figure"] for f in flagged if str(f["figure"]) in named]


def build_critic_node(router: FallbackRouter, settings: Settings, debug: bool = False):

    """Build the report critic (bounded self-critique loop, D-22)."""

    def critic_node(state: ResearchState) -> Dict[str, Any]:
        """The agent judges its own output. Runs unconditionally right
        after compiler — including after compiler's own error/abort
        short-circuits, which is why the first check here matters.

        READS   state.planning_error, state.abort_reason — if either is
                set there is nothing meaningful to critique.
                Otherwise: state.raw_query, state.final_report, state.goals,
                state.evidence (for the deterministic gate below).
        CALLS   router.complete_json() asking ONLY two things: is the report
                faithful to its OWN stated evidence, and does it address
                every goal. The prompt explicitly forbids judging whether
                MORE research was needed — that question belongs to
                progress_checker, two phases upstream. One judge, one
                question, each — merging these two would make the loop
                unable to tell which remedy (rewrite vs. re-research) to
                apply. SKIPPED (no LLM call) when the deterministic
                zero-citation gate below already fails the report — see
                that block's own comment (D-66).
        WRITES  state.critique_passed = bool
                state.revision_count += 1        (ticks on EVERY pass,
                                                  pass or fail)
                state.counters["llm_calls"] += 1, ["revision_cycles"] += 1
                IF FAILED: state.critique_notes += notes  (accumulates via
                    reducer — these are what gets injected into the NEXT
                    compile prompt, per compiler_node's D-22 grounding)
                    IF revision_count has hit settings.max_revisions:
                        hitl_enabled -> state.escalation_trigger = "E4"
                        else         -> just log a WARNING, ship the
                                        report unreviewed
        NEXT    graph.py's route_after_critique: passed -> memory_writer;
                failed + budget left -> compiler (rewrite loop); failed +
                E4 triggered -> human_escalation; failed + budget spent,
                HITL off -> telemetry directly, SKIPPING memory_writer (a
                report that failed its own quality bar never enters
                long-term memory).

        The short-circuit below (nothing to judge on error/abort paths)
        exists so those two paths still reach telemetry/END with a clean
        signal, rather than being critiqued against evidence that was
        never gathered.
        """
        if debug:
            log_event(logger, "node.enter", node="critic")
        if state.planning_error or state.abort_reason:
            return {"critique_passed": True}
        if (settings.llm_mode != "stub" and state.evidence
                and not cited_goal_ids(state.final_report)):
            # D-66: a report can reach here having cited NOTHING despite
            # evidence being available -- observed live, twice, both times
            # via the same chain: every candidate in FallbackRouter's
            # chain failed the quality bar, chain_exhausted_low_quality
            # (D-60c) served the best-scoring survivor anyway, and that
            # survivor's prose carried zero [gN] markers. The LLM judge
            # has nothing to fail such a report ON -- zero claims means
            # zero unsupported claims, so "faithful" and "cited nothing"
            # were indistinguishable to it, and one live run's judge
            # actually passed a report shaped exactly like this. This
            # deterministic gate closes that -- same posture as
            # D-40/D-43/D-51: enforce what a prompt instruction cannot
            # reliably guarantee, without spending a judging call on an
            # answer already known to fail. Gated on state.evidence being
            # non-empty: a report legitimately citing nothing because
            # NOTHING was ever retrieved is not this failure mode. Also
            # gated OFF in stub mode: StubClient's fixed placeholder report
            # (llm/client.py) never carries [gN] markers by design -- it
            # exists to prove the graph executes offline, not to model
            # citation discipline, and every existing offline test expects
            # a stub run to pass critique.
            passed = False
            notes = [f"Report cites no evidence ([gN] markers) despite "
                     f"{len(state.evidence)} evidence item(s) being "
                     f"available. Every claim must attribute to retrieved "
                     f"evidence."]
            revision = state.revision_count + 1
            update: Dict[str, Any] = {
                "critique_passed": passed,
                "revision_count": revision,
                "critique_notes": notes,
                "counters": {"revision_cycles": 1},
            }
            if revision >= settings.max_revisions:
                update.update(raise_or_log(state, settings, "E4",
                                           reason="report_cites_no_evidence",
                                           revisions=revision, notes=notes))
            log_event(logger, "critic.zero_citation_gate", level=logging.WARNING,
                      revision=revision, evidence_items=len(state.evidence))
            lf.score(run_id_var.get(), "critique_passed", passed,
                    comment=f"revision={revision}")
            return update
        # D-95: the claim-verification gate. Deterministic pass first --
        # cheap, whole-report, no call -- and the model is asked only
        # about what it flagged. Shaped after D-66's zero-citation gate
        # directly above: when the answer is already known to fail, fail
        # here rather than spending the large critique call to rediscover
        # it.
        #
        # Ordered AFTER D-66 deliberately: a report citing nothing has no
        # cited figures to audit, so that gate must get first refusal.
        if settings.claim_verification_enabled and state.evidence:
            flagged, _ = audit_cited_figures(
                state.final_report, state.goals, state.evidence)
            confirmed = _confirm_unsupported_figures(router, flagged,
                                                     state.evidence)
            if confirmed:
                revision = state.revision_count + 1
                notes = [
                    f"Figure {f} appears in no evidence cited by its own "
                    f"sentence, and a verification pass confirmed the "
                    f"evidence does not support it in any form. Remove it, "
                    f"or attribute it to evidence that does."
                    for f in confirmed]
                update = {
                    "critique_passed": False,
                    "revision_count": revision,
                    "critique_notes": notes,
                    "counters": {"revision_cycles": 1,
                                 "claim_figures_confirmed": float(len(confirmed)),
                                 **router.drain_counters()},
                }
                if revision >= settings.max_revisions:
                    update.update(raise_or_log(
                        state, settings, "E4",
                        reason="unsupported_figures_confirmed",
                        revisions=revision, figures=confirmed))
                log_event(logger, "critic.unsupported_figures_gate",
                          level=logging.WARNING, revision=revision,
                          figures=confirmed, flagged=len(flagged))
                lf.score(run_id_var.get(), "critique_passed", False,
                        comment=f"revision={revision}")
                return update
        router.set_node("critic")

        # FIX-5, same pass as compiler_node. D-46 exists because the critic
        # was being asked to verify claims against evidence it could not
        # see; templates.critique caps its evidence block at the last 60
        # items, so 67 occurrences of four sentences could crowd out the
        # evidence a claim actually rests on and reintroduce the exact
        # problem D-46 fixed. Counted only in compiler_node, so the run's
        # evidence_deduplicated figure stays one number about one report
        # rather than double-counting the same duplicates twice per cycle.
        prompt_evidence, _ = dedupe_evidence(state.evidence)
        result = router.complete_json(templates.critique(
            state.raw_query, state.final_report, state.goals,
            prompt_evidence))
        passed = bool(result.get("passed", False))
        notes = [str(n) for n in result.get("notes", [])]
        revision = state.revision_count + 1
        update: Dict[str, Any] = {
            "critique_passed": passed,
            "revision_count": revision,
            "counters": {"llm_node_calls": 1, "revision_cycles": 1,
                        **router.drain_counters()},
        }
        if not passed:
            update["critique_notes"] = notes  # accumulates via reducer
            if revision >= settings.max_revisions:
                # D-23 bound: an E4 redirect routes back to compiler ->
                # critic, and revision_count only ever grows, so this
                # branch is true on every subsequent pass — the same
                # unbounded re-raise E2/E3 has. raise_or_log
                # (agents/escalation.py) folds escalation_allowed with
                # the suppressed/stub logging shared by all four triggers.
                update.update(raise_or_log(state, settings, "E4",
                                           reason="critique_budget_exhausted",
                                           revisions=revision, notes=notes))
        log_event(logger, "node.critique", passed=passed, revision=revision)
        lf.score(run_id_var.get(), "critique_passed", passed,
                comment=f"revision={revision}")
        # `result.get("score")` is the critic's own self-reported 0..1
        # confidence -- part of templates.critique's schema but otherwise
        # unread anywhere in this codebase (see Learning Guide Part 7).
        # Recording it here costs nothing and gives Langfuse a
        # "groundedness"-shaped signal without changing what any node
        # itself does with it.
        if "score" in result:
            lf.score(run_id_var.get(), "critique_self_score", result.get("score"))
        return update

    return critic_node


def build_memory_writer_node(memory: SemanticMemory, settings,
                             debug: bool = False):
    """Build the memory write-back node (runs only after a passed critique)."""

    def memory_writer_node(state: ResearchState) -> Dict[str, Any]:
        """Only reachable via route_after_critique when
        state.critique_passed is True — a report that failed its own
        quality bar never gets here, and never teaches future runs
        anything (D-24).

        READS   state.raw_query, state.evidence (everything gathered this
                run — memory.store_run internally filters out anything
                already tagged source="memory", so recalled facts are never
                re-written and cannot echo themselves across runs).
        CALLS   memory.store_run(query, evidence) — embeds each fresh
                evidence item's content and upserts it into Qdrant's
                long-term memory collection. No LLM call.
        WRITES  state.counters["memory_writes"] += written_count
                (the actual write lands in Qdrant, not in graph state —
                see memory/semantic_memory.py for the point structure)
        NEXT    graph.py routes unconditionally to telemetry.
        """
        if debug:
            log_event(logger, "node.enter", node="memory_writer")
        # D-24 quality gate -- see SemanticMemory.store_run. Evidence that
        # never cleared the memory-write floor must not become permanently
        # promoted cross-run memory. M-3: this is settings.memory_write_min_score,
        # a SEPARATE (lower) floor from the coverage gate's
        # min_evidence_score -- gating on the coverage floor made every
        # single-leg RRF hit ineligible, since it can score at most the
        # single-leg ceiling. See memory_write_min_score's own docstring
        # in config.py.
        written = memory.store_run(state.raw_query, state.evidence,
                                   min_score=settings.memory_write_min_score)
        return {"counters": {"memory_writes": float(written)}}

    return memory_writer_node


def build_telemetry_node(settings: Settings, debug: bool = False):
    """Build the telemetry aggregator — pure aggregation, no invention.

    `settings` is new here (Guardrail G1): the only thing it's used for
    is settings.retrieval_floor_warn_ratio, an observational threshold —
    nothing else in this node's behaviour changes, and every OTHER
    figure below is still counted straight from state.counters exactly
    as D-12 already requires.
    """

    def telemetry_node(state: ResearchState) -> Dict[str, Any]:
        """The mandatory final node — every single path through this graph
        reaches here, whether it went through memory_writer, an abort, a
        planning error, or an exhausted critique budget.

        READS   state.classification, state.goals, state.iteration_depth,
                state.evidence, state.recall_score, state.counters (the
                additive tallies every earlier node has been contributing
                to all run), state.critique_passed, state.planning_error,
                state.escalation_history.
        CALLS   nothing external — pure aggregation.
        WRITES  state.telemetry = {intent, goals, iterations,
                    evidence_items, evidence_by_source,
                    goals_without_evidence, grounding_ratio, recall,
                    grounded_score, hedge_specific_items,
                    retrieval_dense_candidates, retrieval_dropped_by_floor,
                    retrieval_floor_drop_ratio,
                    llm_node_calls, llm_provider_calls, llm_fallback_hops,
                    llm_quality_calls, llm_quality_calls_failed,
                    llm_quality_failure_ratio,
                    retrieval_dense_calls, retrieval_keyword_calls,
                    retrieval_leg_unavailable, producer_rejects,
                    search_calls, search_failures, memory_hits,
                    memory_writes, revision_cycles, critique_passed,
                    planning_error}
        NEXT    graph.py routes unconditionally to END. cli.py then prints
                state.final_report followed by this telemetry dict, and
                persists a summary row to Postgres (CLI only — the API
                path never calls record_run).

        D-12: this function ONLY adds up numbers other nodes already
        recorded in state.counters — it never invents a figure.
        P2-07: "llm_calls" is renamed "llm_node_calls" for honesty — it
        counts NODE executions that made an LLM call, not provider
        requests (a node that fell through two fallback hops still counts
        as one node-level call). The new "llm_provider_calls" and
        "llm_fallback_hops" (from llm/router.py's boundary, drained into
        every LLM-calling node's own counters — see planning.py/
        gathering.py/compilation.py's individual nodes) give the actual
        provider-request volume this dict couldn't previously show.
        "llm_quality_calls" counts judge-scoring calls (compiler_node's
        free-text path only — P2-11: the judge is always the NEXT provider
        in the fallback chain, never the answering one). "llm_quality_calls_failed"
        (P2-11 follow-up) is the subset of those where the judge itself
        couldn't be reached/parsed — fail-open kept the answer either way,
        but a nonzero count here means that many gate checks had no real
        opinion behind them, which is worth knowing even though the run
        still succeeded. "retrieval_dense_calls"/"retrieval_keyword_calls"
        (P2-07 follow-up) count actual HybridRetriever.search() attempts per
        leg — one pair per search_worker invocation that used the real
        corpus tool; a fake tool in tests contributes none, since it isn't
        really doing retrieval. "retrieval_leg_unavailable" counts DEGRADED
        legs specifically (a store that was unreachable when checked), not
        legs that legitimately returned zero hits — see
        retrieval/hybrid.py::HybridRetriever._bump_retrieval_counts for
        that distinction. "producer_rejects" (P2-06) counts malformed
        goal/task items the LLM returned that were dropped rather than
        crashing the run. Read a debug trace (--debug) for full per-call
        detail (exact prompts/latencies) beyond what these aggregate counts
        show.

        "evidence_by_source" (P2-13 follow-up): a plain {source: count}
        breakdown -- {"corpus": 6} when the corpus tool is active,
        {"mcp": 6} when tools/mcp_client.py's tool is active instead
        (settings.mcp_enabled), {"memory": 5, "corpus": 6} when memory
        recall contributed too, etc. Computed FRESH from state.evidence
        here, not threaded through as an incrementally-bumped counter the
        way llm_*/retrieval_*/search_* above are -- state.evidence is
        already the complete, final list by the time this node runs, so
        there's nothing to accumulate; recounting it directly here is
        simpler and cannot drift out of sync with the actual evidence
        list the way a separately-bumped counter theoretically could.
        This exists because there was previously NO deterministic way to
        confirm evidence came from MCP versus the corpus tool -- only an
        indirect, LLM-dependent hint (whether the compiled report's own
        citations happened to preserve the "[goal | source | score]"
        tag templates.py's compile_report prompt includes per item).
        """
        if debug:
            log_event(logger, "node.enter", node="telemetry")
        c = state.counters
        # Counter(...) tallies how many Evidence items each distinct
        # `source` value appears with -- e.g. Counter(["corpus", "corpus",
        # "mcp"]) -> {"corpus": 2, "mcp": 1}. dict(...) converts that
        # Counter into a plain dict so it serializes the same simple way
        # every other telemetry field already does (json.dumps has no
        # trouble with a plain dict; a Counter object is unnecessary here
        # once the counting itself is done).
        evidence_by_source = dict(Counter(e.source for e in state.evidence))
        # GROUNDING AUDIT -- which goals reached the compiler with NO
        # evidence behind them at all.
        #
        # WHY THIS IS MEASURED HERE AND NOT PARSED OUT OF THE REPORT: the
        # obvious check is "extract every [gN] citation from final_report
        # and verify that goal has evidence". That does not survive contact
        # with real output -- across four live runs the model cited as
        # `[g1 | corpus | score=0.50]`, `[g1, g4]`, `(g1)` in headings, and
        # in one run used no bracket citations whatsoever. A regex would
        # have found zero citations in exactly the run whose report was
        # LEAST grounded (it named Cassandra's compression codecs from a
        # corpus containing no Cassandra documents), and zero citations is
        # not evidence of grounding -- it is evidence the citation
        # instruction was ignored.
        #
        # state.evidence is structured data with goal_id on every item, so
        # counting it needs no parsing, no LLM, and no format stability. A
        # goal with zero items that still gets discussed in the report is a
        # provable gap-fill from the model's parametric knowledge; this
        # does not prove it HAPPENED, but it is the precondition, and it is
        # the number that tells you how often the opportunity arises.
        #
        # D-12 holds: every figure below is counted from state, not judged.
        goal_ids = [g.goal_id for g in state.goals]
        # D-59: derived from the shipped report, not from the additive
        # counter dict -- see the web_sources_listed entry below.
        web_sources_listed = count_listed_sources(state.final_report)
        web_sourced_items = sum(1 for e in state.evidence
                                if e.source == "web")
        # corpus_recall must apply the SAME topical gate the retrieval
        # ladder uses (D-39), not just the score floor. Live (runs
        # p205.99/.100-check) this reported corpus_recall 1.0 against a
        # ten-document Redis corpus for queries about armies and about
        # India vs the US: off-topic hits still cleared score > 0.5 via
        # cross-leg agreement, so the one metric added specifically as the
        # honesty counterpart to recall was fooled exactly the way the
        # chain's sufficiency test used to be. A goal counts here only if a
        # DOCUMENT both scored above the floor and shares vocabulary with
        # that goal's own description -- has_grounded_evidence (M-1) is
        # the single implementation of that predicate, shared with
        # progress_checker_node and gap_generator_node in gathering.py, so
        # this floor and theirs can never drift apart again the way a
        # second, hardcoded `> 0.5` copy previously let them.
        _goal_terms = {g.goal_id: distinctive_terms(g.description)
                       for g in state.goals}
        _doc_covered = {
            g.goal_id for g in state.goals
            if has_grounded_evidence(g.goal_id, _goal_terms[g.goal_id],
                                     state.evidence, settings.min_evidence_score)}
        corpus_recall = (round(len([g for g in goal_ids if g in _doc_covered])
                               / len(goal_ids), 3) if goal_ids else 0.0)
        evidenced = {e.goal_id for e in state.evidence}
        goals_without_evidence = [g for g in goal_ids if g not in evidenced]
        # Ratio, not just the list, so it is trendable across runs and
        # comparable across queries with different goal counts. Guarded
        # against a run that produced no goals at all (a planning failure),
        # where 0/0 is undefined rather than perfect.
        grounding_ratio = (
            round((len(goal_ids) - len(goals_without_evidence)) / len(goal_ids), 3)
            if goal_ids else 0.0
        )
        # Guardrail G1: aggregate, run-level view of retrieval/hybrid.py's
        # per-query "retrieval.below_floor" log lines. A single dropped
        # query is normal (that's the floor doing its job); a run where
        # NEARLY EVERY dense candidate is dropped means min_similarity is
        # starving the dense leg outright — live evidence (run
        # p205.131-check) showed floor=0.55 dropping literally 100% of
        # dense candidates all run, invisible outside raw debug logs.
        # candidates=0 (no dense leg ever ran, or min_similarity=0.0 so
        # this counter was never bumped) means "nothing to report", not
        # "starved" — ratio stays 0.0 rather than a misleading 0/0.
        # D-66 (backstop): the deterministic critic gate above is meant to
        # stop a zero-citation report before it ships, but a run with HITL
        # disabled and its revision budget exhausted still ships whatever
        # the gate last rejected (raise_or_log's stub path -- "log loudly
        # and ship the report unreviewed" is this codebase's existing
        # posture for an exhausted budget, unchanged here). This is the
        # last-line-of-sight check for that case: same shape as G1/G4/G7
        # above, purely observational, WARNs rather than blocks. Gated
        # off in stub mode for the same reason the critic gate is (see
        # that block's comment) -- StubClient's fixed placeholder report
        # never carries [gN] markers by design.
        if (settings.llm_mode != "stub" and state.evidence
                and not cited_goal_ids(state.final_report)):
            log_event(logger, "report.shipped_with_no_citations",
                      level=logging.WARNING,
                      evidence_items=len(state.evidence))
        # D-85, same last-line-of-sight shape as the check just above:
        # did a run whose corpus contributed too little actually ship
        # saying so? Read from the SHIPPED report (D-59's rule), never
        # from a counter -- compiler_node runs once per revision and its
        # counters merge additively, so a counter here would describe the
        # compile attempts rather than the artifact the reader received.
        grounding_notice_shipped = report_carries_grounding_notice(
            state.final_report)
        # D-91: the cited-figure audit -- the third last-line-of-sight
        # check on the SHIPPED report, alongside the two above. For every
        # sentence that states a figure AND cites a goal, does any
        # evidence under that goal actually contain the figure?
        #
        # Run HERE rather than in compiler_node for D-59's reason: this is
        # a property of the artifact, and compiler_node runs once per
        # revision with additively-merged counters, so a count taken there
        # would describe the compile attempts (exactly the D-88 problem).
        # Taken against state.final_report it is report-scoped by
        # construction.
        #
        # Stub-gated like its two neighbours: StubClient's fixed
        # placeholder report carries no [gN] markers by design, so there
        # is nothing here for this to audit.
        figure_findings: list = []
        figure_counters: Dict[str, float] = {}
        if settings.llm_mode != "stub":
            figure_findings, figure_counters = audit_cited_figures(
                state.final_report, state.goals, state.evidence)
        if figure_findings:
            log_event(logger, "report.unsupported_cited_figures",
                      level=logging.WARNING,
                      unsupported=len(figure_findings),
                      checked=int(figure_counters.get(
                          "cited_figures_checked", 0)),
                      # Capped: a report can state many figures, and one
                      # log line should stay readable. The full count is
                      # the field above; telemetry carries the same
                      # capped sample for the run record.
                      examples=figure_findings[:5])
        # D-99: what the paste guard LEFT, not what it removed. A run
        # reporting 21 removals and a run reporting 21 removals with four
        # pastes still standing are indistinguishable in telemetry
        # otherwise -- and live (run p205.253-check) the second one is
        # what shipped.
        residual_pastes = 0
        if settings.llm_mode != "stub":
            residual_pastes = residual_paste_sites(state.final_report,
                                                   state.evidence)
        if residual_pastes:
            log_event(logger, "report.residual_pasted_evidence",
                      level=logging.WARNING, sites=residual_pastes,
                      removed=int(state.last_compile_guardrails.get(
                          "citations_pasted_evidence_removed", 0)))
        if (settings.llm_mode != "stub" and state.evidence and state.goals
                and corpus_recall < settings.grounded_recall_target):
            log_event(logger, "report.shipped_ungrounded",
                      level=logging.WARNING,
                      corpus_recall=corpus_recall,
                      grounded_recall_target=settings.grounded_recall_target,
                      notice_shipped=grounding_notice_shipped,
                      web_sourced_items=web_sourced_items,
                      model_sourced_items=int(
                          evidence_by_source.get("model", 0)))
        retrieval_dense_candidates = int(c.get("retrieval_dense_candidates", 0))
        retrieval_dropped_by_floor = int(c.get("retrieval_dropped_by_floor", 0))
        retrieval_floor_drop_ratio = (
            round(retrieval_dropped_by_floor / retrieval_dense_candidates, 3)
            if retrieval_dense_candidates else 0.0
        )
        if (retrieval_dense_candidates > 0
                and retrieval_floor_drop_ratio >= settings.retrieval_floor_warn_ratio):
            log_event(logger, "retrieval.floor_starvation", level=logging.WARNING,
                      dropped=retrieval_dropped_by_floor,
                      candidates=retrieval_dense_candidates,
                      ratio=retrieval_floor_drop_ratio,
                      floor=settings.min_similarity,
                      warn_ratio=settings.retrieval_floor_warn_ratio)
        # Guardrail G4 (P205 Phase 2): same shape as G1's check just
        # above -- a run-level ratio, WARNed past a threshold, purely
        # observational. evaluation/quality.py::score_answer is
        # correctly designed to fail OPEN (a broken judge must never
        # reject a perfectly good answer) -- but every live run in this
        # session has shown llm_quality_calls_failed == llm_quality_calls
        # (2/2, every single time): the judge has never once actually
        # scored anything, and nothing before this distinguished that
        # from "scored everything 1.0 on merit". A 100% failure rate
        # means the quality gate has been silently inert this whole
        # session, not merely lenient.
        llm_quality_calls = int(c.get("llm_quality_calls", 0))
        llm_quality_calls_failed = int(c.get("llm_quality_calls_failed", 0))
        llm_quality_failure_ratio = (
            round(llm_quality_calls_failed / llm_quality_calls, 3)
            if llm_quality_calls else 0.0
        )
        if (llm_quality_calls > 0
                and llm_quality_failure_ratio >= settings.quality_judge_warn_ratio):
            log_event(logger, "quality.judge_unreliable", level=logging.WARNING,
                      failed=llm_quality_calls_failed, attempted=llm_quality_calls,
                      ratio=llm_quality_failure_ratio,
                      warn_ratio=settings.quality_judge_warn_ratio)
        # Guardrail G7 (P205 Phase 3, observability half only): same
        # shape again -- a run-level count, WARNed past a threshold,
        # purely observational, checked here (once, at the very end of
        # a completed run) rather than mid-run, since this is visibility
        # into total cost after the fact, not a gate on continuing.
        # Deliberately NOT a circuit breaker -- see
        # settings.run_call_budget_warn's own comment for why a hard
        # mid-run enforcement isn't justified by anything observed yet.
        llm_provider_calls = int(c.get("llm_provider_calls", 0))
        if llm_provider_calls >= settings.run_call_budget_warn:
            log_event(logger, "run.call_budget_high", level=logging.WARNING,
                      llm_provider_calls=llm_provider_calls,
                      warn_threshold=settings.run_call_budget_warn,
                      revision_cycles=int(c.get("revision_cycles", 0)),
                      escalations=len(state.escalation_history))
        telemetry = {
            "intent": state.classification.get("intent"),
            "goals": len(state.goals),
            "iterations": state.iteration_depth,
            "evidence_items": len(state.evidence),
            "evidence_by_source": evidence_by_source,
            # Deliberately distinct from `recall`. recall asks "did enough
            # evidence clear the coverage threshold" and is computed from
            # scores; grounding_ratio asks the cruder, prior question "did
            # this goal get ANY evidence at all", and cannot be affected by
            # threshold tuning. A run can show recall 1.0 and
            # grounding_ratio 0.5 -- that combination means the coverage
            # rule is passing goals the retriever never fed.
            "goals_without_evidence": goals_without_evidence,
            "grounding_ratio": grounding_ratio,
            "recall": round(state.recall_score, 3),
            # D-38 honesty counterpart to recall. recall now includes goals
            # served by the model tier; corpus_recall counts ONLY goals a
            # real DOCUMENT covered. A large gap between them means the
            # answer came from recollection, not from the corpus -- which
            # is legitimate and attributed in the report, but must never be
            # invisible in telemetry.
            "corpus_recall": corpus_recall,
            # Guardrail G2. Written by progress_checker_node every gather
            # cycle (agents/gathering.py); this is just its last value,
            # same pattern as "recall" above.
            "grounded_score": round(state.grounded_score, 3),
            "model_sourced_items": int(evidence_by_source.get("model", 0)),
            # D-57. Deliberately reported ALONGSIDE corpus_recall and
            # grounded_score rather than folded into either: web evidence
            # covers goals but never grounds them, so a run reading
            # recall 1.0 / grounded_score 0.0 / web_sourced_items 12 is
            # telling you precisely and honestly where its answer came
            # from. Without this number that same run is indistinguishable
            # from one that found nothing at all.
            "web_sourced_items": web_sourced_items,
            # How many DISTINCT domains that evidence came from. A useful
            # honesty check on its own: twelve items from one domain is one
            # source repeated, not twelve agreeing -- which is what
            # websearch/filtering.py::cap_by_domain limits at retrieval
            # time and what this makes visible after the fact.
            "web_source_domains": len({e.domain for e in state.evidence
                                       if e.source == "web" and e.domain}),
            # How many web pages the SHIPPED report attributed, and how many
            # were retrieved but never made it there -- either because the
            # compiler cited none of their goals, or because they were not
            # topically about the goal they were filed under (D-59). A high
            # suppressed count against a low listed count is the signature
            # of the web tier doing work that never reached the report.
            #
            # D-59: counted from state.final_report, NOT from c[...]. These
            # two were previously read out of the additive counter dict,
            # which compiler_node contributes to once per REVISION -- so a
            # run with two rewrites reported the sum of three compile
            # attempts as if it described one report. Live (p205.203-check):
            # 44 listed / 25 suppressed against 35 web items, for a report
            # containing 34 entries.
            "web_sources_listed": web_sources_listed,
            "web_sources_suppressed": max(
                0, web_sourced_items - web_sources_listed),
            # D-85. Read together with corpus_recall: `corpus_recall 0.0,
            # grounding_notice_shipped true` is a run that answered from
            # the web or from recollection AND told its reader so in the
            # report itself. The same pair reading `false` means the
            # deliverable claimed nothing about its own provenance --
            # which is the state every run was in before D-85, and which
            # report.shipped_ungrounded now WARNs about.
            "grounding_notice_shipped": grounding_notice_shipped,
            # D-91. Read as a pair: `checked` is how many cited figures
            # the shipped report stated at all, `unsupported` how many of
            # those appear in no evidence under the goal the sentence
            # cites. `0 / 0` means the report stated no cited figures --
            # not that it passed. A nonzero `unsupported` is the first
            # claim-level (rather than report-level or evidence-set-level)
            # honesty signal this harness has had.
            "citations_residual_paste_sites": residual_pastes,
            "cited_figures_checked": int(
                figure_counters.get("cited_figures_checked", 0)),
            "cited_figures_unsupported": int(
                figure_counters.get("cited_figures_unsupported", 0)),
            # A capped sample, so the run record shows WHICH figures
            # rather than only how many -- the difference between a
            # number you can act on and one you have to reproduce.
            "unsupported_figures": [
                {"figure": f["figure"], "goals": f["goals"]}
                for f in figure_findings[:5]
            ],
            # Guardrail G3: model-tier items whose own text paired a
            # specific year with a specific quantity — flagged, not
            # dropped (see tools/model_knowledge.py::_looks_overspecific).
            "hedge_specific_items": sum(
                1 for e in state.evidence if e.hedge_specific),
            # Guardrail G1.
            "retrieval_dense_candidates": retrieval_dense_candidates,
            "retrieval_dropped_by_floor": retrieval_dropped_by_floor,
            "retrieval_floor_drop_ratio": retrieval_floor_drop_ratio,
            "llm_node_calls": int(c.get("llm_node_calls", 0)),
            "llm_provider_calls": int(c.get("llm_provider_calls", 0)),
            "llm_fallback_hops": int(c.get("llm_fallback_hops", 0)),
            "llm_quality_calls": int(c.get("llm_quality_calls", 0)),
            "llm_quality_calls_failed": int(c.get("llm_quality_calls_failed", 0)),
            # Guardrail G4.
            "llm_quality_failure_ratio": llm_quality_failure_ratio,
            "retrieval_dense_calls": int(c.get("retrieval_dense_calls", 0)),
            "retrieval_keyword_calls": int(c.get("retrieval_keyword_calls", 0)),
            "retrieval_leg_unavailable": int(c.get("retrieval_leg_unavailable", 0)),
            "producer_rejects": int(c.get("producer_rejects", 0)),
            # D-86: what the run actually COST, as opposed to how many
            # requests it made. llm_provider_calls cannot distinguish
            # three cheap classify calls from three 7,000-token compile
            # calls; these can. Additive across the run is correct here --
            # unlike the compile-scoped guardrail counts below, every
            # token genuinely was spent.
            "llm_prompt_tokens": int(c.get("llm_prompt_tokens", 0)),
            "llm_completion_tokens": int(c.get("llm_completion_tokens", 0)),
            "llm_total_tokens": int(c.get("llm_prompt_tokens", 0)
                                    + c.get("llm_completion_tokens", 0)),
            # D-93: hops skipped because the prompt could not fit the
            # provider's configured context window. 0 unless
            # LLM_PRIMARY_CONTEXT_TOKENS is set. A nonzero value is the
            # count of guaranteed-failed provider calls this run did NOT
            # make -- read against llm_provider_calls, which no longer
            # includes them.
            "llm_context_skips": int(c.get("llm_context_skips", 0)),
            # D-87: which tier of the D-38 ladder actually answered, and
            # how often a tier failed outright. Previously readable only
            # by grepping `chain.answered` out of a debug trace. Read
            # against corpus_recall: `{"corpus": 6}` with corpus_recall
            # 1.0 is a healthy corpus run; `{"web": 6}` with corpus_recall
            # 0.0 is the p205.246-check shape, and now says so in one
            # field rather than three inferred ones.
            "tier_answers": {
                key[len("chain_answered_"):]: int(value)
                for key, value in sorted(c.items())
                if key.startswith("chain_answered_") and value
            },
            "chain_tier_failures": int(c.get("chain_tier_failed", 0)),
            "chain_exhausted": int(c.get("chain_exhausted", 0)),
            # D-88: guardrail work on the SHIPPED report specifically --
            # citation repairs, hedge markers, dedup -- as opposed to the
            # sum across every compile attempt, which is what reading
            # these out of `counters` would give. See
            # ResearchState.last_compile_guardrails.
            "last_compile_guardrails": {
                key: int(value)
                for key, value in sorted(state.last_compile_guardrails.items())
            },
            "search_calls": int(c.get("search_calls", 0)),
            "search_failures": int(c.get("search_failures", 0)),
            "memory_hits": int(c.get("memory_hits", 0)),
            "memory_writes": int(c.get("memory_writes", 0)),
            "revision_cycles": int(c.get("revision_cycles", 0)),
            "critique_passed": state.critique_passed,
            "planning_error": state.planning_error,
            # escalation_history was written by human_escalation on every
            # resume and then read by NOTHING — the audit trail for D-23
            # existed in state and never left it. Surfacing the trigger/
            # action pairs here puts it in the same place every other
            # per-run fact already lands (telemetry, the agent_runs row,
            # the run.telemetry log line).
            "escalations": [
                {"trigger": h.get("trigger"), "action": h.get("action")}
                for h in state.escalation_history
            ],
        }
        log_event(logger, "run.telemetry", **telemetry)
        return {"telemetry": telemetry}

    return telemetry_node

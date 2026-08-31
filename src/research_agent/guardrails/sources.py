"""
guardrails/sources.py -- the deterministic attribution pass for web evidence
(Phase 4 / D-57).

Purpose:
    Append a "## Sources" section listing the web pages behind the report,
    built from the evidence the compiler ACTUALLY CITED, without touching a
    single character of the prose above it.

Why deterministic rather than a prompt instruction:
    The obvious alternative is to teach the compile prompt to write
    "[g3] (arxiv.org)" inline. This codebase has already run that
    experiment and lost. D-51 exists precisely because a prompt instruction
    to hedge UNVERIFIED-SPECIFIC claims was followed unreliably enough that
    one run reached hedge_specific_items: 29 with ZERO visible hedging in
    the shipped report -- so guardrails/hedging.py was written to enforce
    afterwards what the prompt had asked for beforehand. Attribution is the
    same shape of problem and gets the same shape of answer.

    The second reason is D-40. Its attribution rule is that report prose
    carries [gN] markers and nothing else -- no URLs, no bare titles. A
    Sources section appended BELOW the report does not relax that rule at
    all: the prose stays exactly as clean_citations and enforce_hedging left
    it, and the reader still gets somewhere to click. Relaxing D-40 would
    also have meant teaching guardrails/citations.py, guardrails/hedging.py
    and templates.critique about a new inline form that none of them
    currently recognizes -- three more places to get it wrong, for no gain.

CALLED BY   agents/compilation.py::compiler_node, LAST -- after
            clean_citations and enforce_hedging. Order matters: both of
            those search the report for literal spans of evidence content,
            and a Sources block containing titles and URLs is exactly the
            kind of text that could be mistaken for a paste. Appending after
            they have run puts it out of their reach entirely.

Scope, deliberately narrow:
    ONLY source="web" evidence, and ONLY for goals the report actually
    cites. Corpus and MCP evidence is not listed -- those are documents
    already in the operator's own ingested corpus, addressable by their own
    identifiers, not by a URL. Model-tier recollection has no source to
    cite by definition, which is the whole reason D-49/D-51 hedge it
    instead.

    D-144 adds ONE exception, and labels it in the output: with
    list_when_uncited=True and a report that cites nothing at all, the
    block is emitted anyway under a note saying these are what the run
    retrieved rather than support for a claim. See append_web_sources.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

from research_agent.logging_setup import log_event
from research_agent.state import Evidence, Goal
from research_agent.retrieval.terms import distinctive_terms

logger = logging.getLogger(__name__)

# Matches the [gN] markers D-40 requires in report prose -- the same pattern
# agents/compilation.py::compiler_node already uses to count evidence_cited.
# Shared shape rather than a second, subtly different definition of "what a
# citation looks like", for the same reason D-47 reuses distinctive_terms
# instead of writing a second notion of "on topic".
_CITATION_RE = re.compile(r"\[g(\d+)\]")

SOURCES_HEADING = "## Sources"

# One numbered entry in the block this module emits: "1. [g3] Title (dom) — url".
_SOURCE_ENTRY_RE = re.compile(r"(?m)^\d+\. \[g\d+\] ")


def count_listed_sources(report: str) -> int:
    """How many sources the FINAL report actually lists (D-59).

    Exists because compiler_node's counters are merged additively
    (state.py::merge_counters) and compiler_node runs once per revision --
    so `web_sources_listed` accumulated across every compile ATTEMPT while
    telemetry documented it as a property of the report. Live (run
    p205.203-check, two revision cycles): telemetry reported 44 listed and
    25 suppressed against 35 web items total, while the shipped report
    contained 34 entries. Every one of those numbers was arithmetically
    correct and none of them described the artifact the reader received.

    Counting the shipped text instead is pure, deterministic, and keeps
    D-12 intact -- telemetry still only counts what happened, it just
    counts the right thing. Returns 0 when no Sources section exists,
    which is every run with WEB_SEARCH_ENABLED false.
    """
    head, marker, block = report.rpartition(f"\n{SOURCES_HEADING}\n")
    if not marker:
        return 0
    return len(_SOURCE_ENTRY_RE.findall(block))


def cited_goal_ids(report: str) -> set:
    """Which goals the prose actually cites, as goal_id strings ("g3").

    D-66: made public (was `_cited_goal_ids`) so agents/compilation.py's
    compiler_node (the evidence_cited count) and critic_node (the
    zero-citation gate) share this exact definition of "cited" with the
    Sources attribution logic below, rather than each maintaining its own
    `\\[g(\\d+)\\]` regex that could quietly drift apart from this one.
    """
    return {f"g{m.group(1)}" for m in _CITATION_RE.finditer(report)}


# The line that appears under the heading when the block is listed for an
# UNCITED report (D-144). It is the whole reason the fallback is honest:
# the entries below it are what the run drew on, not support for a named
# claim, and the reader is told which of the two they are looking at.
UNCITED_NOTE = ("_Retrieved for this report. Its prose carries no `[gN]` "
                "citations, so these are listed as the web results this run "
                "drew on -- not as support for any specific claim._")


def append_web_sources(report: str,
                       evidence: List[Evidence],
                       goals: Optional[List[Goal]] = None,
                       guidance: str = "",
                       list_when_uncited: bool = False,
                       ) -> Tuple[str, Dict[str, float]]:
    """Append a Sources section for cited web evidence. Returns (report, counters).

    RETURNS the report UNCHANGED, and zeroed counters, when there is nothing
    to list -- no web evidence, or none of it attached to a goal the report
    cited. That no-op path is the common one today (WEB_SEARCH_ENABLED
    defaults false), so it must be exactly byte-identical, not merely
    similar: a trailing newline difference would be a visible diff in every
    existing report.

    WHY FILTER BY CITED GOAL rather than listing every web result retrieved:
    a Sources list is a claim about what backed THIS report. Listing pages
    the compiler never drew on would be padding at best and misattribution
    at worst -- the reader reasonably assumes a listed source supported
    something they just read. The retrieval-side count lives in telemetry
    (web_sourced_items), which is where "what did we fetch" belongs.

    DEDUPLICATED BY URL, keeping the highest-scoring occurrence. The same
    page can legitimately be returned under two different goals, and listing
    it twice would imply two independent sources.

    ORDERED BY SCORE, best first -- which, because
    websearch/scoring.py::rank_to_score maps rank onto a band monotonically,
    means the engine's own ranking survives all the way to the reader.
    """
    counters = {"web_sources_listed": 0.0, "web_sources_suppressed": 0.0}

    web_items = [e for e in evidence
                 if e.source == "web" and e.url and e.goal_id]
    if not web_items:
        return report, counters

    # D-144: cited-goal membership is the RULE, not the only rule.
    #
    # This filter is correct and stays correct -- a Sources list is a claim
    # about what backed this report, and listing pages the compiler never
    # drew on is misattribution. But it is a strict subset of what the
    # prose cited, so when the prose cites NOTHING it returns nothing, and
    # one LLM formatting failure takes out attribution twice over. Live
    # (p205.280-check): 58 web items across 33 distinct domains retrieved,
    # 0 listed, and the shipped report carried D-85's provenance notice
    # telling the reader to trust figures "unless a listed source confirms
    # them" -- above a page with no listed sources.
    #
    # guardrails/attribution.py runs first and usually fixes the cause. The
    # fallback below is for when it cannot: rather than silently dropping
    # 33 domains, list them under a heading that says exactly what they
    # are. That does not weaken D-59, whose rule is "do not assert support
    # that does not exist" -- the note asserts nothing of the kind. The
    # topical gate below still applies in full, so the nine Redis URLs that
    # motivated D-59 are still dropped on this path too.
    cited = cited_goal_ids(report)
    uncited_fallback = list_when_uncited and not cited
    if uncited_fallback:
        listed = list(web_items)
    else:
        listed = [e for e in web_items if e.goal_id in cited]
    # D-59: cited-goal membership alone is NOT a claim of support. A
    # drifted gather cycle can retrieve web results about an entirely
    # different subject and tag them with a real, correctly-formed goal id;
    # if the report then cites that goal for its OWN reasons, every one of
    # those pages gets listed as a source. Live (run p205.203-check): nine
    # Redis-monitoring URLs appeared under [g1] in an India-vs-US report.
    # The prose was clean -- clean_citations and the compiler's citation
    # discipline both held -- but the Sources section asserted support that
    # did not exist, which is precisely the failure D-51 exists to prevent.
    #
    # Same topical gate as D-39/D-47, reused rather than reinvented so
    # "on topic" cannot come to mean three different things in three files.
    # Skipped entirely when goals are not supplied (the pre-D-59 signature)
    # or when a goal's description yields no distinctive terms -- an
    # untestable claim is left alone rather than resolved against the item.
    if goals:
        goal_terms = {g.goal_id: distinctive_terms(g.description)
                      for g in goals}
        if uncited_fallback:
            # On the fallback path an item's own goal_id is not evidence
            # that the report used it, so the gate is widened from "this
            # item's goal" to "any goal this run pursued". Still a real
            # gate: an item sharing no distinctive term with ANY goal is
            # still dropped, which is what keeps D-59's nine Redis URLs out.
            every_goal_term = set().union(*goal_terms.values()) if goal_terms else set()
            goal_terms = {e.goal_id: every_goal_term for e in listed}
        # D-64: the reviewer's redirect guidance counts as on-topic too.
        # WHY: the gate above tests evidence against the goal descriptions
        # composed BEFORE the human intervened. A redirect changes what the
        # run is looking for -- E4's handler routes guidance to
        # gap_generator precisely so it reaches retrieval (see
        # agents/escalation.py) -- but the goal text is never rewritten to
        # match. So web pages fetched BECAUSE A HUMAN ASKED FOR THEM were
        # judged against goals that predate the request, found to share no
        # distinctive terms, and silently suppressed. Live shape: goals
        # about "political systems and governance structures", reviewer
        # guidance "UN reports of press freedom, democracy index", and the
        # resulting watchdog URLs dropped to a man -- the report came back
        # with no Sources section at all and nothing in the prose to
        # explain why.
        #
        # This inverts the right priority. The reviewer's guidance is the
        # most authoritative statement of relevance anywhere in the run: a
        # person read the report and said what was missing. Stale goal text
        # should not outrank it.
        #
        # D-59 still holds. Its motivating failure was nine Redis-monitoring
        # URLs listed under [g1] in an India-vs-US report; those match
        # neither the goal terms nor any guidance a reviewer of that report
        # would type, so they are still dropped. The union WIDENS the gate
        # by exactly one thing -- what the human explicitly asked for --
        # and by nothing else. With no redirect, `guidance` is "",
        # distinctive_terms("") is empty, and the union is byte-identical
        # to the pre-D-64 behaviour.
        guidance_terms = distinctive_terms(guidance)
        on_topic = [e for e in listed
                    if not goal_terms.get(e.goal_id)
                    or (goal_terms[e.goal_id] | guidance_terms)
                    & distinctive_terms(e.content)]
        if guidance_terms:
            # Visible when guidance is what saved an item, so a reviewer can
            # tell "the gate passed it" from "the gate passed it because I
            # asked for it" without re-running anything.
            strict = [e for e in listed
                      if not goal_terms.get(e.goal_id)
                      or goal_terms[e.goal_id] & distinctive_terms(e.content)]
            if len(on_topic) != len(strict):
                log_event(logger, "sources.kept_by_guidance",
                          rescued=len(on_topic) - len(strict),
                          on_goal_terms=len(strict))
        if len(on_topic) != len(listed):
            log_event(logger, "sources.off_topic_dropped",
                      dropped=len(listed) - len(on_topic),
                      kept=len(on_topic))
        listed = on_topic
    counters["web_sources_suppressed"] = float(len(web_items) - len(listed))
    if not listed:
        # Web evidence was retrieved but the compiler cited none of the goals
        # it belonged to. Worth a log line rather than silence: it is the
        # signature of the web tier doing work that never reached the report.
        log_event(logger, "sources.no_cited_web_evidence",
                  retrieved=len(web_items))
        return report, counters

    best_by_url: Dict[str, Evidence] = {}
    for item in listed:
        existing = best_by_url.get(item.url)
        if existing is None or item.score > existing.score:
            best_by_url[item.url] = item

    ordered = sorted(best_by_url.values(), key=lambda e: -e.score)

    lines = [SOURCES_HEADING, ""]
    if uncited_fallback:
        lines += [UNCITED_NOTE, ""]
    for i, item in enumerate(ordered, start=1):
        # The evidence content is "Title — snippet" (see
        # tools/mcp_client.py::make_web_search_tool). Only the title half is
        # wanted as a link label; the snippet is the substance and already
        # informed the prose above. Splitting on the same em-dash that
        # joined them, once, from the left.
        label = item.content.split(" — ", 1)[0].strip()
        # Fall back to the domain when there is no usable title, rather than
        # emitting an empty link label. Never falls back to the URL itself,
        # which would be unreadable at typical search-result URL lengths.
        if not label:
            label = item.domain or "source"
        domain = f" ({item.domain})" if item.domain else ""
        lines.append(f"{i}. [{item.goal_id}] {label}{domain} — {item.url}")
    block = "\n".join(lines)

    counters["web_sources_listed"] = float(len(ordered))
    if uncited_fallback:
        # Counted separately so telemetry, and anyone reading it later, can
        # always tell "the compiler cited these goals" from "the compiler
        # cited nothing and we listed what it had". Same reason D-144
        # counts citations_attached rather than quietly rescuing a report.
        counters["web_sources_listed_uncited"] = float(len(ordered))
    log_event(logger, "sources.web_sources_appended",
              listed=len(ordered),
              suppressed=int(counters["web_sources_suppressed"]),
              uncited_fallback=uncited_fallback)
    # rstrip then two newlines: the compiler's own output has inconsistent
    # trailing whitespace across providers, and a Sources heading that lands
    # one line below the last paragraph in one run and three in the next is
    # a diff nobody wants to read.
    return f"{report.rstrip()}\n\n{block}\n", counters
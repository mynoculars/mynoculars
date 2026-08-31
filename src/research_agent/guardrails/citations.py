"""
guardrails/citations.py — deterministic citation repair
(D-40/D-43/D-45/D-96/D-99).

CALLED BY   agents/compilation.py::compiler_node, immediately after the
            free-text compile call returns.

Moved from agents/compilation.py, unchanged — it was already a single,
self-contained function with no dependency on anything else in that file;
this relocation changes where it lives, not what it does or how it's
called.
"""

import re

# The MISSING DELIMITER signature this module exists to catch: a
# lowercase letter or digit immediately followed by a capital, with no
# space between them -- "...strategic support forcesChina's armed
# forces...". On its own this is a weak signal (it also matches "eBay",
# "LinkedIn", "McKinsey"), which is why a match is only ever acted on
# when the text that FOLLOWS it is verbatim evidence; see
# _paste_run_end. The signal locates the wound, the verbatim test
# confirms it.
_GLUE_RE = re.compile(r"(?<=[a-z0-9,)])(?=[A-Z])")

# Word boundaries, used to grow a candidate paste one word at a time.
_WORD_BOUNDARY_RE = re.compile(r"\s+")

# A paste is always a substantial run of source text. Six words is long
# enough that an accidental verbatim collision with the model's own
# prose is vanishingly unlikely, and short enough to catch the truncated
# pastes that motivated D-96 ("China fields 2,535,000 active troops.",
# seven words lifted out of the middle of a much longer web snippet).
_MIN_PASTE_WORDS = 6

# A CONTINUATION span -- one that follows an already-confirmed paste --
# is allowed to be shorter, because the expensive part of the judgement
# (is this a paste at all?) has already been settled by the glue site
# that opened the run. Live (run p205.251-check) the second span of the
# first run was "China fields 2,535,000 active troops." -- five words,
# and invisible to a flat six-word floor.
_MIN_CONTINUATION_WORDS = 4

# Hard stop on how far a single verbatim run may be grown, so a
# pathological report cannot turn this into a quadratic scan.
_MAX_PASTE_WORDS = 200


def _evidence_corpus(evidence) -> str:
    """All evidence content as one haystack for verbatim substring tests.

    Joined with a newline so a run cannot accidentally span two
    unrelated items: report prose never contains a bare newline in the
    middle of a sentence, and the runs grown below never cross one
    because the growth stops as soon as `in corpus` fails.
    """
    return "\n".join((e.content or "") for e in evidence)


def _word_ends(text: str, start: int, limit: int):
    """Offsets in `text` at which a word ending at/after `start` stops.

    Yields the end offset of the 1st, 2nd, ... word after `start`, up to
    `limit` words. Used to grow a candidate paste word by word instead
    of character by character -- growing by character would happily stop
    halfway through a word and delete "China fields 2,535,000 act".
    """
    pos = start
    for _ in range(limit):
        match = _WORD_BOUNDARY_RE.search(text, pos)
        if match is None:
            if pos < len(text):
                yield len(text)
            return
        pos = match.start()
        yield pos
        pos = match.end()


def _verbatim_run_end(report: str, start: int, corpus: str,
                      min_words: int = _MIN_PASTE_WORDS):
    """End offset of the longest verbatim-from-evidence run at `start`.

    Returns None when nothing at `start` is a run of at least
    `min_words` words that appears verbatim somewhere in `corpus`.

    Grows greedily: the longest prefix that is still a substring of the
    evidence wins. Growth stops at the first word that breaks the match,
    which is exactly where the model resumed writing in its own voice.
    """
    best = None
    words = 0
    for end in _word_ends(report, start, _MAX_PASTE_WORDS):
        words += 1
        candidate = report[start:end]
        if candidate in corpus:
            if words >= min_words:
                best = end
            continue
        # The model routinely closes a lifted span with punctuation the
        # source did not have -- live (run p205.251-check) it pasted
        # "China fields 2,535,000 active troops." out of a snippet that
        # reads "China fields 2,535,000 active troops vs 3,068,000 for
        # India". Comparing with the added '.' failed on the last word of
        # every such paste and left the tail of the run in the report.
        #
        # Retry once without trailing punctuation, and end the run at the
        # last character that ACTUALLY matched. That boundary matters in
        # both directions: punctuation the source did not have is left
        # standing in the report (where it closes the claim's own
        # sentence), and punctuation the CLAIM owns is never swallowed by
        # the deletion.
        # Trimmed ONE character at a time, longest match wins, so a run
        # keeps the punctuation the source really had and gives back only
        # what the claim owns: "...975,000 troops.." (the paste's full
        # stop plus the claim's) settles on "...975,000 troops.", not on
        # "...975,000 troops".
        trimmed = candidate
        while trimmed and trimmed[-1] in ".,;:!?":
            trimmed = trimmed[:-1]
            if trimmed in corpus:
                if words >= min_words:
                    best = start + len(trimmed)
                break
        break
    return best


def _paste_run_end(report: str, start: int, corpus: str):
    """End offset of a whole paste RUN beginning at `start`.

    A run is one or more verbatim evidence spans laid end to end. The
    model does not paste once and stop: live (run p205.251-check) it
    emitted "...support forces" + <source sentence A> + " " + <source
    sentence B> + "." -- only the FIRST span is glued to the claim, so a
    check that anchored on glue alone removed A and shipped B.

    Absorbing the continuation is safe precisely because it is gated
    behind a confirmed glue: an evidence sentence quoted with proper
    delimiters anywhere ELSE in the report is never reached by this
    function, which is what keeps D-45's p205.107-check protection
    intact (that run had corpus_recall 1.0 and legitimately restated
    corpus sentences; deleting those emptied six whole sections).
    """
    end = _verbatim_run_end(report, start, corpus)
    if end is None:
        return None
    while True:
        # Skip the punctuation and whitespace BETWEEN two pasted spans.
        # Both have to be skippable: the model closes each lifted span
        # with its own full stop, so the gap it leaves is ". " rather
        # than " ". Requiring whitespace alone stopped every run after
        # its first span. Anything skipped here falls inside the deleted
        # region and is removed with it.
        nxt = end
        while nxt < len(report) and (report[nxt].isspace()
                                     or report[nxt] in ".,;:!?"):
            nxt += 1
        if nxt == end:
            return end
        extended = _verbatim_run_end(report, nxt, corpus,
                                     _MIN_CONTINUATION_WORDS)
        if extended is None:
            return end
        end = extended


def _strip_pasted_evidence(report: str, evidence) -> tuple:
    """Remove glued runs of pasted evidence. Returns (report, count).

    Single left-to-right pass: every glue candidate is examined once, and
    a candidate that falls inside a span already removed is skipped.

    Sentence punctuation is CARRIED OVER rather than invented, and only
    where removing the run would otherwise weld two sentences together:
    the run must have ended in '.', '!' or '?', and the next non-space
    character must open a new sentence with a capital. So

      "...support forces<PASTE ending '.'> The PLA's ground component..."

    closes as "...support forces. The PLA's ground component...", while

      "...map naturally to hashes<PASTE ending '.'> and update partially."

    keeps flowing as one sentence and gains nothing. Where the claim's
    own punctuation already follows the run -- "...security
    roles<PASTE.>." -- nothing is added and the existing '.' stands.
    """
    corpus = _evidence_corpus(evidence)
    if not corpus.strip() or not report:
        return report, 0

    out = []
    pos = 0
    removed = 0
    for match in _GLUE_RE.finditer(report):
        start = match.start()
        if start < pos:
            continue
        end = _paste_run_end(report, start, corpus)
        if end is None:
            continue
        out.append(report[pos:start])
        tail = report[end - 1:end]
        following = report[end:].lstrip()[:1]
        if tail in ".!?" and following[:1].isupper():
            out.append(tail)
        pos = end
        removed += 1
    if not removed:
        return report, 0
    out.append(report[pos:])
    return "".join(out), removed


# ---------------------------------------------------------------------------
# D-137: the glue signature, WITHOUT the verbatim confirmation.
# ---------------------------------------------------------------------------
#
# _strip_pasted_evidence above acts only where the text following a glue
# site is VERBATIM evidence. Live (runs p205.276-check and p205.277-check,
# "Compare the Armies of China and India") the compiler stopped pasting and
# started PARAPHRASING: it wrote its own condensed restatement and welded
# it to the claim exactly where a [gN] marker belongs --
#
#   "...along the disputed Himalayan borderIndia raised a new mountain
#    strike corps to strengthen its defence along its disputed border with
#    China in the high reaches of the Himalayas."
#
# -- and that sentence appears NOWHERE in the evidence block. The verbatim
# test cannot see it, so `citations_pasted_evidence_removed` read 1,
# `citations_residual_paste_sites` read 0, and the two reports shipped with
# 9 and 22 welded sentence joins respectively. Both counters were telling
# the truth about PASTES and leaving a false impression about the REPORT.
#
# WHY A SECOND, NARROWER SIGNATURE RATHER THAN A LOOSER VERBATIM TEST.
# Overlap between the glued run and its supposed source was measured across
# every site in both shipped reports: 0.17 to 0.64 against the best-matching
# evidence item, and 0.00 to 1.00 against the claim it was welded to. No
# ratio separates these from ordinary prose, and a guard that DELETES text
# on a ratio is how D-45's p205.107-check failure happened (legitimate
# restatements removed, six whole sections emptied). So this pass tightens
# the SIGNATURE instead of loosening the confirmation:
#
#   left  -- at least four lowercase letters, so "eBay", "iPhone" and
#            "McKinsey" (two letters or fewer before the capital) and
#            "Type 054B" (digits, not letters) cannot match;
#   right -- a capital followed by at least two lowercase letters, so
#            "LinkedIn" ("In") and "...systemsA new..." ("A") cannot match;
#   run   -- at least six words, ending at sentence punctuation, so what is
#            being separated is a whole SENTENCE and never two words that
#            merely lost a space.
#
# Measured on those two shipped reports: 9 and 22 sites confirmed, and zero
# on a control set of eBay / LinkedIn / McKinsey / PayPal / iPhone /
# PostgreSQL / "Type 054B" sentences.
#
# AND IT INSERTS RATHER THAN DELETES. The verbatim test proves the words
# belong to the SOURCE, so removing them loses nothing the report wrote.
# This signature proves only that a sentence boundary is missing, which is
# a formatting fact and not a claim about authorship -- so the repair is
# the boundary itself. A false positive costs a spurious full stop; it can
# never cost a sentence.
_GLUED_SENTENCE_RE = re.compile(r"(?<=[a-z]{4})(?=[A-Z][a-z]{2,})")

# Sentence end: terminal punctuation followed by whitespace or the end of
# the report. Deliberately not a general sentence splitter -- it only has
# to answer "does a whole sentence follow this glue site?".
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")

# A glued run must be a whole sentence, not a lost space between two words.
_MIN_GLUED_WORDS = 6

# How far back to look for a URL before acting. A link's path can carry the
# same lowercase-then-capital shape ("/researchIndia..."), and a full stop
# inserted into a URL breaks it. Prose that carries a bare URL is already
# outside what compile_report asks for; this is belt and braces.
_URL_GUARD_CHARS = 60


def _confirmed_glue_sites(report: str):
    """Yield the offset of every confirmed glue site, left to right.

    One definition, read by both the repair and the residual counter, so
    the two can never disagree about what a glue site is -- the same rule
    residual_paste_sites states for itself.
    """
    for match in _GLUED_SENTENCE_RE.finditer(report):
        start = match.start()
        if "http" in report[max(0, start - _URL_GUARD_CHARS):start]:
            continue
        end_match = _SENTENCE_END_RE.search(report, start)
        if end_match is None:
            continue
        if len(report[start:end_match.end()].split()) >= _MIN_GLUED_WORDS:
            yield start


def repair_glued_sentences(report: str) -> tuple:
    """Restore the missing sentence boundary at each confirmed glue site.

    Returns (report, counters). Byte-identical, with empty counters, when
    nothing matches -- which is every report the compiler formats
    correctly, so that path must be exactly unchanged rather than merely
    similar.

    Runs AFTER _strip_pasted_evidence (see clean_citations) so a verbatim
    paste is REMOVED rather than merely punctuated: the stronger verdict
    gets first refusal at every site, the same ordering D-66's gate has
    ahead of D-91's figure audit.
    """
    if not report:
        return report, {}
    out, pos, repaired = [], 0, 0
    for start in _confirmed_glue_sites(report):
        if start < pos:
            continue
        out.append(report[pos:start])
        out.append(". ")
        pos = start
        repaired += 1
    if not repaired:
        return report, {}
    out.append(report[pos:])
    return "".join(out), {"citations_glued_sentences_repaired": float(repaired)}


def residual_glue_sites(report: str) -> int:
    """How many confirmed glue sites are STILL in a finished report.

    The companion to residual_paste_sites, here for that counter's own
    reason stated the other way round: it read 0 on two reports carrying 9
    and 22 welded joins, because a paste was the only thing it could see.
    This one reads the shipped artifact with the signature that actually
    matched them, so a repair that silently stops working cannot present
    as a clean report.
    """
    return sum(1 for _ in _confirmed_glue_sites(report)) if report else 0


# One citation block, in any delimiter the compiler has actually been
# observed using (D-99). Matches a bracket or paren containing ONLY goal
# ids, their separators, and an optional pipe-suffix -- nothing else. A
# parenthesis around prose ("(g1 is the largest)") cannot match, because
# the content between the delimiters has to be goal ids end to end.
_CITATION_BLOCK_RE = re.compile(
    r"[\[(]\s*(g\d+(?:\s*(?:,|;|and|&)\s*g\d+)*)\s*(\|[^\])]*)?\s*[\])]",
    re.IGNORECASE)

_GOAL_ID_RE = re.compile(r"g\d+", re.IGNORECASE)


def normalise_citation_form(report: str, goals) -> tuple:
    """Rewrite every citation the compiler wrote into the one form the
    rest of the system reads: `[gN]`, one goal per marker (D-99).

    WHY THIS EXISTS, and why its absence went unnoticed for so long.

    D-40 asks the model for `[gN]`. Everything downstream -- the Sources
    block (D-57), the zero-citation gate (D-66), the cited-figure audit
    (D-91) -- reads citations through `sources.py::cited_goal_ids`, whose
    pattern is exactly `\[g(\d+)\]`. Anything else the model writes is
    not a malformed citation to those readers; it is not a citation at
    all, and they fail silently and unanimously.

    Live (run p205.253-check, "Compare Armies of China and India") the
    compiler wrote its goal ids in PARENTHESES -- "## 1. Military
    Personnel Strength (g1)" through "(g4)" -- and the consequences ran
    the whole length of the pipeline: `cited_goal_ids` found zero, the
    D-66 gate failed the report twice, two revision cycles and an E4
    escalation were spent on a defect no rewrite instruction addresses,
    `append_web_sources` listed 0 of 78 web items so the report shipped
    with no Sources section at all, and D-91 audited nothing. The report
    still carried the D-85 provenance notice telling the reader to treat
    figures as unverified "unless a listed source confirms them" --
    above a page with no listed sources.

    This shape was KNOWN. guardrails/claims.py's own docstring lists
    `(g1)` in headings as one of four variants observed across live runs,
    and then asserts the problem is handled: "D-40 fixed the form,
    D-43/D-45's clean_citations repairs it deterministically." It does
    not. `clean_citations` DROPS markers for unevidenced goals; it has
    never normalised the form. D-91's feasibility argument rested on a
    repair nobody had written.

    WHAT IS AND IS NOT CONVERTED. Only a delimiter containing goal ids
    end to end is a citation:

        (g1)                     -> [g1]
        (g1, g4)                 -> [g1] [g4]
        [g1, g4]                 -> [g1] [g4]
        [g1 | corpus | score=.5] -> [g1]
        [g1]                     -> [g1]        (already correct, no-op)
        (g1 is the largest)      -> unchanged   (prose, not a citation)

    A block naming NO goal that exists in this run is left alone
    entirely. That keeps the rewrite from inventing a citation out of
    prose that merely looks like one, and it costs nothing: a marker for
    a goal that does not exist would be dropped by `clean_citations`
    immediately afterwards anyway.

    Returns (report, {counter_name: count}).
    """
    known = {g.goal_id.lower() for g in goals}
    if not report or not known:
        return report, {}

    changed = 0

    def _rewrite(match):
        nonlocal changed
        ids = [gid.lower() for gid in _GOAL_ID_RE.findall(match.group(1))]
        # De-duplicated, order preserved: "[g1, g1]" is one citation.
        seen, ordered = set(), []
        for gid in ids:
            if gid in known and gid not in seen:
                seen.add(gid)
                ordered.append(gid)
        if not ordered:
            return match.group(0)
        replacement = " ".join(f"[{gid}]" for gid in ordered)
        if replacement != match.group(0):
            changed += 1
        return replacement

    rewritten = _CITATION_BLOCK_RE.sub(_rewrite, report)
    counters = {"citations_form_normalised": float(changed)} if changed else {}
    return rewritten, counters


def residual_paste_sites(report: str, evidence) -> int:
    """How many glued evidence pastes are STILL in a finished report.

    `citations_pasted_evidence_removed` says what the guard took out. It
    says nothing about what it left, and those are different questions --
    a run reporting 21 removals and a run reporting 21 removals with four
    pastes still standing look identical in telemetry today.

    That gap is the D-96 lesson pointed the other way. D-96 existed
    because a guard went silent and its zero was read as "nothing to do";
    this counter exists so a guard that is working but INCOMPLETE cannot
    hide behind a healthy-looking removal count. Live (run
    p205.253-check) the shipped report carried four glued sites after 21
    removals, and nothing in the run record said so.

    Read-only: uses exactly the detector `_strip_pasted_evidence` uses,
    so the two numbers can never disagree about what a paste is.
    """
    corpus = _evidence_corpus(evidence)
    if not corpus.strip() or not report:
        return 0
    sites, pos = 0, 0
    for match in _GLUE_RE.finditer(report):
        start = match.start()
        if start < pos:
            continue
        end = _paste_run_end(report, start, corpus)
        if end is None:
            continue
        sites += 1
        pos = end
    return sites


def clean_citations(report: str, goals, evidence) -> tuple:
    """Deterministically repair two citation failures the prompt alone
    does not reliably prevent (D-40 asks for correct behaviour; this
    enforces what can be enforced without judging meaning).

    1. PASTED EVIDENCE TEXT. Live (run p205.95-check) the compiler ran the
       source sentence straight into the claim -- "...whole session
       blobRedis is an in-memory data store..." -- unreadable and
       unattributable. A verbatim run of evidence text glued to the
       preceding word is removed, together with any further evidence text
       laid contiguously after it.

       D-96 replaced the original whole-body exact match. That match
       required an evidence item's ENTIRE content to appear in the prose,
       which held for short corpus chunks and essentially never for web
       snippets -- live (run p205.251-check, "Compare Armies of China and
       India") every paragraph of the shipped report carried glued source
       text and this function removed nothing at all, because the model
       pastes a SPAN out of a long snippet, not the whole snippet. The
       replacement matches any verbatim run of at least six words. See
       _strip_pasted_evidence and the helpers above it.
    2. CITATIONS TO GOALS WITH NO EVIDENCE. A [gN] marker asserts that goal
       N's retrieved evidence supports the sentence. If goal N retrieved
       nothing at all, that assertion is false on its face and the marker
       is dropped.

    What this deliberately does NOT do: judge whether an evidence-BACKED
    goal's evidence actually supports a given sentence. Live (run
    p205.98-check) the report cited [g5] for "Netflix operates Cassandra
    clusters exceeding 1 PB ... ~1 million writes per second" while g5's
    evidence was Redis session-caching text. Detecting that requires
    reading meaning, which is the critic's job -- see templates.critique,
    which now asks for it explicitly. This function is the deterministic
    half; it is not a substitute for the semantic half.

    Returns (cleaned_report, {counter_name: count}).
    """
    counters = {}

    # The counter name is unchanged from D-45 so existing dashboards and
    # the D-88 per-report guardrail block keep working, but what it
    # counts is now RUNS removed rather than whole-body matches removed.
    cleaned, pasted = _strip_pasted_evidence(report, evidence)
    if pasted:
        counters["citations_pasted_evidence_removed"] = float(pasted)

    evidenced = {e.goal_id for e in evidence}
    unevidenced = {g.goal_id for g in goals if g.goal_id not in evidenced}
    dropped = 0
    for goal_id in unevidenced:
        marker = f"[{goal_id}]"
        dropped += cleaned.count(marker)
        cleaned = cleaned.replace(marker, "")
    if dropped:
        counters["citations_to_unevidenced_goals"] = float(dropped)

    return cleaned, counters

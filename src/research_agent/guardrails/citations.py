"""
guardrails/citations.py — deterministic citation repair (D-40/D-43/D-45).

CALLED BY   agents/compilation.py::compiler_node, immediately after the
            free-text compile call returns.

Moved from agents/compilation.py, unchanged — it was already a single,
self-contained function with no dependency on anything else in that file;
this relocation changes where it lives, not what it does or how it's
called.
"""


def clean_citations(report: str, goals, evidence) -> tuple:
    """Deterministically repair two citation failures the prompt alone
    does not reliably prevent (D-40 asks for correct behaviour; this
    enforces what can be enforced without judging meaning).

    1. PASTED EVIDENCE TEXT. Live (run p205.95-check) the compiler ran the
       source sentence straight into the claim -- "...whole session
       blobRedis is an in-memory data store..." -- unreadable and
       unattributable. Any verbatim run of an evidence item's own content
       appearing in the prose is removed.
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
    cleaned = report

    pasted = 0
    for e in evidence:
        body = (e.content or "").strip()
        # Short fragments produce false positives against ordinary prose;
        # a pasted citation is always a whole retrieved sentence.
        if len(body) < 40:
            continue
        # Only strip a match that is GLUED to the preceding word. Removing
        # every verbatim occurrence was far too blunt: on an in-corpus
        # query the compiler legitimately states corpus sentences almost
        # word for word, which is what a grounded report is supposed to do.
        # Live (run p205.107-check, "Compare Redis vs Memcached for
        # production systems"): retrieval was perfect -- corpus_recall 1.0,
        # 36 corpus items -- and this function deleted six whole sections
        # of the finished report, shipping "### Scalability" and "###
        # Security" as empty headings. The defect this guards against was
        # never quoting; it was the MISSING DELIMITER, e.g. "...rewriting
        # the whole session blobRedis is an in-memory data store...", where
        # the source sentence runs into the claim with no boundary. That
        # signature is exactly detectable: a preceding alphanumeric
        # character. Properly separated evidence text is left alone.
        start = cleaned.find(body)
        while start != -1:
            glued = start > 0 and cleaned[start - 1].isalnum()
            if not glued:
                start = cleaned.find(body, start + len(body))
                continue
            cleaned = cleaned[:start] + cleaned[start + len(body):]
            pasted += 1
            start = cleaned.find(body)
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

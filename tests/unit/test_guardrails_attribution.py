"""
tests/unit/test_guardrails_attribution.py — guardrails/attribution.py (D-144).

Covers deterministic citation attachment: the all-or-nothing gate, the
unambiguity and overlap rules that stop it inventing attribution, the
add-only contract, and a replay of the shipped p205.280-check report that
motivated the whole pass.
"""

import pytest

from research_agent.guardrails.attribution import (MIN_TERM_OVERLAP,
                                                   attach_missing_citations)
from research_agent.guardrails.sources import cited_goal_ids
from research_agent.state import Evidence, Goal, Volatility


def _ev(goal_id, content, source="web", score=0.72):
    return Evidence(task_key=f"{goal_id}:{content[:12]}", goal_id=goal_id,
                    source=source, content=content, score=score,
                    volatility=Volatility.SEMI_STABLE)


def _goal(goal_id, description):
    return Goal(goal_id=goal_id, description=description)


# ---------------------------------------------------------------------------
# The all-or-nothing gate
# ---------------------------------------------------------------------------


def test_a_report_that_already_cites_is_left_byte_identical():
    """A partially cited report is a model making choices about which
    sentence rests on which evidence. Second-guessing those needs to read
    meaning, which is the critic's job."""
    report = "China fields more submarines than India by a wide margin. [g1]\nIndia operates one carrier and is building another one now.\n"
    goals = [_goal("g1", "naval forces")]
    evidence = [_ev("g1", "India operates one aircraft carrier submarines navy")]

    out, counters = attach_missing_citations(report, goals, evidence)

    assert out == report
    assert counters == {}


def test_no_goal_has_evidence_so_nothing_is_attached():
    report = "China fields more submarines than India by a very wide margin.\n"
    goals = [_goal("g1", "naval forces")]

    out, counters = attach_missing_citations(report, goals, [])

    assert out == report and counters == {}


def test_memory_evidence_alone_never_makes_a_sentence_look_supported():
    """P2-02 namespaces recall as memory::gN, which is deliberately never a
    current goal id. Recalled text must not be able to attribute a sentence
    to this run's retrieval."""
    report = "Redis stores sessions in memory and Memcached uses slab allocation here.\n"
    goals = [_goal("g1", "session caching")]
    evidence = [_ev("memory::g1", "Redis memory Memcached slab allocation sessions",
                    source="memory")]

    out, counters = attach_missing_citations(report, goals, evidence)

    assert out == report and counters == {}


# ---------------------------------------------------------------------------
# The rules that stop it inventing attribution
# ---------------------------------------------------------------------------


def test_a_tie_between_two_goals_attributes_nothing():
    """"Probably g1" is not a citation. Two goals whose evidence shares the
    same terms with the sentence is genuinely undecidable here."""
    report = "Submarine fleets and carrier aviation both expanded considerably.\n"
    goals = [_goal("g1", "navy"), _goal("g2", "aviation")]
    evidence = [_ev("g1", "submarine carrier aviation fleets expanded"),
                _ev("g2", "submarine carrier aviation fleets expanded")]

    out, counters = attach_missing_citations(report, goals, evidence)

    assert out == report and counters == {}


def test_one_shared_term_is_below_the_floor():
    """One shared word is the bar that produced D-59's motivating failure
    -- nine Redis-monitoring URLs listed under [g1] in an India-vs-US
    report."""
    assert MIN_TERM_OVERLAP == 2
    report = "Beijing has pursued a broad range of institutional reforms lately.\n"
    goals = [_goal("g1", "reform")]
    evidence = [_ev("g1", "institutional restructuring across provinces")]

    out, counters = attach_missing_citations(report, goals, evidence)

    assert out == report and counters == {}


def test_short_fragments_are_not_claims():
    """A four-word fragment can clear a two-term overlap on its own subject
    and gains nothing from a citation."""
    report = "Naval forces. Air forces.\n"
    goals = [_goal("g1", "branches")]
    evidence = [_ev("g1", "naval forces air forces branches strength")]

    out, counters = attach_missing_citations(report, goals, evidence)

    assert out == report and counters == {}


def test_headings_are_never_cited():
    """Citing a heading would open a SCOPE (claims.py::iter_cited_sentences)
    and silently attribute every sentence beneath it."""
    report = ("## Modernization Efforts and Technological Advancements\n"
              "China invested heavily in next-generation platforms and networked systems.\n")
    goals = [_goal("g3", "modernization")]
    evidence = [_ev("g3", "next-generation platforms networked systems modernization investment")]

    out, counters = attach_missing_citations(report, goals, evidence)

    lines = out.splitlines()
    assert lines[0] == "## Modernization Efforts and Technological Advancements"
    assert cited_goal_ids(lines[0]) == set()
    assert counters["citations_attached"] == 1


# ---------------------------------------------------------------------------
# The add-only contract
# ---------------------------------------------------------------------------


def test_it_only_adds_and_never_alters_the_prose():
    report = "China operates multiple aircraft carriers and advanced expeditionary platforms.\n"
    goals = [_goal("g2", "naval forces")]
    evidence = [_ev("g2", "aircraft carriers expeditionary platforms operated by China")]

    out, counters = attach_missing_citations(report, goals, evidence)

    assert counters["citations_attached"] == 1
    assert out == ("China operates multiple aircraft carriers and advanced "
                   "expeditionary platforms. [g2]\n")


def test_the_sources_block_and_provenance_notice_are_out_of_scope():
    """D-139's rule: text this codebase generated is not text to annotate.

    Also the reason cited_goal_ids_in_prose exists: the Sources block below
    contains "[g2]", and a whole-report read of it would conclude this
    report is already cited and skip the pass entirely.
    """
    report = (
        "> **Provenance notice — inserted automatically, not written by the model.**\n"
        "> None of this report's 1 research goal(s) are supported by a document.\n"
        "\n"
        "China operates multiple aircraft carriers and advanced expeditionary platforms.\n"
        "\n"
        "## Sources\n"
        "\n"
        "1. [g2] Chinese carrier aviation platforms (x.com) — https://x.com/1\n")
    goals = [_goal("g2", "naval forces")]
    evidence = [_ev("g2", "aircraft carriers expeditionary platforms operated by China")]

    out, counters = attach_missing_citations(report, goals, evidence)

    assert counters["citations_attached"] == 1
    assert out.startswith("> **Provenance notice")
    assert out.endswith("1. [g2] Chinese carrier aviation platforms (x.com) — https://x.com/1\n")


# ---------------------------------------------------------------------------
# Replay: the run that motivated this
# ---------------------------------------------------------------------------


P205_280_REPORT = """# Comparative Analysis of China's and India's Armed Forces

## 1. Size and Composition of Active-Duty Military Forces

China's active-duty military is substantially larger than India's. India's active force is reported at approximately 1.45 million personnel, while China's active force is significantly larger.

## 3. Modernization Efforts and Technological Advancements

China's military modernization has been systematic and sustained, with large-scale investments in next-generation platforms and indigenous defense research.
"""


def _p205_280_inputs():
    goals = [
        _goal("g1", "Compare the size and composition of the active-duty "
                    "military forces of China and India"),
        _goal("g3", "Examine the modernization efforts and technological "
                    "advancements in the military forces of both nations"),
    ]
    evidence = [
        _ev("g1", "India Military Strength — India ranks #4 with 1.45 million "
                  "active personnel"),
        _ev("g1", "China military personnel by type 2023 — active personnel "
                  "2,035,000"),
        _ev("g3", "Defence R&D budgets — modernization, next-generation "
                  "platforms, indigenous development"),
    ]
    return goals, evidence


def test_the_shipped_p205_280_report_is_rescued():
    """The exact failure: 100 evidence items, zero [gN] markers, and three
    guardrails silently dead together (Sources listed 0 of 58, D-91 audited
    nothing, and an E4 escalation a human approved because the prose read
    fine)."""
    goals, evidence = _p205_280_inputs()

    assert cited_goal_ids(P205_280_REPORT) == set(), "precondition"

    out, counters = attach_missing_citations(P205_280_REPORT, goals, evidence)

    assert cited_goal_ids(out) == {"g1", "g3"}
    assert counters["citations_attached"] >= 3


def test_the_rescue_is_counted_so_it_is_never_invisible():
    """A reader must always be able to tell a report the model cited from
    one this module rescued."""
    goals, evidence = _p205_280_inputs()

    _, counters = attach_missing_citations(P205_280_REPORT, goals, evidence)

    assert "citations_attached" in counters


def test_running_it_twice_changes_nothing_the_second_time():
    """compiler_node runs again on every revision; the pass must be
    idempotent or a rewrite accumulates markers."""
    goals, evidence = _p205_280_inputs()

    once, _ = attach_missing_citations(P205_280_REPORT, goals, evidence)
    twice, counters = attach_missing_citations(once, goals, evidence)

    assert twice == once and counters == {}


@pytest.mark.parametrize("empty", ["", "\n", "# Heading only\n"])
def test_degenerate_reports_are_returned_untouched(empty):
    goals, evidence = _p205_280_inputs()

    out, counters = attach_missing_citations(empty, goals, evidence)

    assert out == empty and counters == {}


# ---------------------------------------------------------------------------
# The latent defect D-144 had to fix before it could decouple Sources
#
# cited_goal_ids matches [gN] ANYWHERE, and every Sources entry begins
# "1. [g1] " by construction (D-57). So a report whose prose cites nothing
# but which carries a Sources block read back as fully cited -- which would
# have made evidence_cited wrong, the D-66 zero-citation gate silent, and
# telemetry's backstop agree with both.
# ---------------------------------------------------------------------------


def _uncited_report_with_sources():
    return (
        "China operates multiple aircraft carriers and expeditionary platforms.\n"
        "\n"
        "## Sources\n"
        "\n"
        "1. [g2] Chinese carrier aviation (x.com) — https://x.com/1\n"
        "2. [g2] PLAN order of battle (y.com) — https://y.com/2\n")


def test_a_sources_block_no_longer_masquerades_as_prose_citations():
    from research_agent.guardrails.claims import cited_goal_ids_in_prose

    report = _uncited_report_with_sources()

    assert cited_goal_ids(report) == {"g2"}, "the old whole-report read"
    assert cited_goal_ids_in_prose(report) == set(), "what the prose says"


def test_prose_citations_are_still_seen_when_a_sources_block_exists():
    from research_agent.guardrails.claims import cited_goal_ids_in_prose

    report = (
        "China operates multiple aircraft carriers. [g2]\n"
        "\n"
        "## Sources\n"
        "\n"
        "1. [g5] Something else entirely (x.com) — https://x.com/1\n")

    assert cited_goal_ids_in_prose(report) == {"g2"}


def test_the_pass_still_runs_on_a_report_that_only_cites_in_its_sources():
    goals = [_goal("g2", "naval forces")]
    evidence = [_ev("g2", "aircraft carriers expeditionary platforms operated by China")]

    out, counters = attach_missing_citations(_uncited_report_with_sources(),
                                             goals, evidence)

    assert counters["citations_attached"] == 1
    assert out.splitlines()[0].endswith("[g2]")


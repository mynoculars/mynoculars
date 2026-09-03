"""Tests for guardrails/critique.py -- D-155.

The notes replayed here reproduce the four objections that failed run
p205.287-check; NOTE_2M is the run's own wording. The evidence strings
carry the fragments the run's own evidence carried -- 2.1, 1.23, 1948 and
2015 each appeared verbatim in an evidence item while the critic was
asserting they did not. The module was written against a real verdict,
not a hypothetical one, and these tests keep it that way.
"""
import logging

import pytest

from research_agent.guardrails.critique import (disputed_figures,
                                                falsified_by_evidence,
                                                resolve_verdict)
from research_agent.state import Evidence


def _e(content, goal_id="g1"):
    return Evidence(task_key="t", goal_id=goal_id, source="corpus",
                    content=content, score=0.9)


# The live notes. Shortened only where the omitted text carries no figure.
NOTE_2M = ("Unfaithful: The report claims the PLA 'fields the world's "
           "largest standing military with approximately 2 million "
           "personnel'. The evidence states 'approximately 2 million to "
           "2.1 million', so the report's figure omits the upper bound.")
NOTE_INDIA = ("Unfaithful: the report gives India's active strength as "
              "1.23 million, which appears in no evidence item.")
NOTE_1948 = ("Unfaithful: the report dates the founding to 1948; no "
             "evidence item supports 1948.")
NOTE_2015 = ("Unfaithful: the report attributes the reforms to 2015, "
             "which the evidence does not state.")

LIVE_NOTES = [NOTE_2M, NOTE_INDIA, NOTE_1948, NOTE_2015]

# Every figure above, in the wording the run's evidence actually used.
LIVE_EVIDENCE = [
    _e("The PLA is estimated at approximately 2 million to 2.1 million "
       "active personnel."),
    _e("India's armed forces number roughly 1.23 million active troops."),
    _e("The force was formally established in 1948 and reorganised "
       "repeatedly thereafter."),
    _e("A round of structural reforms began in 2015."),
]


def test_years_count_here_even_though_claims_figures_in_drops_them():
    """guardrails/claims.py::figures_in deliberately ignores bare years --
    a year in a heading is formatting. This module's input is a critic
    NOTE, where the year IS the disputed claim, so it must not inherit
    that rule. 1948 and 2015 were two of the four live objections."""
    from research_agent.guardrails.claims import figures_in

    assert figures_in("the founding was in 1948") == set()
    assert disputed_figures(NOTE_1948) == {"1948"}


def test_a_note_naming_no_figure_yields_no_figures():
    """Coverage and semantic findings are what an LLM critic is FOR. They
    must be unadjudicatable here, and an empty set is how this says so."""
    assert disputed_figures("the report never addresses goal g3") == set()


def test_short_numbers_are_ignored():
    """A bare '2' or '31' matches almost any evidence block by accident,
    and dismissing a note on an accidental match is worse than no rule."""
    assert disputed_figures("goal 2 is weak, see item 31") == set()


def test_thousands_separators_normalise_on_both_sides():
    assert disputed_figures("the evidence says 2,300,000") == {"2300000"}
    assert falsified_by_evidence("the report says 2,300,000",
                                 "about 2300000 personnel") is True


def test_a_note_whose_every_figure_is_in_the_evidence_is_falsified():
    text = " ".join(e.content for e in LIVE_EVIDENCE)
    for note in LIVE_NOTES:
        assert falsified_by_evidence(note, text) is True, note


def test_a_note_naming_an_absent_figure_survives():
    """The critic doing its job correctly."""
    text = " ".join(e.content for e in LIVE_EVIDENCE)
    assert falsified_by_evidence(
        "Unfaithful: the report claims 4.7 million reservists.", text) is False


def test_a_note_quoting_both_figures_survives_by_design():
    """The KNOWN LIMIT, asserted so it stays deliberate. A note that
    quotes the report's figure AND the evidence's -- "the report says 1.4
    where the evidence says 1.23" -- names 1.4, which is absent from the
    evidence precisely because the report rounded it. Nothing here can
    tell which of the two figures the note is disputing, so it survives
    and the verdict stands. Guessing would mean dismissing notes on a
    figure this module never checked, which is the failure it exists to
    prevent, in reverse."""
    text = " ".join(e.content for e in LIVE_EVIDENCE)

    assert falsified_by_evidence(
        "Unfaithful: the report says 1.4 million where the evidence "
        "says 1.23 million.", text) is False


def test_a_note_naming_no_figure_is_never_falsified():
    text = " ".join(e.content for e in LIVE_EVIDENCE)
    assert falsified_by_evidence(
        "the report never addresses goal g3", text) is False


def test_the_live_p205_287_verdict_is_resolved(caplog):
    """The whole reason this module exists: four notes, every figure in
    the evidence, D-91 clean -- and an E4 escalation on a correct
    report."""
    with caplog.at_level(logging.WARNING):
        passed, notes, counters = resolve_verdict(
            False, LIVE_NOTES, LIVE_EVIDENCE, 0)

    assert passed is True
    assert notes == LIVE_NOTES, "the notes are kept; only the verdict moves"
    assert counters == {"critique_notes_dismissed": 4.0}
    assert [r for r in caplog.records
            if "critic.failure_not_corroborated" in r.message], \
        "a resolved verdict must never be silent"


def test_one_surviving_note_stops_the_whole_flip():
    """Not per-note arithmetic: a single finding this cannot adjudicate
    leaves the critic's verdict exactly as it stood."""
    notes = LIVE_NOTES + ["the report never addresses goal g3"]

    passed, out, counters = resolve_verdict(False, notes, LIVE_EVIDENCE, 0)

    assert passed is False
    assert out == notes
    assert counters == {}


def test_a_genuinely_absent_figure_stops_the_flip():
    notes = [NOTE_2M, "Unfaithful: the report claims 4.7 million reservists."]

    passed, _, counters = resolve_verdict(False, notes, LIVE_EVIDENCE, 0)

    assert passed is False
    assert counters == {}


def test_a_dirty_d91_audit_stops_the_flip():
    """The two checks are only in conflict when one of them is clean. If
    the deterministic audit flagged anything on this report, it AGREES
    with the critic and the verdict stands however the notes read."""
    passed, _, counters = resolve_verdict(False, LIVE_NOTES, LIVE_EVIDENCE, 1)

    assert passed is False
    assert counters == {}


@pytest.mark.parametrize("passed,notes", [
    (True, []),
    (True, ["Goal g1: supported by evidence [g1]."]),
    (False, []),
])
def test_the_no_op_paths_return_their_inputs_unchanged(passed, notes):
    """The healthy path is the common one and must stay exact."""
    out_passed, out_notes, counters = resolve_verdict(
        passed, notes, LIVE_EVIDENCE, 0)

    assert (out_passed, out_notes, counters) == (passed, notes, {})


def test_empty_evidence_cannot_falsify_anything():
    """With nothing to check against, every note survives -- the check
    fails CLOSED, leaving the critic's failure in place."""
    passed, _, counters = resolve_verdict(False, LIVE_NOTES, [], 0)

    assert passed is False
    assert counters == {}


# ---------------------------------------------------------------------------
# D-162 -- a section number is not a disputed figure
# ---------------------------------------------------------------------------


def _one_evidence(text):
    from research_agent.state import Evidence, Volatility
    return [Evidence(task_key="k", goal_id="g1", source="corpus", content=text,
                     score=0.9, volatility=Volatility.SEMI_STABLE)]


def test_a_coverage_note_citing_a_section_number_is_not_falsifiable():
    """The live shape that made this module do the opposite of its job.

    "section 2.1" is a LOCATION. It cleared MIN_FIGURE_CHARS, "2.1"
    appeared in the evidence by coincidence, and the note therefore read
    as refuted -- so a critic failure carrying nothing but a coverage
    finding flipped to PASS and the report routed to memory_writer, which
    D-24 exists to prevent."""
    from research_agent.guardrails.critique import disputed_figures, resolve_verdict

    note = ("Goal g3 (defence budgets) is never addressed; section 2.1 of "
            "the report is missing entirely.")
    evidence = _one_evidence("China fields approximately 2.1 million active personnel.")

    assert disputed_figures(note) == set()
    passed, notes, counters = resolve_verdict(False, [note], evidence, 0)
    assert passed is False
    assert notes == [note]
    assert counters == {}


def test_every_location_word_is_scrubbed_not_just_section():
    from research_agent.guardrails.critique import disputed_figures

    for phrase in ("section 2.1", "part 3.4", "step 12", "figure 4.2",
                   "table 7.1", "item 15", "paragraph 3.2", "page 104",
                   "goal 2.5", "appendix 1.1", "chapter 9.9", "line 250"):
        assert disputed_figures(f"The report's {phrase} is missing.") == set(), phrase


def test_a_real_figure_dispute_is_still_adjudicable_even_beside_a_section_ref():
    """Deliberately narrow: only a number IMMEDIATELY after a location
    word is dropped. A note disputing a figure must stay falsifiable, or
    D-155's counterweight stops working for the case it was built for."""
    from research_agent.guardrails.critique import disputed_figures, resolve_verdict

    note = "The 2.1 million figure in section 3 merely restates the evidence."
    evidence = _one_evidence("China fields approximately 2.1 million active personnel.")

    assert disputed_figures(note) == {"2.1"}
    passed, _notes, counters = resolve_verdict(False, [note], evidence, 0)
    assert passed is True
    assert counters == {"critique_notes_dismissed": 1.0}


def test_a_figure_the_evidence_lacks_still_stops_the_flip():
    from research_agent.guardrails.critique import resolve_verdict

    note = "The report claims 9.87 million reservists, which no evidence supports."
    evidence = _one_evidence("China fields approximately 2.1 million active personnel.")

    passed, _notes, counters = resolve_verdict(False, [note], evidence, 0)
    assert passed is False
    assert counters == {}
# --------------------------------------------------------------------------
# D-181: an entry means "this is wrong, fix it". templates.compile renders
# every one under "Address every note", so an affirmation becomes an
# instruction to change something correct.
# --------------------------------------------------------------------------


# Verbatim from p205.308-check's second critique: 23 entries, 21 of them
# recording that a claim IS supported.
_P205_308_AFFIRMATIONS = [
    "g1: The report states China's PLA has over 2.1 million active "
    "personnel; evidence shows 2,035,000 active personnel and 3,170,900 "
    "total soldiers, so the claim is faithful.",
    "g1: The report states India's active troops are 1,455,550; evidence "
    "shows 14,55,550 active troops, so the claim is faithful.",
    "g1: The report states the PLAGF has 975,000 deployed troops; evidence "
    "shows 975,000 troops, so the claim is faithful.",
    "g2: The report states the CMC is the supreme military leadership "
    "body; evidence supports this claim.",
    "g3: The report states the U.S. DoD refers to China as the 'pacing "
    "challenge'; evidence supports this claim.",
]
_P205_308_VIOLATIONS = [
    "g1: The report states China's PLA is the third largest standing army; "
    "evidence does not explicitly state this ranking, so the claim is "
    "unsupported.",
    "g1: The report states India is the fourth-largest military power "
    "globally; evidence does not explicitly state this ranking, so the "
    "claim is unsupported.",
]


def test_affirmations_are_dropped_and_violations_survive():
    from research_agent.guardrails.critique import drop_affirmations

    kept, dropped = drop_affirmations(
        _P205_308_AFFIRMATIONS + _P205_308_VIOLATIONS)

    assert kept == _P205_308_VIOLATIONS
    assert dropped == len(_P205_308_AFFIRMATIONS)


def test_a_denial_of_support_is_never_dropped():
    """The veto. An entry that denies support is kept whatever else it
    says -- including the word "supported" inside "unsupported"."""
    from research_agent.guardrails.critique import drop_affirmations

    kept, dropped = drop_affirmations([
        "g1: evidence supports the surrounding sentence but the figure "
        "8,000 is not supported by any evidence item.",
        "g2: the claim is unsupported.",
    ])

    assert dropped == 0
    assert len(kept) == 2


def test_a_clean_critique_is_untouched():
    from research_agent.guardrails.critique import drop_affirmations

    assert drop_affirmations([]) == ([], 0)
    assert drop_affirmations(_P205_308_VIOLATIONS) == (_P205_308_VIOLATIONS, 0)

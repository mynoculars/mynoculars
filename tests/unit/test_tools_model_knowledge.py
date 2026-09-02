"""
tests/unit/test_tools_model_knowledge.py — Guardrail G3
(tools/model_knowledge.py::_looks_overspecific and its wiring into
Evidence.hedge_specific).

Regression target: run p205.131-check, where the model tier produced a
claim pairing a specific year-range with a specific figure --
"India's population grew from approximately 900 million in 1970 to over
1.4 billion in 2020" -- that the critic later rejected as fabricated;
no evidence item stated that figure. Self-reported `confidence` did not
catch it; this deterministic heuristic is the guardrail that flags it
for the compiler instead.
"""


from research_agent.tools.model_knowledge import (_looks_overspecific,
                                                    make_model_knowledge_tool)
from research_agent.state import SearchTask


class _FakeRouter:
    def __init__(self, payload: dict):
        self._payload = payload

    def complete_json(self, messages):
        return self._payload


def _task() -> SearchTask:
    return SearchTask(key="g1::t1", goal_id="g1", query="population trends")


def test_looks_overspecific_flags_year_plus_figure():
    assert _looks_overspecific(
        "India's population grew from approximately 900 million in 1970 "
        "to over 1.4 billion in 2020.")
    assert _looks_overspecific(
        "By 2023, unemployment had fallen to 3.6 percent.")


def test_looks_overspecific_does_not_flag_year_alone():
    assert not _looks_overspecific(
        "China implemented the One-Child Policy in 1979.")


def test_looks_overspecific_does_not_flag_figure_alone():
    assert not _looks_overspecific(
        "The population density is approximately 153 people per square kilometer.")


def test_model_knowledge_tool_sets_hedge_specific_on_flagged_claims():
    router = _FakeRouter({"claims": [
        {"text": "India's population grew from approximately 900 million "
                 "in 1970 to over 1.4 billion in 2020.", "confidence": 0.9},
        {"text": "India has a large and youthful population.",
         "confidence": 0.9},
    ]})
    tool = make_model_knowledge_tool(router, score=0.6)
    evidence = tool(_task())

    assert len(evidence) == 2
    flagged = {e.content: e.hedge_specific for e in evidence}
    assert flagged["India's population grew from approximately 900 million "
                   "in 1970 to over 1.4 billion in 2020."] is True
    assert flagged["India has a large and youthful population."] is False


def test_model_knowledge_tool_never_flags_a_low_confidence_dropped_claim():
    """Confidence<0.5 is dropped entirely before hedge_specific is even
    evaluated -- the two guardrails are independent, not layered."""
    router = _FakeRouter({"claims": [
        {"text": "India's population grew from approximately 900 million "
                 "in 1970 to over 1.4 billion in 2020.", "confidence": 0.2},
    ]})
    tool = make_model_knowledge_tool(router, score=0.6)
    assert tool(_task()) == []


# ---------------------------------------------------------------------------
# P205.134 follow-up: wider unit vocabulary + the "%" boundary bug fix
# ---------------------------------------------------------------------------


def test_looks_overspecific_flags_energy_mass_and_area_units():
    """Regression target: run p205.134-check. The critic rejected five
    fabricated claims as unsupported; three used the exact year+quantity
    pairing this guard targets, but in units the original list didn't
    cover at all (only %/million/billion/trillion/per-X)."""
    assert _looks_overspecific(
        "India's National Solar Mission targets 500 GW of non-fossil "
        "capacity by 2030.")
    assert _looks_overspecific(
        "India's CO2 emissions per capita in 2022 were about 1.9 "
        "metric tons.")
    assert _looks_overspecific(
        "The United States had a per capita ecological footprint of "
        "about 8.0 global hectares as of 2021.")


def test_looks_overspecific_flags_bare_percent_sign_with_a_year():
    """Regression for a bug found while widening the unit list: `%\\b`
    only matches when a WORD character immediately follows the sign
    (e.g. "50%increase") -- ordinary prose always has a space after
    "%", and space is itself non-word, so no \\b boundary ever existed
    there. This means the ORIGINAL percentage branch effectively never
    matched real text before this fix, silently making every bare-%
    year+quantity claim (a common shape -- "grew 6.7% in 2022") invisible
    to this guard from day one."""
    assert _looks_overspecific(
        "The United States reduced total CO2 emissions by about 14% "
        "between 2010 and 2020.")
    assert _looks_overspecific("India's GDP grew 6.7% in 2022.")


def test_looks_overspecific_still_ignores_plain_counts():
    """The widened unit list must not start flagging ordinary numbers
    that carry no false-precision risk -- a bare count with a year
    nearby but no recognized unit stays unflagged."""
    assert not _looks_overspecific(
        "India fielded 3 goals in the 2020 review cycle.")


def test_looks_overspecific_flags_air_quality_units():
    """Regression target: run p205.136-check. A shipped report stated
    "Delhi's annual average exceeding 100 µg/m³ in multiple years"
    unflagged -- the concentration-unit class this domain (environmental
    /air-quality goals) keeps producing wasn't covered yet, same pattern
    as the GW/tonnes/hectares additions before it."""
    assert _looks_overspecific(
        "Delhi's PM2.5 concentration exceeded 100 \u00b5g/m\u00b3 in 2022.")
    assert _looks_overspecific(
        "Beijing recorded PM2.5 levels of 45 \u03bcg/m3 in 2021.")
    assert _looks_overspecific(
        "Ozone levels hit 70 ppb in 2020.")


def test_looks_overspecific_does_not_flag_air_quality_unit_without_a_year():
    """The same year+quantity pairing rule applies here as everywhere
    else in this guard -- a concentration figure with no specific year
    attached (e.g. "exceeding 100 µg/m³ in multiple years") is not
    flagged, by design."""
    assert not _looks_overspecific(
        "Delhi's annual average exceeding 100 \u00b5g/m\u00b3 in multiple years.")


# ---------------------------------------------------------------------------
# D-163 -- admission and capability are one comparison
# ---------------------------------------------------------------------------


class _ConfidenceStub:
    def __init__(self, confidence):
        self.confidence = confidence

    def set_node(self, node):
        pass

    def complete_json(self, messages):
        return {"claims": [{"text": "A stable fact about the matter.",
                            "confidence": self.confidence}]}


def _admit(confidence, score=0.6, floor=0.5):
    from research_agent.state import SearchTask
    from research_agent.tools.model_knowledge import make_model_knowledge_tool

    tool = make_model_knowledge_tool(_ConfidenceStub(confidence), score,
                                     min_evidence_score=floor)
    return tool(SearchTask(key="k", query="q", goal_id="g1"))


def test_a_claim_that_could_never_cover_a_goal_is_not_admitted():
    """This tier's whole safety argument is that a shaky item is dangerous
    BECAUSE it can still mark a goal covered. That stopped applying to
    half the admitted band: at the shipped 0.6/0.5 pair a claim scores
    `min(0.6, 0.6*conf + 0.05)` against a strict `>`, so everything from
    confidence 0.50 to 0.75 was retrieved, prompted and made citable while
    being unable to converge anything."""
    for confidence in (0.50, 0.60, 0.70, 0.75):
        assert _admit(confidence) == [], confidence


def test_a_claim_that_can_cover_is_still_admitted_and_keeps_its_score():
    from research_agent.tools.model_knowledge import score_for_confidence

    for confidence in (0.76, 0.85, 0.90, 1.0):
        evidence = _admit(confidence)
        assert len(evidence) == 1, confidence
        assert evidence[0].source == "model"
        assert evidence[0].score == score_for_confidence(confidence, 0.6)


def test_the_rule_follows_the_thresholds_instead_of_hardcoding_them():
    """Derived, not a second constant: lowering the coverage floor admits
    more, raising it admits less, with no edit to this module. Two
    constants drifting apart is exactly how the band went inert."""
    assert _admit(0.60, score=0.6, floor=0.5) == []      # 0.41, cannot cover
    assert len(_admit(0.60, score=0.6, floor=0.3)) == 1  # 0.41 > 0.3, covers
    assert _admit(0.90, score=0.6, floor=0.59) == []     # 0.59 is not > 0.59


def test_a_claim_the_model_disowns_is_dropped_whatever_the_floor_is():
    """The self-reported confidence floor survives the derived rule for
    the low-floor configuration: an item the model half-remembers is
    worse than no item, and that is not a function of anyone's
    threshold."""
    assert _admit(0.2, score=0.6, floor=0.01) == []


def test_dropped_claims_are_reported_not_silent(caplog):
    import logging

    with caplog.at_level(logging.INFO):
        _admit(0.60)

    events = [getattr(r, "event_fields", {}) for r in caplog.records
              if r.message == "tool.model_knowledge"]
    assert events and events[0]["dropped_inert"] == 1, events
    assert events[0]["claims"] == 0 and events[0]["asked"] == 1, (
        "asked-vs-kept is what makes a mismatched pair of thresholds "
        "visible per call")

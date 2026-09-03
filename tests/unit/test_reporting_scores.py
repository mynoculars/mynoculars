"""
tests/unit/test_reporting_scores.py -- S-17's de-duplication.

THE CLAIM BEING TESTED is not "these five scores are correct" -- they were
correct in both copies. It is that there is now ONE copy, and that the
guards which distinguish "absent" from "zero" survived the move. A score of
0.0 recorded for a run that never reached telemetry_node is worse than no
score at all: it is a measurement of something nobody measured.
"""

from research_agent.reporting import scores as scores_mod
from research_agent.reporting.scores import emit_run_scores


class _Spy:
    """Records (name, value, comment) instead of talking to Langfuse."""

    def __init__(self):
        self.calls = []

    def __call__(self, thread_id, name, value, comment=None):
        self.calls.append((name, value, comment))

    @property
    def names(self):
        return [n for n, _v, _c in self.calls]


def _spy(monkeypatch):
    spy = _Spy()
    monkeypatch.setattr(scores_mod.lf, "score", spy)
    return spy


def test_an_empty_telemetry_dict_scores_nothing(monkeypatch):
    """A run that ended without reaching telemetry_node -- a recursion
    limit, an aborted resume -- has nothing to report, and reporting zeros
    for it would put numbers in the record that nothing measured."""
    spy = _spy(monkeypatch)
    emit_run_scores("t1", {})
    assert spy.calls == []


def test_a_full_run_emits_all_five(monkeypatch):
    spy = _spy(monkeypatch)
    emit_run_scores("t1", {"recall": 0.8, "critique_passed": True,
                           "evidence_items": 10, "goals": 5,
                           "search_calls": 4, "memory_hits": 2,
                           "grounding_ratio": 0.6,
                           "goals_without_evidence": []})
    assert spy.names == ["recall", "critique_passed", "evidence_per_goal",
                         "memory_hit_rate", "grounding_ratio"]
    assert ("evidence_per_goal", 2.0, None) in spy.calls
    assert ("memory_hit_rate", 0.5, None) in spy.calls


def test_falsy_but_present_values_are_still_scored(monkeypatch):
    """`in telemetry`, not truthiness, for these two: recall 0.0 and a
    FAILED critique are real measurements and the most interesting ones in
    the record. Using truthiness here would silently drop exactly the runs
    worth looking at."""
    spy = _spy(monkeypatch)
    emit_run_scores("t1", {"recall": 0.0, "critique_passed": False})
    assert spy.calls == [("recall", 0.0, None),
                         ("critique_passed", False, None)]


def test_divisors_are_guarded_by_truthiness_not_presence(monkeypatch):
    """The other two ARE guarded by truthiness, because they divide by the
    value. goals=0 or search_calls=0 must not raise ZeroDivisionError in
    the last step of a run that already went wrong."""
    spy = _spy(monkeypatch)
    emit_run_scores("t1", {"evidence_items": 3, "goals": 0,
                           "search_calls": 0, "memory_hits": 0})
    assert spy.calls == []


def test_the_grounding_comment_names_the_unevidenced_goals(monkeypatch):
    """A low score in the Langfuse UI has to be actionable without opening
    the run's logs, which is what the comment is for."""
    spy = _spy(monkeypatch)
    emit_run_scores("t1", {"grounding_ratio": 0.5,
                           "goals_without_evidence": ["g2", "g4"]})
    assert spy.calls == [("grounding_ratio", 0.5, "unevidenced=g2,g4")]

    spy2 = _spy(monkeypatch)
    emit_run_scores("t1", {"grounding_ratio": 1.0,
                           "goals_without_evidence": []})
    assert spy2.calls == [("grounding_ratio", 1.0, "unevidenced=none")]


def test_both_interfaces_call_the_same_implementation():
    """The whole point of S-17. If either call site grows its own copy
    again, runs served by the two interfaces stop being comparable and
    nothing else would notice."""
    import research_agent.api.server as server
    import research_agent.cli as cli
    assert server.emit_run_scores is emit_run_scores
    assert cli.emit_run_scores is emit_run_scores

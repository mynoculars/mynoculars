"""
tests/unit/test_state.py — state.py's reducer functions.

Covers ONLY the three plain reducer functions LangGraph uses to merge
concurrent worker updates into shared state: merge_key_sets,
merge_failed_keys, merge_counters (D-5/D-16/D-19's concurrency-safety
math). Does NOT cover the Pydantic models themselves (Evidence, Goal,
SearchTask, etc.) — those have no dedicated behavior worth a unit test
beyond what Pydantic itself already guarantees; they're exercised
implicitly throughout the rest of this suite instead.
"""

from research_agent.state import merge_counters, merge_failed_keys, merge_key_sets


def test_key_set_union_is_order_independent():
    assert merge_key_sets({"a"}, {"b"}) == merge_key_sets({"b"}, {"a"}) == {"a", "b"}


def test_failed_keys_keep_deepest_failure():
    # D-16: conservative merge — never permit an earlier retry than any
    # worker observed.
    assert merge_failed_keys({"k": 1}, {"k": 3}) == {"k": 3}
    assert merge_failed_keys({"k": 3}, {"k": 1}) == {"k": 3}


def test_counters_merge_additively():
    assert merge_counters({"llm_calls": 2}, {"llm_calls": 1, "x": 1}) == {
        "llm_calls": 3, "x": 1}

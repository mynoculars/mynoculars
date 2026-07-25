"""
tests/unit/test_config.py — config.py's env-typo detection (P2-09).

Covers ONLY warn_on_likely_env_typos(). Does NOT cover Settings' own
field validation or defaults — those are exercised implicitly by every
other test file in this suite, each of which constructs a Settings
instance suited to what it's testing.
"""

import logging

from research_agent.config import warn_on_likely_env_typos


def test_warn_on_likely_env_typos_flags_known_mistakes(monkeypatch, caplog):
    monkeypatch.setenv("HITL", "true")          # should have been HITL_ENABLED
    monkeypatch.delenv("HITL_ENABLED", raising=False)
    with caplog.at_level(logging.WARNING):
        warn_on_likely_env_typos()
    matches = [r for r in caplog.records if "config.likely_typo" in r.message]
    assert matches
    assert matches[0].event_fields["set_key"] == "HITL"
    assert matches[0].event_fields["probably_meant"] == "HITL_ENABLED"


def test_warn_on_likely_env_typos_silent_when_correct_key_present(monkeypatch, caplog):
    monkeypatch.setenv("HITL", "true")
    monkeypatch.setenv("HITL_ENABLED", "true")  # correct key also set -> no warning
    with caplog.at_level(logging.WARNING):
        warn_on_likely_env_typos()
    assert not [r for r in caplog.records if "config.likely_typo" in r.message]

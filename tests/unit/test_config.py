"""
tests/unit/test_config.py — config.py's env-typo detection (P2-09).

Covers warn_on_likely_env_typos() (P2-09) and warn_on_web_search_band()
(Phase 4 / D-57). Does NOT cover Settings' own field validation or
defaults — those are exercised implicitly by every other test file in this
suite, each of which constructs a Settings instance suited to what it's
testing.
"""

import logging

from research_agent.config import (
    REPO_ROOT,
    Settings,
    warn_on_likely_env_typos,
    warn_on_web_search_band,
)


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


def test_warn_on_likely_env_typos_flags_the_new_web_search_names(monkeypatch, caplog):
    monkeypatch.setenv("WEB_SEARCH", "true")   # should have been WEB_SEARCH_ENABLED
    monkeypatch.delenv("WEB_SEARCH_ENABLED", raising=False)
    with caplog.at_level(logging.WARNING):
        warn_on_likely_env_typos()
    matches = [r for r in caplog.records if "config.likely_typo" in r.message]
    assert any(m.event_fields["probably_meant"] == "WEB_SEARCH_ENABLED"
               for m in matches)


def test_web_search_band_warns_when_the_tier_would_be_inert(caplog):
    """The failure this catches is completely silent otherwise: D-17's
    coverage predicate is a strict `>`, so with the floor at or below
    min_evidence_score the web tier fires, spends real network time, returns
    real evidence, and can never mark a goal covered. Same class of
    config-undoing-code defect as MIN_EVIDENCE_SCORE=0.0."""
    s = Settings(_env_file=None, min_evidence_score=0.6,
                 web_search_min_score=0.6, web_search_max_score=0.75)
    with caplog.at_level(logging.WARNING):
        warn_on_web_search_band(s)
    matches = [r for r in caplog.records
               if "config.web_search_tier_inert" in r.message]
    assert matches
    assert matches[0].event_fields["setting"] == "WEB_SEARCH_MIN_SCORE"


def test_web_search_band_is_silent_on_a_correctly_ordered_band(caplog):
    s = Settings(_env_file=None, min_evidence_score=0.5,
                 web_search_min_score=0.6, web_search_max_score=0.75)
    with caplog.at_level(logging.WARNING):
        warn_on_web_search_band(s)
    assert not [r for r in caplog.records if "config.web_search" in r.message]


def test_web_search_band_warns_when_floor_exceeds_ceiling(caplog):
    """Not a correctness bug -- rank_to_score normalizes an inverted band
    rather than running backwards -- but silently doing the right thing with
    the wrong config teaches nobody anything."""
    s = Settings(_env_file=None, min_evidence_score=0.5,
                 web_search_min_score=0.8, web_search_max_score=0.7)
    with caplog.at_level(logging.WARNING):
        warn_on_web_search_band(s)
    assert [r for r in caplog.records
            if "config.web_search_band_inverted" in r.message]


def test_max_task_retries_is_still_unread_by_anything():
    """Guard against this dead field quietly becoming load-bearing without a
    decision being taken. It survives as the fragment of a partially-applied
    patch, and config.py labels it CURRENTLY UNREAD. If someone wires it,
    this test fails and that label must be corrected in the same change."""
    import pathlib

    root = pathlib.Path(__file__).parent.parent.parent
    hits = []
    for path in (list((root / "src").rglob("*.py"))
                 + list((root / "scripts").rglob("*.py"))):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "max_task_retries" in text and path.name != "config.py":
            hits.append(str(path))
    assert not hits, f"max_task_retries is now read by: {hits}"


# ---------------------------------------------------------------------------
# D-58: MCP server path resolution
# ---------------------------------------------------------------------------


def test_repo_root_points_at_the_actual_repository():
    """Derived from config.py's own __file__, never from the CWD -- which is
    the entire point (see the constant's comment)."""
    assert (REPO_ROOT / "src" / "research_agent" / "config.py").exists()
    assert (REPO_ROOT / "scripts" / "mcp_web_search_server.py").exists()


# ---------------------------------------------------------------------------
# S-11 -- configure_logging() runs before the warn_on_* config checks
# ---------------------------------------------------------------------------


def test_get_settings_configures_logging_before_returning(monkeypatch):
    """S-11: configure_logging() must run inside get_settings(), after
    Settings() is built and before it returns -- so the three warn_on_*
    checks (which fire at WARNING) log against an already-configured root
    logger instead of risking that message being lost."""
    import research_agent.config as config_module

    calls = []
    monkeypatch.setattr(config_module, "configure_logging",
                        lambda level: calls.append(level))
    config_module.get_settings.cache_clear()
    try:
        settings = config_module.get_settings()
        assert calls == [settings.log_level], (
            "configure_logging must be called exactly once, with the "
            "constructed settings' own log_level")
    finally:
        config_module.get_settings.cache_clear()

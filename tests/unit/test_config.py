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
    resolve_repo_path,
    resolve_server_command,
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


def test_a_relative_path_resolves_against_the_repo_not_the_cwd(monkeypatch, tmp_path):
    """The bug this closes: MCPBridge never sets
    StdioServerParameters.cwd, so the server subprocess inherits whatever
    directory the agent was launched from. Verified by spawning real
    subprocesses -- relative args with cwd=/tmp fail as an opaque
    "Connection closed", never as a file-not-found."""
    monkeypatch.chdir(tmp_path)
    resolved = resolve_repo_path("scripts/mcp_web_search_server.py")
    assert resolved == str(REPO_ROOT / "scripts" / "mcp_web_search_server.py")


def test_windows_separators_resolve_on_any_platform():
    """The shipped .env.example uses backslashes. Without normalization,
    pathlib on POSIX reads "scripts\\x.py" as ONE filename containing a
    backslash, so the same .env would behave differently on the two
    platforms."""
    assert (resolve_repo_path(r"scripts\mcp_web_search_server.py")
            == str(REPO_ROOT / "scripts" / "mcp_web_search_server.py"))


def test_an_absolute_path_is_returned_unchanged():
    """Someone who wrote an absolute path meant it."""
    import sys

    assert resolve_repo_path(sys.executable) == sys.executable


def test_a_bare_command_name_is_not_turned_into_a_repo_path():
    """THE reason resolve_repo_path checks existence rather than blindly
    prefixing REPO_ROOT. "python3"/"uvx"/"npx" are names the OS resolves
    through PATH, not paths -- prefixing would make every one of them a
    guaranteed FileNotFoundError."""
    assert resolve_repo_path("python3") == "python3"
    assert resolve_repo_path("uvx") == "uvx"


def test_a_relative_path_that_resolves_nowhere_is_left_alone():
    """Not this function's job to raise: MCPBridge already errors clearly
    naming the command, and check_services.py exists to surface exactly this
    before a real run. Raising here would turn a config mistake into an
    import-time crash of the whole application."""
    assert resolve_repo_path("nope/missing.py") == "nope/missing.py"


def test_an_empty_path_stays_empty():
    assert resolve_repo_path("") == ""


def test_an_empty_command_means_the_agents_own_interpreter():
    """The RECOMMENDED configuration, not a fallback for the careless: it is
    correct on every machine with no configuration at all, and it guarantees
    the server runs in the SAME virtualenv -- which matters because
    mcp_web_search_server.py imports ddgs, and a wrong interpreter dies
    before the MCP handshake and surfaces as "Connection closed" rather than
    as a readable ImportError."""
    import sys

    assert resolve_server_command("") == sys.executable
    assert resolve_server_command("   ") == sys.executable


def test_an_explicit_command_is_honoured_over_sys_executable():
    """A real case, just not the common one: the server genuinely needing a
    DIFFERENT interpreter than the agent."""
    assert resolve_server_command("python3") == "python3"


def test_a_repo_relative_venv_interpreter_resolves(tmp_path, monkeypatch):
    """The portable middle option: ".venv/Scripts/python.exe" committed to
    .env works on any clone, on any drive, from any working directory --
    provided the venv is inside the repo."""
    fake_venv = REPO_ROOT / ".venv-test-marker"
    fake_venv.mkdir(exist_ok=True)
    try:
        monkeypatch.chdir(tmp_path)
        assert (resolve_server_command(".venv-test-marker")
                == str(REPO_ROOT / ".venv-test-marker"))
    finally:
        fake_venv.rmdir()

"""
tests/unit/test_config.py — config.py's env-typo detection (P2-09, D-79).

Covers warn_on_likely_env_typos() (P2-09; D-79 extended it to also see
.env-FILE-only keys, not just os.environ, and to cover the six settings
D-76 removed outright) and warn_on_web_search_band() (Phase 4 / D-57).
Does NOT cover Settings' own field validation or defaults — those are
exercised implicitly by every other test file in this suite, each of
which constructs a Settings instance suited to what it's testing.

D-84 -- WHY EVERY TYPO TEST BELOW TAKES THE `isolated_from_dotenv`
FIXTURE, AND WHY OMITTING IT IS A SILENT FAILURE:

warn_on_likely_env_typos() reads the union of os.environ AND the .env
file in the CURRENT WORKING DIRECTORY (D-79 -- deliberately, since
editing .env is how nearly everyone configures this project). Its rule
is "warn when the WRONG key is set and the RIGHT one is not". So any
test that sets a typo in os.environ and asserts a warning is silently
dependent on the developer's own .env: if that file happens to define
the correct key -- which a working checkout's .env virtually always
does -- the warning correctly does not fire and the test fails through
no fault of the code.

The tests D-79 shipped alongside its own change already knew this and
chdir into tmp_path. The OLDER P2-09-era tests predate the .env read
entirely (when only os.environ was consulted, the CWD was irrelevant)
and were never updated, so D-79 broke them the moment it landed --
10 failures on any checkout with a populated .env, reproduced on two
independent machines. The fixture makes the isolation shared and
explicit rather than something each new test has to remember.
"""

import logging

import pytest
from pydantic import ValidationError

from research_agent.config import (
    REPO_ROOT,
    Settings,
    warn_on_likely_env_typos,
    warn_on_web_search_band,
)


@pytest.fixture
def isolated_from_dotenv(tmp_path, monkeypatch):
    """Run the test in an empty directory, so warn_on_likely_env_typos'
    .env half (D-79) sees nothing and only os.environ -- which the test
    itself controls via monkeypatch -- decides the outcome.

    Yields the temp directory, so a test that wants to exercise the .env
    half deliberately can write its own file into it.

    monkeypatch.chdir is undone at teardown like every other monkeypatch
    change, so this cannot leak a changed CWD into any later test.
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_warn_on_likely_env_typos_flags_known_mistakes(
        isolated_from_dotenv, monkeypatch, caplog):
    monkeypatch.setenv("HITL", "true")          # should have been HITL_ENABLED
    monkeypatch.delenv("HITL_ENABLED", raising=False)
    with caplog.at_level(logging.WARNING):
        warn_on_likely_env_typos()
    matches = [r for r in caplog.records if "config.likely_typo" in r.message]
    assert matches
    assert matches[0].event_fields["set_key"] == "HITL"
    assert matches[0].event_fields["probably_meant"] == "HITL_ENABLED"


def test_warn_on_likely_env_typos_silent_when_correct_key_present(
        isolated_from_dotenv, monkeypatch, caplog):
    monkeypatch.setenv("HITL", "true")
    monkeypatch.setenv("HITL_ENABLED", "true")  # correct key also set -> no warning
    with caplog.at_level(logging.WARNING):
        warn_on_likely_env_typos()
    assert not [r for r in caplog.records if "config.likely_typo" in r.message]


def test_warn_on_likely_env_typos_flags_the_new_web_search_names(
        isolated_from_dotenv, monkeypatch, caplog):
    monkeypatch.setenv("WEB_SEARCH", "true")   # should have been WEB_SEARCH_ENABLED
    monkeypatch.delenv("WEB_SEARCH_ENABLED", raising=False)
    with caplog.at_level(logging.WARNING):
        warn_on_likely_env_typos()
    matches = [r for r in caplog.records if "config.likely_typo" in r.message]
    assert any(m.event_fields["probably_meant"] == "WEB_SEARCH_ENABLED"
               for m in matches)


def test_a_correct_key_in_dotenv_silences_a_typo_exported_in_the_shell(
        isolated_from_dotenv, monkeypatch, caplog):
    """D-84: pin the cross-source union semantics that broke the tests
    above, so the interaction is a specification rather than an accident
    nobody wrote down.

    The two sources are checked as ONE set: a typo exported in the shell
    is NOT warned about when the correct key is present in .env. That is
    the right behaviour -- the setting the operator actually needs IS
    configured, so the stray key is redundant rather than harmful, and
    warning about it would be noise. It is also exactly why the tests
    above must run in an empty directory: a real checkout's .env supplies
    the correct key for nearly every entry in _KNOWN_ENV_TYPOS."""
    monkeypatch.setenv("HITL", "true")            # the typo, in the shell
    monkeypatch.delenv("HITL_ENABLED", raising=False)
    (isolated_from_dotenv / ".env").write_text(   # the correct key, in .env
        "HITL_ENABLED=false\n")
    with caplog.at_level(logging.WARNING):
        warn_on_likely_env_typos()
    assert not [r for r in caplog.records if "config.likely_typo" in r.message], (
        "the correct key is configured, in .env -- the redundant typo is "
        "not worth a warning")


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


def test_max_task_retries_no_longer_exists_at_all():
    """D-82: the decision this dead field's predecessor test demanded
    ("either wire it or drop it -- but not silently") has now been taken,
    and it was DROP. D-16's depth-scoped retry gate in
    agents/task_utils.py::cap_and_filter is the one retry policy this
    architecture has.

    Kept as an assertion rather than deleted outright so that
    reintroducing the field -- as a config key, a Settings attribute, or a
    reader anywhere in src/ or scripts/ -- fails loudly and forces D-82 to
    be revisited on purpose instead of by accident."""
    import pathlib

    assert not hasattr(Settings(_env_file=None), "max_task_retries")

    root = pathlib.Path(__file__).parent.parent.parent
    hits = [str(path)
            for path in (list((root / "src").rglob("*.py"))
                         + list((root / "scripts").rglob("*.py")))
            if "max_task_retries" in path.read_text(encoding="utf-8",
                                                    errors="ignore")]
    # config.py is allowed exactly one mention: D-82's own explanatory
    # comment, which says why the field is gone. Any OTHER file mentioning
    # it means something is reading a setting that no longer exists.
    assert hits == [str(root / "src" / "research_agent" / "config.py")], (
        f"max_task_retries reappeared in: {hits}")


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


# ---------------------------------------------------------------------------
# D-79 -- warn_on_likely_env_typos must also see .env-FILE-only keys, and
# must know about the six settings D-76 removed outright
# ---------------------------------------------------------------------------


def test_env_file_keys_reads_the_actual_dotenv_file(tmp_path, monkeypatch):
    """The regression this whole fix exists to close, proven directly:
    a key set ONLY in .env (never exported to the real shell) must be
    visible to _env_file_keys() -- confirmed empirically (constructing a
    real Settings from a temp .env and checking os.environ immediately
    after) that pydantic-settings' own env_file parsing never writes
    those values into os.environ at all."""
    from research_agent.config import _env_file_keys

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("WEB_MCP_SERVER_COMMAND=/some/path\n"
                                   "WEB_SEARCH_ENABLED=true\n")
    keys = _env_file_keys()
    assert "WEB_MCP_SERVER_COMMAND" in keys
    assert "WEB_SEARCH_ENABLED" in keys


def test_env_file_keys_returns_empty_set_for_a_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no .env in this directory
    from research_agent.config import _env_file_keys
    assert _env_file_keys() == set()


def test_warn_on_likely_env_typos_catches_a_dotenv_only_mistake(
        tmp_path, monkeypatch, caplog):
    """The exact live scenario, reproduced directly: WEB_MCP_SERVER_COMMAND
    set ONLY in .env (never exported), WEB_MCP_SERVER_URL never set at
    all. Before D-79 this was invisible to warn_on_likely_env_typos
    entirely -- it only ever checked os.environ, which a .env-only value
    never reaches."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WEB_MCP_SERVER_COMMAND", raising=False)
    monkeypatch.delenv("WEB_MCP_SERVER_URL", raising=False)
    (tmp_path / ".env").write_text(
        "WEB_SEARCH_ENABLED=true\n"
        "WEB_MCP_SERVER_COMMAND=D:\\work\\repo\\.venv\\Scripts\\python.exe\n")
    with caplog.at_level(logging.WARNING):
        warn_on_likely_env_typos()
    matches = [r for r in caplog.records if "config.likely_typo" in r.message]
    assert matches
    assert matches[0].event_fields["set_key"] == "WEB_MCP_SERVER_COMMAND"
    assert matches[0].event_fields["probably_meant"] == "WEB_MCP_SERVER_URL"


@pytest.mark.parametrize("wrong,right", [
    ("MCP_TRANSPORT", "MCP_SERVER_URL"),
    ("MCP_SERVER_COMMAND", "MCP_SERVER_URL"),
    ("MCP_SERVER_ARGS", "MCP_SERVER_URL"),
    ("MCP_SERVER_ENV_ALLOWLIST", "MCP_SERVER_URL"),
    ("WEB_MCP_TRANSPORT", "WEB_MCP_SERVER_URL"),
    ("WEB_MCP_SERVER_COMMAND", "WEB_MCP_SERVER_URL"),
    ("WEB_MCP_SERVER_ARGS", "WEB_MCP_SERVER_URL"),
    ("WEB_MCP_SERVER_ENV_ALLOWLIST", "WEB_MCP_SERVER_URL"),
])
def test_warn_on_likely_env_typos_covers_every_d76_removed_setting(
        wrong, right, isolated_from_dotenv, monkeypatch, caplog):
    """All six settings D-76 deleted from Settings (plus the two
    already-removed args/allowlist siblings) must be recognized as
    "probably meant the new _URL setting" -- not just the one that
    happened to show up in a live run.

    D-84: takes isolated_from_dotenv for the reason given in this
    module's docstring -- a checkout whose .env defines MCP_SERVER_URL /
    WEB_MCP_SERVER_URL (i.e. any working one) satisfies the "right key
    not set" half of the rule and correctly suppresses every warning
    this test asserts."""
    monkeypatch.setenv(wrong, "some-old-value")
    monkeypatch.delenv(right, raising=False)
    with caplog.at_level(logging.WARNING):
        warn_on_likely_env_typos()
    matches = [r for r in caplog.records if "config.likely_typo" in r.message
              and r.event_fields.get("set_key") == wrong]
    assert matches, f"{wrong} was not flagged"
    assert matches[0].event_fields["probably_meant"] == right


# ---------------------------------------------------------------------------
# D-114 -- pricing follows the configured fallback name
# ---------------------------------------------------------------------------


def test_grok_has_a_pricing_row_like_every_other_provider():
    from research_agent.langfuse.pricing import (_PROVIDER_RATE_FIELDS,
                                                 TokenUsage, calculate_cost)
    assert "grok" in _PROVIDER_RATE_FIELDS

    s = Settings(_env_file=None, langfuse_price_grok_in_per_1m=0.20,
                 langfuse_price_grok_out_per_1m=0.50)
    cost = calculate_cost(s, "grok", TokenUsage(1_000_000, 1_000_000))

    assert cost is not None and round(cost.total_usd, 4) == 0.70


def test_an_unpriced_fallback_provider_warns_at_startup(caplog):
    """Same shape as the two inert-setting warnings: a thing that silently
    does nothing is worse than a thing that fails."""
    import logging
    from research_agent.config import warn_on_unpriced_fallback

    s = Settings(_env_file=None, llm_fallback_name="typoed",
                 llm_fallback_api_key="k")
    with caplog.at_level(logging.WARNING):
        warn_on_unpriced_fallback(s)

    rec = [r for r in caplog.records
           if "config.fallback_provider_unpriced" in r.message]
    assert rec and rec[0].event_fields["value"] == "typoed"


def test_a_known_provider_does_not_warn(caplog):
    import logging
    from research_agent.config import warn_on_unpriced_fallback

    for name in ("gemini", "grok"):
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            warn_on_unpriced_fallback(
                Settings(_env_file=None, llm_fallback_name=name,
                         llm_fallback_api_key="k"))
        assert not [r for r in caplog.records
                    if "config.fallback_provider_unpriced" in r.message]


def test_a_keyless_provider_never_warns_however_it_is_named(caplog):
    """With no key the provider is not in the chain at all, so its price
    is irrelevant and a warning would be noise."""
    import logging
    from research_agent.config import warn_on_unpriced_fallback

    with caplog.at_level(logging.WARNING):
        warn_on_unpriced_fallback(
            Settings(_env_file=None, llm_fallback_name="typoed",
                     llm_fallback_api_key=""))

    assert not [r for r in caplog.records
                if "config.fallback_provider_unpriced" in r.message]


# ---------------------------------------------------------------------------
# D-131 (P6-2) -- the evidence budget, and the warning when it is switched off
# ---------------------------------------------------------------------------


def test_the_prompt_evidence_budget_ships_bounded_by_default():
    """An observability or safety feature that ships INERT is the mistake
    MIN_EVIDENCE_SCORE=0.0 was. The default has to actually bound."""
    assert Settings(_env_file=None).prompt_evidence_max_chars == 12000


def test_a_zero_prompt_budget_warns_at_startup(caplog):
    from research_agent.config import warn_on_unbounded_prompt_budget

    s = Settings(_env_file=None, prompt_evidence_max_chars=0)
    with caplog.at_level(logging.WARNING):
        warn_on_unbounded_prompt_budget(s)

    matches = [r for r in caplog.records
               if "config.prompt_budget_unbounded" in r.message]
    assert matches
    assert matches[0].event_fields["setting"] == "PROMPT_EVIDENCE_MAX_CHARS"
    assert "30,199" in matches[0].event_fields["effect"], (
        "the warning cites the measured run, not a hypothetical")


def test_a_configured_prompt_budget_is_silent(caplog):
    from research_agent.config import warn_on_unbounded_prompt_budget

    with caplog.at_level(logging.WARNING):
        warn_on_unbounded_prompt_budget(Settings(_env_file=None))

    assert not [r for r in caplog.records
                if "config.prompt_budget" in r.message]


# ---------------------------------------------------------------------------
# D-132 (P6-4) -- the run budget ships OFF
# ---------------------------------------------------------------------------


def test_both_run_budgets_ship_disabled():
    """This is the first setting in this codebase that can END a run
    early, so it is opt-in like HITL_ENABLED and MCP_ENABLED -- not
    on-by-default like the guardrail thresholds, which only ever measure
    or annotate."""
    s = Settings(_env_file=None)
    assert s.run_deadline_seconds == 0.0
    assert s.run_token_budget == 0


def test_a_mistyped_deadline_is_flagged_like_every_other_known_typo(
        isolated_from_dotenv, monkeypatch, caplog):
    """D-84's fixture is not optional here -- see this file's own header
    for why a typo test without it passes or fails on the developer's
    .env rather than on the code."""
    monkeypatch.setenv("RUN_DEADLINE", "600")
    monkeypatch.delenv("RUN_DEADLINE_SECONDS", raising=False)

    with caplog.at_level(logging.WARNING):
        warn_on_likely_env_typos()

    hits = [r for r in caplog.records if "config.likely_typo" in r.message]
    assert any(r.event_fields["probably_meant"] == "RUN_DEADLINE_SECONDS"
               for r in hits)


# ---------------------------------------------------------------------------
# D-143 -- the primary's context window vs the evidence budget
#
# Two settings decided whether the local model could ever serve a compile
# or a critique, and nothing compared them. Live (p205.280-check):
# LLM_PRIMARY_CONTEXT_TOKENS=1536 against PROMPT_EVIDENCE_MAX_CHARS=12000,
# llm_context_skips 2, and the shipped report came from a fallback hop the
# primary was never allowed to attempt.
# ---------------------------------------------------------------------------


def test_the_p205_280_configuration_warns(caplog):
    from research_agent.config import warn_on_context_below_prompt_budget

    s = Settings(_env_file=None, llm_primary_context_tokens=1536,
                 prompt_evidence_max_chars=12000)
    with caplog.at_level(logging.WARNING):
        warn_on_context_below_prompt_budget(s)

    matches = [r for r in caplog.records
               if "config.context_below_prompt_budget" in r.message]
    assert matches
    fields = matches[0].event_fields
    assert fields["setting"] == "LLM_PRIMARY_CONTEXT_TOKENS"
    assert fields["value"] == 1536
    assert fields["evidence_tokens_estimate"] == 3000
    assert "compile" in fields["effect"] and "critique" in fields["effect"]


def test_an_unconfigured_window_is_silent(caplog):
    """0 means "not configured" and makes D-93 inert -- there is nothing to
    compare, and warning would fire for every default install."""
    from research_agent.config import warn_on_context_below_prompt_budget

    s = Settings(_env_file=None, llm_primary_context_tokens=0)
    with caplog.at_level(logging.WARNING):
        warn_on_context_below_prompt_budget(s)

    assert not [r for r in caplog.records
                if "config.context_below" in r.message]


def test_a_window_that_actually_fits_the_budget_is_silent(caplog):
    from research_agent.config import warn_on_context_below_prompt_budget

    s = Settings(_env_file=None, llm_primary_context_tokens=8192,
                 prompt_evidence_max_chars=12000)
    with caplog.at_level(logging.WARNING):
        warn_on_context_below_prompt_budget(s)

    assert not [r for r in caplog.records
                if "config.context_below" in r.message]


def test_lowering_the_evidence_budget_is_the_other_way_to_satisfy_it(caplog):
    """The warning names two remedies and both must actually silence it,
    or it is telling people to do something that does not work."""
    from research_agent.config import warn_on_context_below_prompt_budget

    s = Settings(_env_file=None, llm_primary_context_tokens=1536,
                 prompt_evidence_max_chars=4000)
    with caplog.at_level(logging.WARNING):
        warn_on_context_below_prompt_budget(s)

    assert not [r for r in caplog.records
                if "config.context_below" in r.message]


def test_the_default_install_does_not_warn(caplog):
    """.env.example ships LLM_PRIMARY_CONTEXT_TOKENS=0, so a clean clone
    must be quiet here."""
    from research_agent.config import warn_on_context_below_prompt_budget

    with caplog.at_level(logging.WARNING):
        warn_on_context_below_prompt_budget(Settings(_env_file=None))

    assert not [r for r in caplog.records
                if "config.context_below" in r.message]



# ---------------------------------------------------------------------------
# D-148 -- a numeric setting may carry a thousands separator
#
# LLM_PRIMARY_CONTEXT_TOKENS=8,876 -- a model's real context window, written
# the way a person writes a number -- made Settings() raise. Because 43
# tests reach Settings() through get_settings(), that ONE line reported as
# "26 failed, 1087 passed, 17 errors", none of it near the field in
# question. Reproduced exactly from the developer's own .env.
# ---------------------------------------------------------------------------


def test_the_exact_value_that_broke_the_suite_now_parses():
    from research_agent.config import Settings

    s = Settings(_env_file=None, llm_primary_context_tokens="8,876")

    assert s.llm_primary_context_tokens == 8876
    assert isinstance(s.llm_primary_context_tokens, int)


def test_a_grouped_float_parses_too():
    from research_agent.config import Settings

    s = Settings(_env_file=None, decay_half_life_days_semi_stable="1,095")

    assert s.decay_half_life_days_semi_stable == 1095.0


def test_quotes_and_spaces_are_stripped_from_a_number():
    """A .env value copied out of documentation picks these up."""
    from research_agent.config import Settings

    assert Settings(_env_file=None,
                    llm_primary_context_tokens=' "8 192" ').llm_primary_context_tokens == 8192


def test_a_clean_value_is_untouched():
    from research_agent.config import _normalise_numeric

    for raw in ("8192", "0.6", "-1", "1e3", ""):
        assert _normalise_numeric(raw) == raw


def test_something_that_is_not_a_number_is_left_for_pydantic_to_reject():
    """This widens what is accepted; it must never guess. "1,2,3" is not a
    number anyone meant, and neither is "abc"."""
    from research_agent.config import Settings, _normalise_numeric

    assert _normalise_numeric("1,2,3") == "1,2,3", "invalid grouping is not a number"
    assert _normalise_numeric("abc") == "abc"
    assert _normalise_numeric("1,23") == "1,23"
    assert _normalise_numeric("8,8765") == "8,8765"
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_primary_context_tokens="abc")


def test_non_numeric_fields_are_never_touched():
    """A comma in a string setting is content, not grouping."""
    from research_agent.config import Settings

    s = Settings(_env_file=None, llm_primary_model="qwen,cogito")

    assert s.llm_primary_model == "qwen,cogito"


def test_the_normalisation_is_reported_not_silent(caplog):
    """Accepting the value is right; rewriting a person's configuration
    without saying so is not."""
    from research_agent.config import (Settings, _NUMERIC_NORMALISATIONS,
                                       warn_on_normalised_numerics)

    _NUMERIC_NORMALISATIONS.clear()
    Settings(_env_file=None, llm_primary_context_tokens="8,876")
    with caplog.at_level(logging.WARNING):
        warn_on_normalised_numerics()

    matches = [r for r in caplog.records
               if "config.numeric_normalised" in r.message]
    assert matches
    fields = matches[0].event_fields
    assert fields["setting"] == "LLM_PRIMARY_CONTEXT_TOKENS"
    assert fields["raw"] == "8,876"
    assert fields["parsed"] == "8876"


def test_the_report_drains_so_it_cannot_repeat(caplog):
    from research_agent.config import (Settings, _NUMERIC_NORMALISATIONS,
                                       warn_on_normalised_numerics)

    _NUMERIC_NORMALISATIONS.clear()
    Settings(_env_file=None, llm_primary_context_tokens="8,876")
    warn_on_normalised_numerics()
    # caplog.records accumulates for the WHOLE test, so the first call's
    # record is still there and would look like a repeat.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        warn_on_normalised_numerics()

    assert not [r for r in caplog.records
                if "config.numeric_normalised" in r.message]


def test_a_clean_configuration_reports_nothing(caplog):
    from research_agent.config import (Settings, _NUMERIC_NORMALISATIONS,
                                       warn_on_normalised_numerics)

    _NUMERIC_NORMALISATIONS.clear()
    Settings(_env_file=None, llm_primary_context_tokens=8192)
    with caplog.at_level(logging.WARNING):
        warn_on_normalised_numerics()

    assert not [r for r in caplog.records
                if "config.numeric_normalised" in r.message]


def test_a_real_dotenv_carrying_the_bad_value_still_loads(tmp_path, monkeypatch):
    """The REAL-RUN path, not the test path: a .env file on disk with the
    grouped value must produce a working Settings. conftest's isolation
    stops the suite reading .env at all, so this test supplies its own."""
    from research_agent.config import Settings

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("LLM_PRIMARY_CONTEXT_TOKENS=8,876\n",
                                   encoding="utf-8")

    s = Settings(_env_file=str(tmp_path / ".env"))

    assert s.llm_primary_context_tokens == 8876


# ---------------------------------------------------------------------------
# D-147 -- the suite reads no ambient configuration at all
#
# These assert the contract of tests/conftest.py::_no_ambient_config, which
# is autouse and session-scoped. Without it, one malformed line in a
# developer's .env fails 43 tests that do not test configuration.
# ---------------------------------------------------------------------------


def test_no_dotenv_file_is_read_during_the_suite():
    from research_agent.config import Settings

    assert Settings.model_config["env_file"] is None


def test_no_settings_field_is_visible_as_an_os_environment_variable():
    """Settings(_env_file=None) does NOT insulate against these --
    pydantic-settings always checks os.environ first. A developer who ran
    `$env:HITL_ENABLED = "true"` earlier in the same shell would otherwise
    change what this suite asserts."""
    import os

    from research_agent.config import Settings

    leaked = {name.upper() for name in Settings.model_fields} & set(os.environ)

    assert not leaked, f"ambient configuration reached the suite: {sorted(leaked)}"


def test_a_test_can_still_set_a_value_deliberately(monkeypatch):
    """The fixture removes AMBIENT configuration. It must not stop a test
    from asking for a value."""
    import os

    from research_agent.config import Settings

    monkeypatch.setenv("HITL_ENABLED", "true")

    assert Settings().hitl_enabled is True
    assert os.environ["HITL_ENABLED"] == "true"



# ---------------------------------------------------------------------------
# D-153 -- a context window per provider slot
#
# D-93 shipped primary-only and said so: cloud fallbacks "have context
# windows orders of magnitude larger ... so giving them a knob would be
# configuration nobody needs". True of the providers it was written for;
# not true of the SLOT, since D-114 lets the third one point at any
# OpenAI-compatible endpoint, including a second local llama-server.
# ---------------------------------------------------------------------------


def test_all_three_slots_ship_unconfigured():
    """0 means "not configured". An existing .env must behave exactly as it
    did before these fields existed."""
    s = Settings(_env_file=None)

    assert s.llm_primary_context_tokens == 0
    assert s.llm_mistral_context_tokens == 0
    assert s.llm_fallback_context_tokens == 0


def test_the_new_settings_read_their_own_env_names(monkeypatch):
    monkeypatch.setenv("LLM_MISTRAL_CONTEXT_TOKENS", "32768")
    monkeypatch.setenv("LLM_FALLBACK_CONTEXT_TOKENS", "8192")

    s = Settings()

    assert s.llm_mistral_context_tokens == 32768
    assert s.llm_fallback_context_tokens == 8192
    assert s.llm_primary_context_tokens == 0, "the slots are independent"


def test_the_slot_names_match_their_siblings():
    """LLM_FALLBACK_*, not LLM_GEMINI_*: D-114 renames that slot by
    configuration, and every other setting for it is slot-based."""
    names = set(Settings.model_fields)

    assert "llm_fallback_context_tokens" in names
    assert "llm_gemini_context_tokens" not in names
    for sibling in ("llm_fallback_base_url", "llm_fallback_model",
                    "llm_fallback_api_key"):
        assert sibling in names


def test_a_grouped_number_works_for_the_new_slots_too():
    """D-148 applies to every int field, not just the one that exposed it."""
    s = Settings(_env_file=None, llm_mistral_context_tokens="32,768")

    assert s.llm_mistral_context_tokens == 32768


# --- the D-143 check, now covering all three -------------------------------


def test_a_misconfigured_mistral_slot_is_no_longer_silent(caplog):
    """The reason this generalisation is not cosmetic: a check that covers
    one of three configurable things is worse than none, because it reads
    as coverage."""
    from research_agent.config import warn_on_context_below_prompt_budget

    s = Settings(_env_file=None, llm_mistral_context_tokens=1536,
                 prompt_evidence_max_chars=12000)
    with caplog.at_level(logging.WARNING):
        warn_on_context_below_prompt_budget(s)

    matches = [r for r in caplog.records
               if "config.context_below_prompt_budget" in r.message]
    assert matches
    assert matches[0].event_fields["setting"] == "LLM_MISTRAL_CONTEXT_TOKENS"


def test_a_misconfigured_fallback_slot_is_caught_too(caplog):
    from research_agent.config import warn_on_context_below_prompt_budget

    s = Settings(_env_file=None, llm_fallback_context_tokens=2048,
                 prompt_evidence_max_chars=12000)
    with caplog.at_level(logging.WARNING):
        warn_on_context_below_prompt_budget(s)

    settings_named = [r.event_fields["setting"] for r in caplog.records
                      if "config.context_below_prompt_budget" in r.message]
    assert settings_named == ["LLM_FALLBACK_CONTEXT_TOKENS"]


def test_every_failing_slot_is_reported_not_just_the_first(caplog):
    """Two misconfigured providers are two things to fix. Stopping at the
    first hides the second until the first is fixed."""
    from research_agent.config import warn_on_context_below_prompt_budget

    s = Settings(_env_file=None, llm_primary_context_tokens=1536,
                 llm_mistral_context_tokens=1024,
                 llm_fallback_context_tokens=512,
                 prompt_evidence_max_chars=12000)
    with caplog.at_level(logging.WARNING):
        warn_on_context_below_prompt_budget(s)

    named = [r.event_fields["setting"] for r in caplog.records
             if "config.context_below_prompt_budget" in r.message]
    assert named == ["LLM_PRIMARY_CONTEXT_TOKENS",
                     "LLM_MISTRAL_CONTEXT_TOKENS",
                     "LLM_FALLBACK_CONTEXT_TOKENS"]


def test_a_generous_cloud_window_stays_silent(caplog):
    """The shipped defaults: Mistral Small is 32k, Gemini Flash far more,
    and a compile prompt is a few thousand tokens."""
    from research_agent.config import warn_on_context_below_prompt_budget

    s = Settings(_env_file=None, llm_mistral_context_tokens=32768,
                 llm_fallback_context_tokens=1048576,
                 prompt_evidence_max_chars=12000)
    with caplog.at_level(logging.WARNING):
        warn_on_context_below_prompt_budget(s)

    assert not [r for r in caplog.records
                if "config.context_below" in r.message]



# ---------------------------------------------------------------------------
# D-162 -- the two decay half-lives are divisors and must be bounded
# ---------------------------------------------------------------------------


def test_a_zero_or_negative_half_life_is_rejected_at_startup():
    """These were the only numerics in config.py with no validator, and
    they are DIVISORS: decay_factor computes exp(-ln2 * age / half_life).

    0 is a plausible guess for "turn decay off" and used to raise
    ZeroDivisionError at memory_retrieve_node -- the SECOND node of every
    run, where nothing catches it and SemanticMemory's contract says an
    unusable memory degrades to []. A negative value was worse because it
    was silent: the exponent's sign flips, decay becomes GROWTH, and the
    oldest memories rank highest (measured at 70,515,084x for a 365-day
    item against a half-life of -14)."""
    import pytest
    from pydantic import ValidationError

    for field in ("decay_half_life_days_semi_stable",
                  "decay_half_life_days_volatile"):
        for bad in (0.0, -14.0):
            with pytest.raises(ValidationError):
                Settings(_env_file=None, **{field: bad})

    # And the shipped values are still accepted.
    settings = Settings(_env_file=None)
    assert settings.decay_half_life_days_semi_stable == 90.0
    assert settings.decay_half_life_days_volatile == 14.0

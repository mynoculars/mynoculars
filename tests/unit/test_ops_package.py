"""
tests/unit/test_ops_package.py -- research_agent/ops/package.py (D-190).

Everything about the packaging step that can be wrong WITHOUT writing a
zip lives in three pure functions -- `is_excluded` (what travels),
`scan_text` (what must not) and `verify_archive` (what actually landed)
-- so this file exercises those directly, plus one real build into a
temp directory to prove the three agree.

TWO OF THESE TESTS EXIST BECAUSE THEY CAUGHT A DEFECT IN THE CHECK
ITSELF, which is worth saying plainly: a first version of `scan_text`
treated any identifier-shaped value as a variable reference, and an API
key IS identifier-shaped, so a planted 32-character Mistral-shaped key
packaged cleanly. Testing a guard only against the tree it was written
on proves nothing; it has to be tested against the thing it is supposed
to stop.

AND THEN THE FIXTURE ITSELF BECAME THE DEFECT. That planted key was
realistic enough that GitHub's push protection refused the commit
carrying this file, naming five of its lines. See FAKE_KEY below: the
fixture is now assembled at runtime and shaped so no provider regex
matches it, while keeping every property `scan_text` actually tests.
"""

import zipfile

import pytest

from research_agent.ops import package

# A credential-SHAPED value that is not shaped like any real provider's
# key, ASSEMBLED rather than written as a literal.
#
# WHY, and it is not paranoia: the first version of this file used a
# realistic 32-character alphanumeric Mistral-shaped literal, on the
# reasoning that a fixture built from obvious placeholders would
# exercise the placeholder path and prove nothing. That reasoning holds.
# What it missed is that a realistic key literal IS a real key to every
# scanner that reads this repository -- and GitHub's push protection
# rejected the commit that introduced it, naming five lines of this
# file, before any human saw it.
#
# So the fixture keeps the properties `scan_text` actually tests for --
# long enough (>=16 chars), digits present (so it is not a prose value),
# no dot and no snake_case underscore (so it is not read as a Python
# reference), no placeholder marker -- while carrying hyphens, which no
# provider key format uses and every provider regex therefore rejects.
# Joining the parts at runtime means no scannable literal exists in the
# source at all.
#
# The irony is the point, and it is worth leaving written down: the
# fixture was realistic enough to trip a production secret scanner,
# which is the strongest evidence available that ops/package.py's own
# check is looking for the right shape.
FAKE_KEY = "-".join(["Kd9Qm2Xv7Rn4", "Ts8Wp1Zb6Lc3", "Hy5Nf0Gj7"])


# ---------------------------------------------------------------------------
# is_excluded -- what travels and what does not
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [
    ".env",
    ".env.local",
    ".env.bak",
    "logs/run-p205.324-check.txt",
    "tmp/console-output.txt",
    "run.log",
    "src/research_agent/__pycache__/config.cpython-311.pyc",
    "src/research_agent/llm/__pycache__/router.cpython-311.pyc",
    "tests/__pycache__/conftest.cpython-311-pytest-9.1.1.pyc",
    ".pytest_cache/CACHEDIR.TAG",
    ".ruff_cache/0.15.11/1234",
    ".venv/Lib/site-packages/httpx/__init__.py",
    ".git/config",
    "build/lib/research_agent/cli.py",
    "dist/research_agent-0.3.0.tar.gz",
    "research_agent.egg-info/PKG-INFO",
    "certs/server.key",
    "certs/server.pem",
])
def test_forbidden_paths_never_travel(path):
    """Every one of these was in the archive that shipped, or is the
    same kind of thing. 197 of that archive's 423 entries were."""
    assert package.is_excluded(path) is True


@pytest.mark.parametrize("path", [
    ".env.example",
    "README.md",
    "OPERATIONS.md",
    "DECISIONS.md",
    "pyproject.toml",
    "src/research_agent/cli.py",
    "tests/unit/test_config.py",
    "sample_data/corpus.jsonl",
    "sample_data/golden_queries.jsonl",
    "design/Research_Agent_Design.md",
    "scripts/sanity.py",
    ".github/workflows/tests.yml",
])
def test_the_things_that_must_travel_do(path):
    assert package.is_excluded(path) is False


def test_env_example_survives_the_env_wildcard():
    """`.env.*` would take `.env.example` with it, and `.env.example` is
    the one file in that family a reviewer cannot do without -- it is
    the entire configuration surface. The negation is the reason
    KEEP_ANYWAY exists rather than a cleverer glob."""
    assert package.is_excluded(".env") is True
    assert package.is_excluded(".env.example") is False


def test_internal_is_deliberately_packaged():
    """`internal/` is IN the archive and OUT of git (README's document
    map), which is exactly why `git archive HEAD` cannot be the
    packaging mechanism and an explicit list applied to the working tree
    has to be."""
    assert package.is_excluded("internal/LEARNING_GUIDE.md") is False
    assert package.is_excluded("internal/PHASE9-SHOWCASE-READINESS.md") is False


def test_exclusion_matches_any_path_segment_not_just_the_name():
    """A pattern names a directory; the walk must reject everything
    beneath it at any depth, rather than needing a pattern that
    anticipates how deep the tree goes."""
    assert package.is_excluded("a/b/c/__pycache__/d.pyc") is True
    assert package.is_excluded("a/logs/b/c.txt") is True


# ---------------------------------------------------------------------------
# scan_text -- the credential backstop
# ---------------------------------------------------------------------------


def test_a_real_looking_api_key_is_caught():
    """THE REGRESSION THIS FILE EXISTS FOR. A first version of the check
    skipped any identifier-shaped value as "a variable reference" -- and
    a provider key is letters and digits with no punctuation, i.e.
    identifier-shaped. This exact planted line packaged cleanly."""
    findings = package.scan_text(
        f"LLM_MISTRAL_API_KEY={FAKE_KEY}\n", "OPS.md")
    assert len(findings) == 1
    number, name, _why = findings[0]
    assert number == 1 and name == "LLM_MISTRAL_API_KEY"


def test_a_finding_never_repeats_the_value():
    """A tool whose failure output prints the secret it is protecting
    has moved the problem, not solved it."""
    secret = FAKE_KEY
    findings = package.scan_text(f"API_KEY={secret}\n", "OPS.md")
    assert findings and all(secret not in str(part)
                            for finding in findings for part in finding)


@pytest.mark.parametrize("line", [
    "LLM_MISTRAL_API_KEY=your-mistral-key",
    "LANGFUSE_SECRET_KEY=",
    "API_KEY=pick-something-long-and-random",
    'password = "changeme-please"',
    "OPENSEARCH_PASSWORD=<your-password-here>",
])
def test_placeholders_are_not_findings(line):
    """A check that fails on this repo's own documentation is a check
    that gets overridden by habit."""
    assert package.scan_text(line + "\n", "OPS.md") == []


@pytest.mark.parametrize("line", [
    "        password=settings.opensearch_password,",
    "    secret_key = build_client(settings)",
    "    api_key = opensearch_password",
])
def test_passing_a_credential_BY_NAME_is_not_a_leak(line):
    """This is what good code looks like -- credentials referenced, not
    inlined -- so flagging it would penalise exactly the right pattern
    and bury the one real finding among dozens."""
    assert package.scan_text(line + "\n", "src/x.py") == []


def test_a_prose_value_is_not_a_credential():
    """`PROPRIETARY-CORPUS-SENTENCE` is a test fixture and
    `pick-something-long-and-random` is advice. Neither carries a digit;
    every provider key observed here does."""
    assert package.scan_text(
        'secret = "PROPRIETARY-CORPUS-SENTENCE"\n', "tests/x.py") == []


def test_a_dsn_with_a_real_password_is_caught():
    findings = package.scan_text(
        "POSTGRES_DSN=postgresql://agent:8Hq2ZrmVx4@db.internal:5432/prod\n",
        "OPS.md")
    assert any(name == "connection string" for _n, name, _w in findings)


def test_the_repos_own_sample_dsn_is_not_a_finding():
    """`agent:agent` -- password equal to username -- is a documented
    sample, and this repo ships it as its default DSN and prints it in
    OPERATIONS.md twice. Without this rule the check fails forever on
    its own documentation."""
    assert package.scan_text(
        "postgres_dsn: str = \"postgresql://agent:agent@localhost:5432/agent\"\n",
        "src/research_agent/config.py") == []
    assert package.scan_text(
        "Connection string format: `postgresql://user:pass@host:port/db`.\n",
        "internal/GLOSSARY.md") == []


def test_env_example_is_exempt_from_the_scan():
    """Its whole job is to SHOW these shapes."""
    line = f"LLM_MISTRAL_API_KEY={FAKE_KEY}\n"
    assert package.scan_text(line, ".env.example") == []
    assert package.scan_text(line, "OPS.md") != []


# ---------------------------------------------------------------------------
# The real thing: build, then verify the file that was written
# ---------------------------------------------------------------------------


def _tree(root):
    (root / "src").mkdir()
    (root / "src" / "__pycache__").mkdir()
    (root / "logs").mkdir()
    (root / "internal").mkdir()
    (root / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    (root / "src" / "__pycache__" / "app.pyc").write_text("junk", encoding="utf-8")
    (root / "logs" / "run-1.txt").write_text("trace", encoding="utf-8")
    (root / "internal" / "GUIDE.md").write_text("# guide\n", encoding="utf-8")
    (root / ".env").write_text(
        f"LLM_MISTRAL_API_KEY={FAKE_KEY}\n",
        encoding="utf-8")
    (root / ".env.example").write_text(
        "LLM_MISTRAL_API_KEY=your-mistral-key\n", encoding="utf-8")
    (root / "README.md").write_text("# readme\n", encoding="utf-8")


def test_a_built_archive_carries_exactly_what_the_policy_says(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _tree(root)
    out = tmp_path / "out.zip"
    included, findings = package.build(root, out, "repo")
    assert findings == []
    with zipfile.ZipFile(out) as archive:
        names = sorted(archive.namelist())
    assert names == sorted([
        "repo/.env.example", "repo/README.md",
        "repo/internal/GUIDE.md", "repo/src/app.py",
    ])
    # The .env was excluded by policy, so its planted key never reached
    # the scan -- which is the correct order: the exclude list is the
    # defence, the scan is the backstop behind it.
    assert "repo/.env" not in names


def test_verify_reads_the_ARTIFACT_not_the_list_that_produced_it(tmp_path):
    """Checking the include list would only prove the list agrees with
    itself. The claim being made to a reviewer is about the bytes on
    disk, so that is what gets checked."""
    out = tmp_path / "bad.zip"
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr("repo/README.md", "ok")
        archive.writestr("repo/.env", "LLM_MISTRAL_API_KEY=real")
        archive.writestr("repo/src/__pycache__/x.pyc", "junk")
    offenders = package.verify_archive(out)
    assert sorted(offenders) == ["repo/.env", "repo/src/__pycache__/x.pyc"]


def test_a_clean_archive_verifies_empty(tmp_path):
    out = tmp_path / "good.zip"
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr("repo/README.md", "ok")
        archive.writestr("repo/.env.example", "KEY=your-key-here")
    assert package.verify_archive(out) == []


def test_a_credential_in_an_INCLUDED_file_refuses_the_build(tmp_path):
    """The backstop's actual job: `.env` is excluded by name, but a key
    pasted into a runbook is not, and that file travels."""
    root = tmp_path / "repo"
    root.mkdir()
    _tree(root)
    (root / "RUNBOOK.md").write_text(
        f"Set LLM_MISTRAL_API_KEY={FAKE_KEY}\n",
        encoding="utf-8")
    out = tmp_path / "out.zip"
    _included, findings = package.build(root, out, "repo")
    assert findings and findings[0][0] == "RUNBOOK.md"
    assert not out.exists(), "a refused build must leave no archive behind"


def test_allow_suspect_packages_anyway_for_a_false_positive(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _tree(root)
    (root / "RUNBOOK.md").write_text(
        f"Set LLM_MISTRAL_API_KEY={FAKE_KEY}\n",
        encoding="utf-8")
    out = tmp_path / "out.zip"
    _included, findings = package.build(root, out, "repo",
                                        allow_suspect=True)
    assert findings                      # still REPORTED, just not fatal
    assert out.exists()


def test_a_refused_build_leaves_no_partial_file(tmp_path):
    """Nothing half-written that someone could pick up and send."""
    root = tmp_path / "repo"
    root.mkdir()
    _tree(root)
    (root / "RUNBOOK.md").write_text(
        f"API_KEY={FAKE_KEY}\n", encoding="utf-8")
    out = tmp_path / "out.zip"
    package.build(root, out, "repo")
    assert list(tmp_path.glob("*.partial")) == []

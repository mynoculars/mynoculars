"""
research_agent/langfuse/client.py — the ONLY file in this codebase that
imports the `langfuse` SDK package.

WHY THIS FILE EXISTS: Phase 3 requires "except for the actual
instrumentation calls made from existing files, all Langfuse SDK code
must live in its own implementation module" and "No Langfuse SDK objects
should leak into business logic." This file is where that boundary is
drawn concretely -- `import langfuse` appears exactly once in this whole
package, right here, and nowhere else. observer.py (the module every
other file actually talks to) only ever sees the return value of
`build_client()` below, typed as a plain `object` from its perspective --
it never imports the SDK itself.

WHY THE IMPORT IS LAZY (inside the function, not at module top): Langfuse
is an OPTIONAL dependency. `LANGFUSE_ENABLED=false` (the default) must
produce "zero Langfuse initialization and zero network calls" -- if this
module imported the SDK at top level, simply importing
`research_agent.langfuse` anywhere (even with the feature disabled) would
require the package to be installed and would pay its import cost. A
lazy import means an install with no `langfuse` package at all still runs
every existing test and every disabled-by-default code path with no
error -- confirmed by test_langfuse_client.py's
test_build_client_returns_none_without_the_sdk_installed.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("research_agent.langfuse")


def build_client(settings) -> Optional[Any]:
    """Construct a real Langfuse client, or return None.

    Returns None (never raises) whenever:
      - settings.langfuse_enabled is False (the default) -- the whole
        point of this being the FIRST check is that every branch after
        it is unreachable when disabled, so "disabled" costs nothing at
        all, not even an attempted import.
      - the `langfuse` package isn't installed (ImportError) -- an
        optional dependency missing is a configuration fact, not a
        crash.
      - required credentials are missing (empty public/secret key) --
        constructing a client with blank keys would either raise deep
        inside the SDK or silently produce a client that fails on first
        flush; failing here, with one clear log line, is more honest.
      - the SDK raises anything at all during construction (a bad host
        URL, a version mismatch, whatever) -- observability must never
        be able to take the research agent down with it.

    LANGFUSE_HOST is passed through EXACTLY as configured, with no
    validation of its shape and no assumption about which Langfuse
    deployment it points to -- Cloud, self-hosted, and enterprise all
    work identically from this function's point of view; only the
    string differs.
    """
    if not getattr(settings, "langfuse_enabled", False):
        return None

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning(
            "langfuse.disabled_missing_credentials",
            extra={"reason": "LANGFUSE_ENABLED=true but public/secret key is empty"},
        )
        return None

    try:
        from langfuse import Langfuse  # the one and only SDK import site
    except ImportError:
        logger.warning(
            "langfuse.disabled_sdk_not_installed",
            extra={"reason": "LANGFUSE_ENABLED=true but the langfuse package "
                              "is not installed (pip install langfuse)"},
        )
        return None

    try:
        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            release=settings.langfuse_release or None,
        )
        logger.info("langfuse.client_active", extra={"host": settings.langfuse_host})
        return client
    except Exception as exc:  # noqa: BLE001 -- observability must fail open
        logger.warning(
            "langfuse.disabled_client_init_failed",
            extra={"reason": type(exc).__name__, "error": str(exc)[:300]},
        )
        return None

"""Redaction of payloads on their way out to Langfuse.

WHY THIS FILE EXISTS: every other file in this package decides WHETHER
something is recorded. This one decides WHAT LEAVES THE PROCESS. Until
now nothing did: `client.py` passed five arguments to `Langfuse(...)`,
none of them `mask`, so every prompt, every model output, every retrieved
evidence chunk and every compiled report went to a third-party SaaS
verbatim. The retrieval path already treats retrieved evidence as
untrusted enough to fence against prompt injection before showing it to a
model -- shipping that same text unredacted to an external service was
the inconsistency this closes.

WHAT THE SDK GIVES US: `Langfuse(mask=...)` takes a callable invoked as
`mask(data=...)` for the `input`, `output` and `metadata` of every span,
generation and event. Two properties of the SDK's own handling, verified
against the installed package rather than assumed, shape the design here:

  1. It FAILS CLOSED. `_mask_attribute` wraps the call in try/except and
     substitutes "<fully masked due to failed mask function>" if the mask
     raises (langfuse/_client/span.py). So a bug in this file degrades to
     over-redaction, never to a leak. That is the opposite of this
     package's usual fail-open rule, and it is the correct direction for
     this one concern.
  2. Passing `mask=None` short-circuits before the callable -- so mode
     "off" returns None rather than an identity function, keeping the
     disabled path genuinely zero-cost.

WHAT THIS DOES NOT COVER, stated plainly rather than implied:

  * `create_score(comment=...)` does not go through `mask`; scores leave
    via score ingestion, not the observation path. Keep comments free of
    payload text (today they carry only "depth=N"-style values).
  * Observation NAMES are not masked. They are code-chosen constants
    (`node:classify`, `llm.fallback`), never user data.
  * `mask_otel_spans` is a SEPARATE, batch-oriented SDK hook with a
    different signature (`MaskOtelSpansFunction`, keyed by
    `OtelSpanIdentifier`). It is not wired here and would be its own
    piece of work.
  * MODE_PATTERNS is defence in depth, NOT a compliance guarantee. It
    catches four common shapes. Free-text PII that matches none of them
    still leaves the process. If that is not acceptable for a given
    deployment, MODE_ALL is the honest answer.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

MODE_OFF = "off"
MODE_PATTERNS = "patterns"
MODE_ALL = "all"
_VALID_MODES = (MODE_OFF, MODE_PATTERNS, MODE_ALL)

REDACTED = "[REDACTED]"

# Depth cap so a pathologically nested (or self-referential, which the SDK
# would fail to serialize anyway) payload cannot turn masking into a
# runaway recursion on the request path.
_MAX_DEPTH = 12
_TOO_DEEP = "[REDACTED:depth]"

# Deliberately conservative. Each of these has a distinctive shape that
# does not occur in ordinary research prose, because a pattern that fires
# on legitimate corpus text would quietly destroy the traces this package
# exists to produce. Order matters only in that bearer/key run before the
# digit-run rule, so a token containing digits is labelled as a token.
_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    # Authorization: Bearer <token>
    ("bearer", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{12,}")),
    # Provider API keys: sk-..., pk-..., rk-...
    ("api_key", re.compile(r"\b[sprk]k-[A-Za-z0-9_\-]{16,}\b")),
    # Email addresses
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b")),
    # Card-like runs: 13-19 digits, optionally split by spaces or hyphens.
    # Bounded at both ends so ordinary 4-digit years and 10-digit figures
    # in a corpus are untouched.
    ("card", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
)


def redact_text(text: str) -> str:
    """Apply every pattern to one string. Pure, no logging -- this runs on
    the request path and is called once per string leaf."""
    for label, pattern in _PATTERNS:
        text = pattern.sub(f"[REDACTED:{label}]", text)
    return text


def _walk(value: Any, mode: str, depth: int) -> Any:
    """Recurse through a payload, redacting string leaves and preserving
    structure. Structure is kept on purpose: knowing a trace carried a
    3-message prompt with a 12-key metadata dict is still useful when the
    contents are gone."""
    if depth > _MAX_DEPTH:
        return _TOO_DEEP

    # bool before int/float: bool IS an int in Python, and neither carries
    # payload text, so both pass through untouched in either mode.
    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        return REDACTED if mode == MODE_ALL else redact_text(value)

    if isinstance(value, dict):
        # Keys are field names chosen by this codebase, not user data, so
        # they are preserved -- a masked payload whose keys are also gone
        # is not debuggable at all.
        return {k: _walk(v, mode, depth + 1) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_walk(v, mode, depth + 1) for v in value]

    if isinstance(value, (set, frozenset)):
        return [_walk(v, mode, depth + 1) for v in value]

    # Anything else -- a dataclass, a Pydantic model, bytes. The SDK would
    # serialize it with `default=str`, which would carry whatever its repr
    # contains straight through unmasked, so stringify HERE and redact the
    # result rather than letting it past.
    try:
        text = str(value)
    except Exception:  # noqa: BLE001 -- a __str__ that raises must not leak
        return REDACTED
    return REDACTED if mode == MODE_ALL else redact_text(text)


def resolve_mode(raw: Any) -> str:
    """Normalize a configured mode. An unrecognized value resolves to
    MODE_PATTERNS, not MODE_OFF -- an unparseable setting must fail toward
    MORE redaction, never less."""
    mode = str(raw or "").strip().lower()
    if mode in _VALID_MODES:
        return mode
    logger.warning(
        "langfuse.mask_mode_unrecognized",
        extra={"configured": str(raw)[:60], "using": MODE_PATTERNS,
               "valid": ",".join(_VALID_MODES)},
    )
    return MODE_PATTERNS


def build_mask(settings: Any) -> Optional[Callable[..., Any]]:
    """Return the callable for `Langfuse(mask=...)`, or None for mode
    "off" so the SDK skips masking entirely.

    CALLED BY   langfuse/client.py::build_client, once per process.
    """
    mode = resolve_mode(getattr(settings, "langfuse_mask_mode", MODE_PATTERNS))
    if mode == MODE_OFF:
        return None

    def _mask(*, data: Any = None, **_kwargs: Any) -> Any:
        # The SDK calls this by keyword and tolerates extra kwargs in the
        # Protocol, hence the signature shape. Any exception raised here
        # is caught by the SDK and turned into full redaction, so there is
        # no fail-open hole to guard -- but _walk is written not to raise
        # anyway, so the common path never pays for that.
        return _walk(data, mode, 0)

    return _mask

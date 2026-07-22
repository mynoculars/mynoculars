"""
llm/client.py — Minimal chat clients: one real, one stub.

Purpose:
    Isolate ALL model I/O behind a two-method protocol so the rest of the
    codebase never touches HTTP, SDKs, or provider quirks.

Responsibilities:
    - ChatClient protocol: complete() and complete_json().
    - OpenAICompatibleClient: works for BOTH the local Qwen Cogito server
      (llama-server exposes /v1/chat/completions) and Gemini Flash (Google
      publishes an OpenAI-compatibility endpoint). One class, two configs.
    - StubClient: deterministic canned responses so the entire graph runs
      offline — used by tests and by LLM_MODE=stub.

Design decisions:
    - Why one client class instead of per-provider classes: both targets
      speak the same wire protocol; a second class would be duplicate code
      pretending to be abstraction. Alternative considered: LiteLLM (heavy
      dependency for a learning repo). Adding a genuinely different provider
      later = implement ChatClient, register in router — nothing else moves.
    - Why raw httpx instead of the openai SDK: ~40 lines of transparent HTTP
      teaches more than an SDK call, and drops a dependency.

Python mechanics used in this file, if any of this is new to you:
    Protocol (from typing)
        A "Protocol" defines a SHAPE a class must have (which methods, with
        which argument/return types) WITHOUT that class needing to
        explicitly inherit from it. This is Python's version of an
        "interface" in Java/C# or a "duck-typing contract": ANY object with
        a matching .complete(...) and .complete_json(...) method satisfies
        ChatClient, whether or not its class literally writes
        "class Foo(ChatClient):". Both OpenAICompatibleClient and
        StubClient below satisfy this Protocol simply by having those two
        methods with matching signatures — neither one inherits from
        ChatClient in its class definition.
    class OpenAICompatibleClient:  (no parent class in parentheses)
        A completely ordinary Python class — this is what you get by
        default when you don't inherit from anything. Its self.something
        assignments inside __init__ are how it stores its own configuration
        (base URL, API key, model name, etc.) for later use by its methods.
    httpx.Client(...)
        httpx is a third-party HTTP library (like `requests`, but with more
        modern features). httpx.Client(...) creates a reusable connection
        object — reusing one Client across many requests to the same server
        (rather than making a fresh connection every time) is faster
        because it can keep the underlying TCP connection open between
        calls.
    re.sub(pattern, replacement, text, flags=...)
        A regular-expression substitution: "find every part of `text`
        matching `pattern` and replace it with `replacement`". Used below
        to strip ```json / ``` markdown code-fence markers a model might
        wrap its JSON output in, even when explicitly told not to.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Protocol

import httpx

from research_agent.logging_setup import log_event

logger = logging.getLogger(__name__)

# A type alias (see agents/gathering.py's docstring for the general idea):
# "Message" is just a readable name for "a dict with 'role' and 'content'
# string keys", the shape every chat-API message takes, e.g.
# {"role": "user", "content": "What is Redis?"}.
Message = Dict[str, str]  # {"role": ..., "content": ...}

# Chat-template end-of-turn markers some local models (llama.cpp / Llama-
# family chat formats in particular) append after their actual answer —
# confirmed by a live debug trace showing Cogito doing exactly this after
# valid JSON. None of these strings can legally appear inside real JSON,
# so removing them is always safe.
_SENTINELS = ("<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>", "</s>")


class ChatClient(Protocol):
    """The only surface the agent uses to talk to a model.

    See the module docstring's Protocol explanation. Every node file in
    agents/*.py calls router.complete(...) or router.complete_json(...)
    (never these methods directly on a specific client) — the router,
    covered in llm/router.py, is what decides which underlying ChatClient
    actually handles a given call.
    """

    name: str

    def complete(self, messages: List[Message], temperature: float = 0.2) -> str:
        """Return the model's text response for a chat transcript."""
        ...

    def complete_json(self, messages: List[Message], temperature: float = 0.0) -> Dict[str, Any]:
        """Return the model's response parsed as a JSON object.

        Raises ValueError if the response is not parseable JSON — the router
        treats that as a quality failure and may fall back (see router.py).
        """
        ...


def _extract_json(text: str) -> Dict[str, Any]:
    """Parse JSON out of a model reply, tolerating ```json fences AND
    trailing chat-template sentinels.

    CALLED BY   OpenAICompatibleClient.complete_json and
                StubClient.complete_json, immediately below — this is the
                one place in the codebase that turns a raw string response
                into an actual Python dict.
    Why fences: small local models frequently wrap JSON in markdown fences
    even when told not to. Stripping fences before parsing avoids a
    fallback round-trip for a purely cosmetic failure.

    Why sentinels (added after a live trace confirmed this actually
    happens): a llama.cpp-style local model can append its chat template's
    end-of-turn marker right after the JSON, with no code fence around it
    at all — e.g. `{"goals": [...]} <|im_end|>`. The fence regex alone
    leaves that trailing text in place, json.loads then fails on it, and
    every single structured call from that provider gets thrown away as
    unparseable — even though the model answered correctly. _SENTINELS
    below lists the common ones seen across llama.cpp/Llama-family chat
    templates; stripping them is cheap and never touches genuinely valid
    JSON, since none of these strings can legally appear inside one.

    Belt-and-braces fallback: if the text STILL doesn't parse after both
    cleanup steps (some other, unanticipated junk before/after), extract
    the outermost {...} span and parse just that, rather than giving up
    immediately. This can't turn bad JSON into good JSON — if there's
    nothing shaped like an object in the text at all, it still raises,
    exactly as before.

    r"^```(?:json)?\\s*|\\s*```$" is a regular expression with two
    alternatives joined by "|" (meaning "match either side"):
      - ^```(?:json)?\\s*   matches a ``` (optionally followed by "json")
                            at the very START of the text, plus any
                            following whitespace. "(?:...)" is a
                            non-capturing group — it groups "json" together
                            with the "?" (optional) marker without also
                            creating a numbered capture group we'd have to
                            deal with.
      - \\s*```$            matches a ``` at the very END of the text,
                            with any preceding whitespace.
    flags=re.MULTILINE makes "^" and "$" match the start/end of EACH LINE
    rather than only the very start/end of the whole string, in case the
    fence markers sit on their own line inside a larger blob of text.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    for sentinel in _SENTINELS:
        cleaned = cleaned.replace(sentinel, "")
    cleaned = cleaned.strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        # str.find/.rfind locate the FIRST "{" and the LAST "}" in the
        # cleaned text — a cheap approximation of "the outermost object
        # span" that doesn't require a real parser. If either is missing,
        # there's nothing object-shaped here at all, so re-raise the
        # original error rather than manufacturing a confusing new one.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        obj = json.loads(cleaned[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("model returned JSON that is not an object")
    return obj


class OpenAICompatibleClient:
    """Chat client for any /v1/chat/completions endpoint."""

    def __init__(self, name: str, base_url: str, api_key: str, model: str,
                 timeout: float = 60.0, tracer: Any = None, display_label: str = ""):
        """Parameters map directly to Settings fields; see config.py.

        `tracer` (a Tracer, optional) receives the exact prompt/response/tokens/
        latency for every call when debug tracing is on. `display_label` is the
        human name shown in the trace banner (e.g. "GOOGLE GEMINI FLASH"); it
        falls back to the model id. `_trace_node` is set by the router before
        each call so the trace shows which graph node issued it.

        CALLED BY   llm/router.py::FallbackRouter.from_settings, once per
                    configured provider (primary always; Mistral/Gemini
                    only if their API key is set).
        """
        self.name = name
        self._model = model
        self._tracer = tracer
        self._label = display_label or model
        # A leading underscore (self._trace_node, self._http, etc.) is a
        # Python NAMING CONVENTION, not an enforced access restriction —
        # unlike some other languages, Python does not have a true
        # "private" keyword. The underscore just signals to other
        # developers "this is an internal implementation detail, please
        # don't reach in and use it directly from outside this class."
        self._trace_node: Optional[str] = None
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def set_trace_node(self, node: Optional[str]) -> None:
        """Router sets the current node name so traces attribute each call.

        CALLED BY   llm/router.py::FallbackRouter.set_node, right before
                    every complete()/complete_json() call — this is how a
                    trace entry (see tracing.py) knows which graph node
                    (e.g. "goal_manager") issued a given LLM call.
        """
        self._trace_node = node

    def complete(self, messages: List[Message], temperature: float = 0.2) -> str:
        """POST the transcript; return assistant text. Raises httpx errors
        upward — the router owns retry/fallback policy, not this class.

        READS   nothing from ResearchState — this class knows nothing
                about the graph at all; it only ever sees the `messages`
                list handed to it.
        CALLS   the actual HTTP request to whichever provider this
                instance was configured for (local Qwen, Mistral, or
                Gemini — determined entirely by the base_url/api_key it
                was constructed with).
        RETURNS the assistant's raw text reply. If the HTTP call fails
                (bad status code, network error, timeout), an exception
                propagates straight out of this method — this class makes
                NO decision about what to do next; that decision belongs
                entirely to FallbackRouter (llm/router.py).
        """
        started = time.perf_counter()
        resp = self._http.post(
            "/chat/completions",
            json={"model": self._model, "messages": messages, "temperature": temperature},
        )
        # raise_for_status() raises an httpx.HTTPStatusError if the response
        # code is 4xx or 5xx (i.e. the server reported an error) — it does
        # nothing (returns None) for a normal 2xx success response.
        resp.raise_for_status()
        data = resp.json()
        latency = time.perf_counter() - started
        text: str = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
        log_event(logger, "llm.call", provider=self.name, model=self._model,
                  label=self._label, node=self._trace_node,
                  latency_s=round(latency, 3),
                  prompt_tokens=pt, completion_tokens=ct)
        if self._tracer is not None:
            self._tracer.record_llm(self._label, self._trace_node, messages,
                                    text, pt, ct, latency)
        return text

    def complete_json(self, messages: List[Message], temperature: float = 0.0) -> Dict[str, Any]:
        """complete() then strict-parse. ValueError on non-JSON output.

        This method does almost nothing itself — it calls self.complete()
        above to get the raw text, then hands that text to the module-level
        _extract_json() function to turn it into an actual dict.
        """
        return _extract_json(self.complete(messages, temperature))


class StubClient:
    """Offline deterministic client.

    Looks at the LAST user message for a routing hint the prompt templates
    embed (e.g. "TASK=classify") and returns a canned, schema-valid answer.
    This keeps stub behavior honest: the same prompts and the same JSON
    schemas as live mode, just fixed content.

    Note this class has NO parent class listed — it satisfies the
    ChatClient Protocol purely by having matching .complete() and
    .complete_json() methods, exactly like OpenAICompatibleClient above.
    """

    name = "stub"

    def __init__(self, tracer: Any = None):
        self._tracer = tracer
        self._trace_node: Optional[str] = None

    def set_trace_node(self, node: Optional[str]) -> None:
        self._trace_node = node

    # A class-level dict (defined once, shared by every StubClient
    # instance — it is never mutated at runtime, so sharing it is safe).
    # Each key is a TASK tag a prompt template embeds (see
    # prompts/templates.py — every builder function's prompt text starts
    # with a line like "TASK=classify"); each value is the canned dict this
    # stub returns when it spots that tag in the last message.
    _CANNED: Dict[str, Dict[str, Any]] = {
        "classify": {"intent": "Comparison", "confidence": 0.9},
        "goals": {"goals": [
            {"goal_id": "g1", "description": "Identify the key differences"},
            {"goal_id": "g2", "description": "Summarize practical tradeoffs"},
        ]},
        "expand": {"tasks": [
            {"query": "key differences", "goal_id": "g1", "priority": 2},
            {"query": "practical tradeoffs", "goal_id": "g2", "priority": 1},
        ]},
        "gaps": {"tasks": []},
        "critique": {"passed": True, "score": 0.9, "notes": []},
        "quality": {"score": 0.9},
    }

    def complete(self, messages: List[Message], temperature: float = 0.2) -> str:
        """Return canned JSON (as text) for known tasks, else a fixed report.

        READS   messages[-1]["content"] — ONLY the last message in the
                transcript (that's where every prompts/templates.py builder
                puts its "TASK=..." tag).
        RETURNS a JSON string matching one of the _CANNED entries above if
                a matching "TASK=<tag>" substring is found anywhere in that
                last message; otherwise a fixed, human-readable placeholder
                report string (used for the compiler's free-text call,
                which has no TASK tag of its own listed in _CANNED).
        """
        last = messages[-1]["content"]
        answer = None
        # self._CANNED.items() loops over (key, value) pairs — here
        # (tag, payload) — same iteration pattern used in state.py's
        # merge_failed_keys/merge_counters.
        for tag, payload in self._CANNED.items():
            if f"TASK={tag}" in last:
                answer = json.dumps(payload)
                break  # stop at the first match — tags are mutually exclusive
        if answer is None:
            answer = ("# Research Report (stub mode)\n\n"
                      "This deterministic report proves the full graph executed "
                      "offline. Switch LLM_MODE=live for real model output.")
        if self._tracer is not None:
            self._tracer.record_llm("STUB (offline)", self._trace_node,
                                    messages, answer, None, None, 0.0)
        return answer

    def complete_json(self, messages: List[Message], temperature: float = 0.0) -> Dict[str, Any]:
        """Parse the canned reply."""
        return _extract_json(self.complete(messages, temperature))

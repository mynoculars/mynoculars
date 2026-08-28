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
import threading
import time
from typing import Any, Dict, List, Optional, Protocol, Tuple

import httpx

from research_agent import langfuse as lf
from research_agent.logging_setup import log_event, run_id_var

logger = logging.getLogger(__name__)

# A type alias (see agents/gathering.py's docstring for the general idea):
# "Message" is just a readable name for "a dict with 'role' and 'content'
# string keys", the shape every chat-API message takes, e.g.
# {"role": "user", "content": "What is Redis?"}.
Message = Dict[str, str]  # {"role": ..., "content": ...}


class TruncatedGenerationError(RuntimeError):
    """A provider reported its own generation was cut off at the token limit.

    RAISED BY   OpenAICompatibleClient.complete, when the response carries
                finish_reason == "length" and no chat-template sentinel was
                trimmed (see FIX-2 there for why that second condition
                matters).
    CAUGHT BY   nobody specifically — llm/router.py's fallback handlers catch
                Exception broadly, which is the whole point: a truncated
                answer should behave exactly like a transport error and move
                the router to the next provider. A distinct type exists so a
                test, or a future retry-with-larger-budget policy, can tell
                "the model was cut off" apart from "the network broke".
    """

# Chat-template end-of-turn markers some local models (llama.cpp / Llama-
# family chat formats in particular) append after their actual answer —
# confirmed by a live debug trace showing Cogito doing exactly this after
# valid JSON. None of these strings can legally appear inside real JSON,
# so removing them is always safe.
_SENTINELS = ("<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>", "</s>")
_SENTINEL_RE = re.compile("|".join(re.escape(s) for s in _SENTINELS))


def sentinel_segments(text: str) -> List[str]:
    """Every non-empty, stripped run of text between chat-template sentinels.

    CALLED BY   _truncate_at_sentinel (free-text path, which takes the
                FIRST -- anything after the model's first end-of-turn is a
                runaway continuation, never report content) and
                complete_json (structured path, which tries them ALL).

    Why the two paths must differ: taking the first segment is right for
    prose and wrong for JSON. Live (runs p205.98/.100-check) the local
    model emitted a short fragment, then a sentinel, then the real answer
    -- raw_chars 2895 kept_chars 21, and raw_chars 1153 kept_chars 18. The
    free-text rule discarded the answer and every structured call fell back
    to the secondary provider. Taking the LONGEST segment instead would be
    just as wrong the other way: it would resurrect exactly the
    hallucinated extra conversation the truncator exists to kill. Neither
    position is reliably correct, so the JSON path stops guessing and tries
    each segment until one parses.
    """
    return [seg.strip() for seg in _SENTINEL_RE.split(text) if seg.strip()]


def _truncate_at_sentinel(text: str) -> str:
    """Cut a FREE-TEXT response off at the first chat-template sentinel.

    CALLED BY   OpenAICompatibleClient.complete, below — the free-text
                path used exclusively by agents/compilation.py::
                compiler_node. This is a DIFFERENT defect from the one
                _extract_json (above this in the file, used by
                complete_json) already handled: that one strips a
                sentinel that appears immediately after otherwise-valid
                JSON. This one guards against something worse, confirmed
                by a live run: a local model whose llama-server chat
                template / stop-token configuration isn't halting
                generation at its own end-of-turn can keep going,
                hallucinating an entire extra fake conversation —
                repeating the prompt, inventing a fictitious "system"
                turn, even re-generating a second copy of its own answer
                — all of which, with NO cleanup at all before this fix,
                became the literal, user-facing final_report verbatim
                (compiler_node has no JSON schema to validate against, so
                nothing catches this the way a malformed JSON parse
                would).

    Unlike _extract_json's approach (remove the sentinel substring
    wherever it occurs, keep everything else), this TRUNCATES — everything
    from the first sentinel onward is either the sentinel itself or a
    hallucinated continuation, never legitimate report content, so cutting
    it off (not just deleting the marker) is the correct response here.

    Returns the text unchanged if no sentinel is found — the overwhelming
    majority of calls, where the model behaved.
    """
    if not _SENTINEL_RE.search(text):
        return text
    # Take the first NON-EMPTY segment between sentinels, not everything
    # before the first one. The original truncate-at-first-index form
    # assumed a sentinel is always TRAILING. Live (runs p205.67/.70/.71),
    # the critic node repeatedly produced a sentinel at index 0 followed
    # by the real answer -- raw_chars 3898, kept_chars 0 -- so complete()
    # returned "", complete_json() then raised JSONDecodeError, and EVERY
    # structured critic call fell back to the secondary provider despite
    # the local model having answered correctly. Splitting keeps the
    # trailing-sentinel case byte-identical (the first segment is the
    # answer) while recovering the leading-sentinel case.
    for segment in sentinel_segments(text):
        return segment
    # Nothing but sentinels: return the raw text rather than "" so the
    # caller sees the real (useless) response instead of a silent empty
    # string that looks like a successful call returning nothing.
    return text.strip()


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


# D-93: response-body fragments that mean "your prompt did not fit",
# lower-cased for matching. llama.cpp, vLLM and the OpenAI-compatible
# servers built on them phrase this differently; these are the shapes
# observed or documented rather than a guess at a standard, because there
# is no standard -- the HTTP status is a plain 400 either way.
_CONTEXT_OVERFLOW_MARKERS = (
    "exceed context",
    "exceeds context",
    "context length",
    "context window",
    "too many tokens",
    "maximum context",
    "prompt is too long",
)


def looks_like_context_overflow(body: str) -> bool:
    """Does this error body say the PROMPT did not fit?

    D-93. Without this, a context rejection is indistinguishable from a
    429, a 500 or a dead port -- every one arrives as
    `llm.fallback reason=HTTPStatusError` and reads as flakiness. It is
    not flakiness: it is deterministic, it recurs on every run with a
    prompt that size, and the remedy (raise `-c`, or configure
    LLM_PRIMARY_CONTEXT_TOKENS so the hop is skipped) is entirely
    different from the remedy for a transient error.

    Substring matching on the response body, deliberately: there is no
    standard error code for this across OpenAI-compatible servers and the
    HTTP status is a plain 400 in every case. A false negative simply
    restores today's behaviour -- an unlabelled fallback -- so being
    wrong here costs nothing.
    """
    lowered = (body or "").lower()
    return any(marker in lowered for marker in _CONTEXT_OVERFLOW_MARKERS)


def estimate_prompt_tokens(messages: List[Message]) -> int:
    """A deliberately crude token estimate for a message list.

    ~4 characters per token is the long-standing rule of thumb for
    English through BPE tokenizers. It is APPROXIMATE and this function
    does not pretend otherwise -- the alternative is a real tokenizer
    dependency to answer a question whose only consumer is "is this
    prompt obviously far too big for a 1536-token window".

    Used ONLY to skip a hop configured as unable to take the prompt
    (llm/router.py). Because an OVER-estimate would skip a provider that
    would have worked, the router applies a safety margin on top rather
    than trusting this exactly -- see _skips_for_context there.
    """
    total = sum(len(m.get("content") or "") + len(m.get("role") or "")
                for m in messages)
    return total // 4


def strip_code_fence(text: str) -> str:
    r"""Remove one leading/trailing markdown code fence, any language tag.

    CALLED BY   agents/compilation.py::compiler_node, on the raw string
                complete() returns for templates.py::compile_report — the
                ONE call site in this file whose caller expects prose back,
                not JSON (see that function's own docstring).
    WHY THIS EXISTS: compile_report's prompt explicitly asks for Markdown
    prose, but a model under fallback (observed live: Mistral, reached
    after a quality-reject bounced the call out of the primary provider)
    can still answer with a ```json ...``` block despite that instruction
    -- the same "small models fence everything even when told not to"
    behaviour _extract_json below already exists to handle for the
    JSON-mode call sites. compiler_node had NO equivalent handling on its
    free-text path, so the fence markers (and, if the model ignored the
    instruction outright, a full JSON document instead of prose) landed
    verbatim in state.final_report and were printed to the user as-is.

    This strips the FENCE, whatever the language tag -- ```json,
    ```markdown, bare ```, or a tag containing non-word characters like
    ```c++ or ```objective-c. It deliberately does NOT try to detect or
    reformat JSON *content* into Markdown: that would mean guessing a
    schema-to-prose transform for arbitrary model output, a much larger
    and riskier change than removing three delimiter lines. A model that
    ignores the prompt and answers with genuine JSON *content* still
    produces JSON content after this call -- just without a fence wrapped
    around it. Closing that remaining gap is a prompting/model-selection
    question, not a safe one-line code fix, and out of scope here.

    IMPLEMENTATION NOTE (this replaced an earlier, buggier version): a
    single regex with a `|` alternation -- one side matching the opening
    fence, the other the closing one -- looks tidy but conflates two
    independent jobs into one expression, and `\w+` for the language tag
    is too narrow: it does not match `c++`, `objective-c`, or any tag
    with a non-word character, silently leaving a stray fragment (e.g.
    "-c\n...") behind in the output instead of the actual content.

    This version does the two jobs separately and in sequence: match the
    OPENING fence first (```<anything but a backtick/newline><newline>,
    so the tag itself is unconstrained -- unlike `\w+`, it can never
    misparse a punctuated tag). Only then does it search for a CLOSING
    fence, and only within the text AFTER the opening match -- never
    independently against the whole string. That ordering is what makes
    the two steps safe to combine: a closing search scoped to "everything
    after the opening" can't be fooled by, and can't overlap with, the
    opening match itself, so no extra bookkeeping is needed to keep the
    two sides from stepping on each other. It also means an unfenced
    string (or one with only an opening OR only a closing marker, not
    both) is left completely untouched -- there is no way for one bare
    delimiter to be "half stripped".

    A single line with no newline at all -- e.g. `` ```hello``` `` -- is
    intentionally left AS-IS: per CommonMark, a fenced code block's
    opening delimiter must end its own line, so a fence-like run with no
    newline separating a "tag" from what follows is not a real fence,
    just three backticks as literal content. Treating it as unfenced
    avoids swallowing genuine content into a falsely-detected tag.
    """
    if not text:
        return text
    body = text.strip()
    if not body:
        return body
    open_match = re.match(r"```[^`\n]*\n", body)
    if not open_match:
        return body
    rest = body[open_match.end():]
    close_match = re.search(r"\n?```[ \t]*$", rest)
    if not close_match:
        return body
    return rest[:close_match.start()]


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

def _prompt_tag_for_node(node: Optional[str]) -> dict:
    """Which prompt template/version produced this node's calls, as
    Langfuse generation metadata -- pulled out of complete() as a plain
    function so it is testable without a live or mocked HTTP call.

    See PROMPT_VERSIONS in prompts/templates.py for the full rationale
    (this is the metadata-only variant of prompt linking, deliberately
    not the network-reading get_prompt()-based one). Imported inside the
    function, not at module level, to avoid a prompts->llm.client->prompts
    import cycle: templates.py imports Message from this module.

    A node absent from the table (StubClient runs, "merger", or any
    future node not driven by one fixed builder) returns an empty dict --
    no prompt_name/prompt_version keys at all, not placeholder values, so
    grouping by those fields in the Langfuse UI cleanly separates "tagged"
    from "untagged" rather than inventing a fake bucket for the latter.
    """
    from research_agent.prompts.templates import PROMPT_VERSIONS
    tagged = PROMPT_VERSIONS.get(node)
    if tagged is None:
        return {}
    name, version = tagged
    return {"prompt_name": name, "prompt_version": version}


class OpenAICompatibleClient:
    """Chat client for any /v1/chat/completions endpoint."""

    def __init__(self, name: str, base_url: str, api_key: str, model: str,
                 timeout: float = 60.0, tracer: Any = None, display_label: str = "",
                 max_tokens: Optional[int] = None,
                 context_tokens: int = 0):
        """Parameters map directly to Settings fields; see config.py.

        `tracer` (a Tracer, optional) receives the exact prompt/response/tokens/
        latency for every call when debug tracing is on. `display_label` is the
        human name shown in the trace banner (e.g. "GOOGLE GEMINI FLASH"); it
        falls back to the model id. `_trace_node` is set by the router before
        each call so the trace shows which graph node issued it.

        `max_tokens` (Guardrail G6, P205 Phase 2): the generation budget
        sent to the provider on every call, or None (the default) to omit
        the field entirely and let the provider apply its own default --
        preserves the exact pre-G6 request shape for any caller that
        doesn't pass this. Before G6 the ONLY control on runaway
        generation was _truncate_at_sentinel below, applied AFTER a full
        response (and its full latency and token cost) already came
        back -- confirmed live, repeatedly, across this whole session's
        traces ("llm.truncated_runaway_generation" fired on classify,
        goal_manager, task_expander, gap_generator, and both compiler
        calls in nearly every run). This bounds the request itself; the
        sentinel truncation still runs on whatever comes back, since a
        capped response can still contain a sentinel mid-stream.

        CALLED BY   llm/router.py::FallbackRouter.from_settings, once per
                    configured provider (primary always; Mistral/Gemini
                    only if their API key is set).
        """
        self.name = name
        self._model = model
        self._tracer = tracer
        self._max_tokens = max_tokens
        # complete() truncates at the first chat-template sentinel, which
        # is correct for prose and lossy for JSON. complete_json needs the
        # untruncated text to try the other segments; threading.local keeps
        # it per-worker, since one client is shared across the parallel
        # search_worker fan-out.
        self._raw = threading.local()
        # D-93: 0 means "unknown", which is what every provider except a
        # deliberately-configured primary reports. Read by llm/router.py
        # via getattr -- the same duck-typed optional-capability pattern
        # drain_usage and drain_retrieval_counts already use.
        self.context_tokens = int(context_tokens or 0)
        self._label = display_label or model
        # A leading underscore (self._trace_node, self._http, etc.) is a
        # Python NAMING CONVENTION, not an enforced access restriction —
        # unlike some other languages, Python does not have a true
        # "private" keyword. The underscore just signals to other
        # developers "this is an internal implementation detail, please
        # don't reach in and use it directly from outside this class."
        # threading.local(), not a plain attribute: ONE client instance is
        # shared by every node in the process, and set_trace_node mutates
        # it out-of-band from the call it labels. Under any future
        # concurrent LLM node that races, mislabelling trace entries and
        # llm.call log lines. Per-thread storage removes the race without
        # changing the single-threaded behaviour at all.
        self._trace_local = threading.local()
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
        self._trace_local.node = node

    @property
    def _trace_node(self) -> Optional[str]:
        """The node name set on THIS thread, or None if none was set."""
        return getattr(self._trace_local, "node", None)

    def drain_usage(self) -> Tuple[int, int]:
        """Return (prompt_tokens, completion_tokens) for THIS thread's most
        recent completed call, and clear it. (0, 0) if there was none.

        CALLED BY   llm/router.py::FallbackRouter._bump_usage, immediately
                    after each successful provider call -- including the
                    quality-judge calls, which are real spend and are
                    counted the same way.
        WHY DRAIN RATHER THAN PEEK: the same reason drain_counters gives on
        the router itself. A read-and-reset makes "each call reports only
        what IT cost" structurally true, instead of something every call
        site has to remember; a peek would let one provider's usage be
        added twice if a later call raised before reporting its own.

        Duck-typed on purpose -- the router looks this up with getattr and
        skips a provider that lacks it, the same optional-capability
        pattern corpus_search's drain_retrieval_counts and this class's own
        set_trace_node/close already use. That is what lets StubClient (and
        every hand-written test fake) stay unchanged.
        """
        usage = getattr(self._raw, "usage", None)
        self._raw.usage = None
        return usage or (0, 0)

    def close(self) -> None:
        """Close the underlying httpx.Client. Safe to call twice.

        CALLED BY   llm/router.py::FallbackRouter.close.
        WHY THIS EXISTS: this class opens a persistent httpx.Client in
        __init__ (one per configured provider) and nothing ever closed it.
        Harmless for a short CLI process the OS is about to reap, a real
        leak for the long-lived FastAPI process — and, like the
        checkpointer connection P2-08 closed, never a design decision,
        just an oversight.
        """
        self._http.close()

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
        wall_start = time.time()
        payload = {"model": self._model, "messages": messages, "temperature": temperature}
        if self._max_tokens is not None:
            payload["max_tokens"] = self._max_tokens
        resp = self._http.post("/chat/completions", json=payload)
        # raise_for_status() raises an httpx.HTTPStatusError if the response
        # code is 4xx or 5xx (i.e. the server reported an error) — it does
        # nothing (returns None) for a normal 2xx success response.
        if resp.status_code >= 400 and looks_like_context_overflow(resp.text):
            # D-93: the SAME exception the caller already handles (the
            # router hops on any Exception) -- this only makes the log say
            # which KIND of failure it was. Logged here, where the body is
            # still in hand, rather than left for the router to guess.
            log_event(logger, "llm.context_overflow", level=logging.WARNING,
                      provider=self.name, node=self._trace_node,
                      estimated_prompt_tokens=estimate_prompt_tokens(messages),
                      configured_context_tokens=self.context_tokens,
                      status=resp.status_code, body=resp.text[:200])
        resp.raise_for_status()
        data = resp.json()
        latency = time.perf_counter() - started
        raw_text: str = data["choices"][0]["message"]["content"]
        self._raw.text = raw_text
        # See _truncate_at_sentinel's docstring above for exactly what
        # this guards against: a model that keeps generating past its own
        # end-of-turn, hallucinating an entire extra fake conversation.
        # We trace the RAW text below (diagnostically useful — it shows
        # you the runaway generation itself, which is what you'd want to
        # see if you're debugging the underlying llama-server config
        # issue) but RETURN the truncated version, since the raw text is
        # never a legitimate answer past its first sentinel.
        text = _truncate_at_sentinel(raw_text)
        usage = data.get("usage", {})
        pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
        # D-102: reasoning models spend the output budget on tokens that
        # `completion_tokens` does not report. The OpenAI-compatible field
        # for them is usage.completion_tokens_details.reasoning_tokens;
        # providers that have no such concept simply omit it, and this
        # stays None. Read here, used ONLY by the truncation branch below
        # -- it is deliberately NOT added to _bump_usage's totals, which
        # count what the provider billed as prompt/completion and must not
        # start meaning something else.
        _details = usage.get("completion_tokens_details") or {}
        reasoning_tokens = _details.get("reasoning_tokens")
        # D-86: stash this call's token usage for FallbackRouter to drain
        # (llm/router.py::_bump_usage) one line after this method returns.
        #
        # On self._raw -- the threading.local() this class ALREADY keeps --
        # and for exactly the reason its own comment gives for _raw.text:
        # one client instance is shared across the parallel search_worker
        # fan-out, so a plain attribute would let one thread drain another
        # thread's usage. Reusing the existing holder rather than adding a
        # second one keeps "per-call state that must not race" in one
        # place.
        #
        # A provider that reports no usage block (some OpenAI-compatible
        # servers omit it) yields (0, 0) rather than None, so the router
        # never has to special-case it -- an unreported call simply adds
        # nothing to the run total, which is honest: we genuinely do not
        # know what it cost.
        self._raw.usage = (int(pt or 0), int(ct or 0))
        # FIX-2 (run p205.211 root cause, third link in the chain). Nothing
        # in this class ever looked at finish_reason, so a generation the
        # PROVIDER itself reported as cut off was returned as a finished
        # answer. Observed live: gemini-3.5-flash returned 162 completion
        # tokens ending mid-number ("...moderate slightly to around 6") and
        # 160 tokens ending mid-sentence, and both shipped as the final
        # report because gemini is last in the chain and the last provider
        # has no quality gate.
        #
        # Raising here is the correct layer: this class's contract is
        # already "raise upward, the router owns fallback policy". A hard
        # error makes the router hop, and — with FIX-3 — fall back to the
        # best earlier answer instead of shipping a fragment.
        #
        # The `text == raw_text` guard matters: when _truncate_at_sentinel
        # DID trim something, the model reached its own end-of-turn and
        # then ran away, so `length` describes the discarded runaway tail,
        # not the answer. Only an untrimmed response is genuinely cut off.
        finish_reason = (data.get("choices") or [{}])[0].get("finish_reason")
        if finish_reason == "length" and text == raw_text:
            # D-102: WHOSE ceiling was hit. As first written this line
            # reported max_tokens unconditionally, which reads as "we hit
            # our own limit" and sends you to .env. Live (p205.254-check)
            # that was false both times it fired: gemini-3.5-flash
            # reported finish_reason=length at completion_tokens 616 and
            # 2150 against max_tokens 8192. Our generation budget was
            # demonstrably NOT the binding constraint -- the ceiling came
            # from the provider side (its own output cap, a gateway limit,
            # or reasoning tokens spending a budget completion_tokens does
            # not report). Raising LLM_MAX_TOKENS would not have helped,
            # and the log line said to go and try.
            #
            # cap_source is DERIVED, never guessed: "ours" only when the
            # provider actually reached the number we sent. Anything short
            # of it means something else stopped the generation, and the
            # only honest thing to say is that it was not us.
            # reasoning_tokens counts toward the budget where a provider
            # reports it, so it is added before the comparison; where it
            # is absent this is exactly the old completion_tokens check.
            billed = (ct or 0) + (reasoning_tokens or 0)
            if self._max_tokens is None:
                cap_source = "provider"      # we sent no cap at all
            elif ct is None:
                cap_source = "unknown"       # nothing to compare against
            elif billed >= self._max_tokens:
                cap_source = "ours"
            else:
                cap_source = "provider"
            log_event(logger, "llm.truncated_by_token_limit", level=logging.WARNING,
                      provider=self.name, node=self._trace_node,
                      completion_tokens=ct, reasoning_tokens=reasoning_tokens,
                      max_tokens=self._max_tokens, cap_source=cap_source,
                      kept_chars=len(text))
            # The same attribution in the exception text, because on the
            # chain-exhaustion path this string is what a human actually
            # reads (D-101 puts it in main()'s message).
            whose = {"ours": f"OUR max_tokens={self._max_tokens}",
                     "provider": "the PROVIDER's own ceiling, not our "
                                 f"max_tokens={self._max_tokens}",
                     "unknown": "an unattributable ceiling"}[cap_source]
            raise TruncatedGenerationError(
                f"{self.name}/{self._model} stopped at the token limit "
                f"(finish_reason=length, completion_tokens={ct}"
                + (f", reasoning_tokens={reasoning_tokens}"
                   if reasoning_tokens is not None else "")
                + f"); cap was {whose}; the answer is "
                f"incomplete and must not be used")
        if text != raw_text:
            log_event(logger, "llm.truncated_runaway_generation", level=logging.WARNING,
                      provider=self.name, node=self._trace_node,
                      raw_chars=len(raw_text), kept_chars=len(text))
        # ONE instrumentation call for this LLM call, not two: prompt_messages/
        # response used to go to a SEPARATE recorder (self._tracer.record_llm,
        # below this comment until this change) with its own file format.
        # Now they're extra fields on the SAME log_event call the summary
        # fields already needed — only attached when a real Tracer is
        # enabled, so a non-debug run's JSON line is byte-identical in shape
        # to before this existed (JsonLineFormatter drops these two keys
        # from the JSON view regardless; NarrativeFormatter is what renders
        # them, only when present). See tracing.py and logging_setup.py's
        # module docstrings for the full "one instrumentation path" design.
        trace_fields = ({"prompt_messages": messages, "response": raw_text}
                        if self._tracer is not None and self._tracer.enabled else {})
        log_event(logger, "llm.call", provider=self.name, model=self._model,
                  label=self._label, node=self._trace_node,
                  latency_s=round(latency, 3),
                  prompt_tokens=pt, completion_tokens=ct, **trace_fields)
        # Phase 3: one Langfuse generation per real provider call -- the
        # SAME provider/model/latency/token figures the log line above
        # just recorded, so this can never drift from what's already
        # logged. thread_id comes from run_id_var, the SAME ContextVar
        # log_event itself reads (see logging_setup.py) -- reusing it
        # here means this class still knows nothing about the graph or
        # about cli.py's thread_id, exactly as its own docstring says.
        meta = {"label": self._label, "temperature": temperature}
        meta.update(_prompt_tag_for_node(self._trace_node))
        lf.generation(
            run_id_var.get(), self._trace_node or "llm",
            provider=self.name, model=self._model,
            input=messages, output=text,
            prompt_tokens=pt or 0, completion_tokens=ct or 0,
            start_time=wall_start, end_time=wall_start + latency,
            metadata=meta,
        )
        return text

    def complete_json(self, messages: List[Message], temperature: float = 0.0) -> Dict[str, Any]:
        """complete() then strict-parse. ValueError on non-JSON output.

        This method does almost nothing itself — it calls self.complete()
        above to get the raw text, then hands that text to the module-level
        _extract_json() function to turn it into an actual dict.
        """
        text = self.complete(messages, temperature)
        try:
            return _extract_json(text)
        except ValueError:
            # complete() kept the FIRST segment, which is right for prose
            # and can be a stray fragment for JSON (see sentinel_segments).
            # Re-split and try the rest before declaring the provider
            # unusable and paying for a fallback hop -- live, this was
            # every structured critic/gap call on the local model.
            raw = getattr(self._raw, "text", text)
            for segment in sentinel_segments(raw)[1:]:
                try:
                    return _extract_json(segment)
                except ValueError:
                    continue
            raise


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
        self._trace_local = threading.local()

    def set_trace_node(self, node: Optional[str]) -> None:
        self._trace_local.node = node

    @property
    def _trace_node(self) -> Optional[str]:
        return getattr(self._trace_local, "node", None)

    def close(self) -> None:
        """No transport to release — present so FallbackRouter.close can
        call close() uniformly across every provider in the chain."""
        return None

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
        # P2-12: empty by default — stub mode has no way to judge real
        # semantic conflict, so it deterministically reports nothing
        # contested. Tests that want to exercise the contested/E2 path
        # supply their own stub returning a non-empty list explicitly.
        "contradictions": {"contested_goal_ids": []},
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
        # Stub mode previously had NO log_event("llm.call", ...) at all —
        # only the now-removed self._tracer.record_llm(...) captured
        # anything about a stub call, and only when tracing was on. That
        # meant stub runs produced zero JSON telemetry for LLM calls, a gap
        # this single-call-site design closes as a direct consequence of
        # having exactly one instrumentation path (see tracing.py's module
        # docstring): every ChatClient implementation now emits the same
        # "llm.call" event, live or stub.
        trace_fields = ({"prompt_messages": messages, "response": answer}
                        if self._tracer is not None and self._tracer.enabled else {})
        log_event(logger, "llm.call", provider=self.name, model="stub",
                  label="STUB (offline)", node=self._trace_node,
                  latency_s=0.0, prompt_tokens=None, completion_tokens=None,
                  **trace_fields)
        return answer

    def complete_json(self, messages: List[Message], temperature: float = 0.0) -> Dict[str, Any]:
        """Parse the canned reply."""
        return _extract_json(self.complete(messages, temperature))

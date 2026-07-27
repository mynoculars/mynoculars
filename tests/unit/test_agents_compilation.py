"""
tests/unit/test_agents_compilation.py — agents/compilation.py::compiler_node.

WHY THIS FILE EXISTS: compile_report's prompt explicitly asks for Markdown
prose, but a live trace showed a fallback provider (Mistral, reached after
a quality-reject bounced the call off the primary provider) answering with
a ```json ...``` block anyway. Nothing on that call path stripped the
fence — the identical defensive pattern already existed for the JSON-mode
call sites (llm/client.py::_extract_json) but was never carried to this
one. These tests cover the fix (llm/client.py::strip_code_fence) and its
wiring into compiler_node.
"""

from research_agent.agents.compilation import build_compiler_node
from research_agent.llm.client import strip_code_fence
from research_agent.state import ResearchState


class _FakeRouter:
    """Just enough of FallbackRouter's interface for compiler_node."""

    def __init__(self, reply: str):
        self._reply = reply
        self.node = None

    def set_node(self, node):
        self.node = node

    def complete(self, messages, temperature=0.2):
        return self._reply

    def drain_counters(self):
        return {}


# ---------------------------------------------------------------------------
# strip_code_fence — pure function
# ---------------------------------------------------------------------------

def test_strip_code_fence_removes_json_tagged_fence():
    text = '```json\n{"a": 1}\n```'
    assert strip_code_fence(text) == '{"a": 1}'


def test_strip_code_fence_removes_bare_fence():
    text = "```\n# Report\nSome prose.\n```"
    assert strip_code_fence(text) == "# Report\nSome prose."


def test_strip_code_fence_removes_any_language_tag():
    text = "```markdown\n# Report\n```"
    assert strip_code_fence(text) == "# Report"


def test_strip_code_fence_is_a_noop_on_unfenced_text():
    text = "# Report\n\nJust plain markdown, no fence at all."
    assert strip_code_fence(text) == text


def test_strip_code_fence_does_not_touch_a_fence_in_the_middle():
    """Only a fence WRAPPING the whole reply should be removed — a code
    block the model legitimately included as part of the report body must
    survive untouched."""
    text = "# Report\n\nExample query:\n```sql\nSELECT 1;\n```\n\nMore prose."
    assert strip_code_fence(text) == text


def test_strip_code_fence_handles_punctuated_language_tags():
    """A regression: the first implementation used \\w+ for the language
    tag, which stops at the first non-word character. A tag like c++ or
    objective-c left a stray fragment (e.g. "-c\\n...") in the output
    instead of the actual content."""
    assert strip_code_fence("```c++\nint x = 1;\n```") == "int x = 1;"
    assert strip_code_fence("```objective-c\n[foo bar];\n```") == "[foo bar];"


def test_strip_code_fence_leaves_a_single_line_pseudo_fence_untouched():
    """```hello``` has no newline separating a "tag" from content, so per
    CommonMark it is not a real fenced code block -- just three literal
    backticks. The first implementation's greedy \\w+ swallowed "hello" as
    a tag and returned an empty string; the correct behaviour is to leave
    it as unfenced content."""
    text = "```hello```"
    assert strip_code_fence(text) == text


def test_strip_code_fence_handles_a_closing_fence_glued_to_content():
    """No newline before the closing ``` (a fully single-line model
    reply) must still be recognised and stripped."""
    assert strip_code_fence('```json\n{"a":1}```') == '{"a":1}'


def test_strip_code_fence_tolerates_trailing_whitespace_on_delimiter_lines():
    assert strip_code_fence("```json   \n{}\n```   ") == "{}"


def test_strip_code_fence_only_strips_the_outer_wrap_around_two_blocks():
    """Two fenced blocks back to back: only the outermost open/close pair
    is a genuine wrap around the whole reply. The inner fence markers are
    content, not delimiters, and must survive."""
    text = "```json\n{}\n```\n```python\nx=1\n```"
    assert strip_code_fence(text) == "{}\n```\n```python\nx=1"


def test_strip_code_fence_is_a_noop_with_only_an_opening_marker():
    """A fence with no matching close is not a real wrap -- leave it
    alone rather than guessing where content should end."""
    text = "```json\n{\"a\": 1}"
    assert strip_code_fence(text) == text


def test_strip_code_fence_on_empty_string():
    assert strip_code_fence("") == ""


# ---------------------------------------------------------------------------
# compiler_node — wiring
# ---------------------------------------------------------------------------

def test_compiler_node_strips_a_fence_the_model_added_despite_instructions():
    """Regression for the exact live-trace symptom: a fallback provider
    ignored the "write Markdown, not JSON, no fence" instruction and wrapped
    its answer in ```json anyway. The fence must not leak into
    final_report."""
    router = _FakeRouter('```json\n{"title": "x", "findings": {}}\n```')
    node = build_compiler_node(router)
    state = ResearchState(raw_query="q")

    result = node(state)

    assert result["final_report"] == '{"title": "x", "findings": {}}'
    assert "```" not in result["final_report"]


def test_compiler_node_leaves_clean_markdown_unchanged():
    router = _FakeRouter("# Report\n\nRedis is fast. [g1 | corpus | score=0.90]")
    node = build_compiler_node(router)
    state = ResearchState(raw_query="q")

    result = node(state)

    assert result["final_report"] == "# Report\n\nRedis is fast. [g1 | corpus | score=0.90]"


def test_compiler_node_abort_path_is_unaffected_by_fence_stripping():
    """The two non-LLM report shapes (abort, planning_error) never call
    strip_code_fence at all — confirm the fix didn't touch that branch."""
    router = _FakeRouter("unused")
    node = build_compiler_node(router)
    state = ResearchState(raw_query="q", abort_reason="reviewer aborted")

    result = node(state)

    assert "aborted by human reviewer" in result["final_report"]
    assert router.node is None  # router.set_node was never called on this path

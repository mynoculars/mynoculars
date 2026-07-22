"""
prompts/templates.py — Every prompt the agent sends, in one place.

Purpose:
    Centralize prompt text so behavior tuning never requires hunting through
    node code, and so the stub client can key off the TASK= tags.

Responsibilities:
    - One builder function per LLM-using node, returning a chat transcript.
    - Keep every structured prompt's JSON schema next to its instructions.

Design decision (plain functions over a templating engine):
    f-strings are fully transparent to a learner and versionable by diff.
    Jinja adds nothing at this scale. The TASK=<name> tag on the first line
    is a deliberate contract with StubClient — live models ignore it.

Python mechanics used in this file, if any of this is new to you:
    f-strings:  f"...{some_variable}..."
        An f-string (the `f` prefix right before the opening quote) lets
        you embed a Python expression directly inside a string literal by
        wrapping it in curly braces — Python evaluates the expression and
        substitutes its value into the string at that point. Every prompt
        builder function below is essentially one big f-string assembling
        a message out of the function's arguments.
    "\\n".join(f"- {x}" for x in some_list) or "(none)"
        A very common pattern in this file: build one bullet-point line per
        item in a list (the "f'- {x}' for x in some_list" part is a
        GENERATOR EXPRESSION — like a list comprehension, but it doesn't
        build an intermediate list; .join() consumes it directly), stitch
        every line together with a newline between each, and if the
        original list was EMPTY (making the joined result an empty string,
        which is falsy in Python), fall back to the literal text "(none)"
        via the `or` operator instead of sending the model a blank section.
    List[Message]  (the return type on every function below)
        See llm/client.py for what "Message" means (a dict with "role" and
        "content" keys) — every function in this file returns a LIST of
        these, i.e. a full chat transcript ready to hand to
        router.complete(...) or router.complete_json(...).
"""

from typing import List

from research_agent.llm.client import Message
from research_agent.state import Evidence, Goal

# A single, shared system message reused by EVERY prompt builder below (see
# each function's return statement, which always starts its list with
# _SYSTEM). Defining it once here means every LLM call in this codebase
# gets the exact same baseline instruction, rather than each node writing
# its own slightly-different version.
_SYSTEM = {"role": "system", "content":
           "You are a precise research assistant. When asked for JSON, "
           "respond with ONLY the JSON object — no prose, no fences."}


def classify(query: str) -> List[Message]:
    """Intent classification. Schema: {"intent": str, "confidence": float}.

    CALLED BY   agents/planning.py::classify_node — the very first LLM
                call of every run.
    RETURNS     a 2-message transcript: the shared system message, then a
                user message containing the query and the exact JSON
                schema the model must reply with.
    """
    return [_SYSTEM, {"role": "user", "content":
            f"TASK=classify\nClassify the research intent of this query as one of: "
            f"Comparison, Survey, Explanation, Diagnosis, Recommendation.\n"
            f'Query: "{query}"\n'
            'JSON schema: {"intent": "<label>", "confidence": <0..1>}'}]


def compose_goals(query: str, intent: str, memory_hints: List[str],
                  guidance: str = "") -> List[Message]:
    """Goal composition. Schema: {"goals": [{"goal_id","description"}]}.

    CALLED BY   agents/planning.py::goal_manager_node.
    `guidance` carries a human redirect from an E1 escalation (D-23) —
    injected verbatim so the reviewer's intent is not paraphrased away.
    """
    # "\n".join(f"- {h}" for h in memory_hints) or "(none)" — see the module
    # docstring's explanation of this exact idiom. It turns a Python list of
    # strings into a bullet-point block, or the literal text "(none)" if the
    # list was empty.
    hints = "\n".join(f"- {h}" for h in memory_hints) or "(none)"
    # A conditional expression: "A if condition else B" evaluates to A when
    # the condition is true, and to B otherwise — a compact inline if/else.
    # Here: only build the "guidance" sentence at all if `guidance` is a
    # non-empty string; otherwise this becomes an empty string that
    # contributes nothing to the final prompt.
    steer = f"\nHuman reviewer guidance (follow it): {guidance}" if guidance else ""
    return [_SYSTEM, {"role": "user", "content":
            f"TASK=goals\nIntent: {intent}\nQuery: \"{query}\"\n"
            f"Relevant facts from earlier research (may inform goals):\n{hints}{steer}\n"
            f"Compose 2-5 concrete research goals that together answer the query. "
            'JSON schema: {"goals": [{"goal_id": "g1", "description": "..."}]}'}]


def expand_tasks(goals: List[Goal], max_tasks: int) -> List[Message]:
    """Initial task expansion (D-13: model ranks; code caps).

    CALLED BY   agents/planning.py::task_expander_node — the first pass,
                always at depth 0.
    """
    listing = "\n".join(f"- {g.goal_id}: {g.description}" for g in goals)
    return [_SYSTEM, {"role": "user", "content":
            f"TASK=expand\nGoals:\n{listing}\n"
            f"Produce at most {max_tasks} search queries covering these goals, "
            f"highest value first. "
            'JSON schema: {"tasks": [{"query": "...", "goal_id": "g1", "priority": <int>}]}'}]


def generate_gaps(goals: List[Goal], evidence: List[Evidence], depth: int,
                  max_tasks: int, guidance: str = "") -> List[Message]:
    """Gap analysis for uncovered goals. Same schema as expand_tasks.

    CALLED BY   agents/gathering.py::gap_generator_node — every gather-loop
                cycle after the first.
    """
    uncovered = "\n".join(f"- {g.goal_id}: {g.description}"
                          for g in goals if not g.covered) or "(none)"
    # evidence[-10:] is a SLICE meaning "the last 10 items of this list" —
    # negative indices count from the end in Python, so -10 is "10 items
    # before the end." Only the most recent evidence is shown to keep the
    # prompt from growing unboundedly as a run accumulates more and more
    # evidence over several gather-loop cycles.
    have = "\n".join(f"- [{e.goal_id}] {e.content[:120]}" for e in evidence[-10:]) or "(none)"
    steer = f"Human reviewer guidance (follow it): {guidance}\n" if guidance else ""
    return [_SYSTEM, {"role": "user", "content":
            f"TASK=gaps\nUncovered goals:\n{uncovered}\n"
            f"Evidence so far (tail):\n{have}\n"
            f"{steer}"
            f"Iteration depth: {depth}. Produce at most {max_tasks} NEW search "
            f"queries that would cover the uncovered goals, highest value first. "
            'JSON schema: {"tasks": [{"query": "...", "goal_id": "g1", "priority": <int>}]}'}]


def compile_report(query: str, goals: List[Goal], evidence: List[Evidence],
                   critique_notes: List[str]) -> List[Message]:
    """Report composition; grounded rewrite when critique notes exist (D-22).

    CALLED BY   agents/compilation.py::compiler_node — the ONLY prompt in
                this file whose call site expects free TEXT back (every
                other function here feeds into a complete_json() call).

    This function's job IS the "context construction" step of this
    project's RAG pipeline: every retrieved Evidence item is turned into
    one line of text and inlined below, with no truncation or re-ranking —
    everything gathered goes into the prompt.
    """
    ev = "\n".join(f"- [{e.goal_id} | {e.source} | score={e.score:.2f}] {e.content}"
                   for e in evidence) or "(no evidence gathered)"
    # A generator expression with an inline conditional inside the f-string
    # itself: for each goal, append the extra "[CONTESTED ...]" marker text
    # only if g.contested is True, otherwise append an empty string.
    gl = "\n".join(f"- {g.goal_id}: {g.description}"
                   f"{' [CONTESTED — present both positions]' if g.contested else ''}"
                   for g in goals)
    notes = ""
    if critique_notes:
        notes = ("\nA reviewer rejected the previous draft. Address every note:\n"
                 + "\n".join(f"- {n}" for n in critique_notes))
    return [_SYSTEM, {"role": "user", "content":
            f"TASK=compile\nWrite a well-structured Markdown research report.\n"
            f"Question: \"{query}\"\nGoals:\n{gl}\nEvidence:\n{ev}{notes}\n"
            f"Cite evidence by goal id. State clearly when evidence is thin."}]


def critique(query: str, report: str, goals: List[Goal]) -> List[Message]:
    """Report critique, scoped to faithfulness/completeness only (D-22).
    Schema: {"passed": bool, "score": float, "notes": [str]}.

    CALLED BY   agents/compilation.py::critic_node.
    Note the explicit instruction below telling the model NOT to judge
    whether more research was needed — that question belongs to a
    different node (progress_checker) entirely; see agents/gathering.py.
    """
    gl = "\n".join(f"- {g.goal_id}: {g.description}" for g in goals)
    return [_SYSTEM, {"role": "user", "content":
            f"TASK=critique\nQuestion: \"{query}\"\nGoals:\n{gl}\n"
            f"REPORT:\n{report}\n\n"
            f"Judge ONLY: (a) is the report faithful to its stated evidence, "
            f"(b) does it address every goal. Do NOT judge whether more research "
            f"was needed — that is a different system's job. "
            'JSON schema: {"passed": <bool>, "score": <0..1>, "notes": ["..."]}'}]

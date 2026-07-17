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
"""

from typing import List

from research_agent.llm.client import Message
from research_agent.state import Evidence, Goal

_SYSTEM = {"role": "system", "content":
           "You are a precise research assistant. When asked for JSON, "
           "respond with ONLY the JSON object — no prose, no fences."}


def classify(query: str) -> List[Message]:
    """Intent classification. Schema: {"intent": str, "confidence": float}."""
    return [_SYSTEM, {"role": "user", "content":
            f"TASK=classify\nClassify the research intent of this query as one of: "
            f"Comparison, Survey, Explanation, Diagnosis, Recommendation.\n"
            f'Query: "{query}"\n'
            'JSON schema: {"intent": "<label>", "confidence": <0..1>}'}]


def compose_goals(query: str, intent: str, memory_hints: List[str]) -> List[Message]:
    """Goal composition. Schema: {"goals": [{"goal_id","description"}]}."""
    hints = "\n".join(f"- {h}" for h in memory_hints) or "(none)"
    return [_SYSTEM, {"role": "user", "content":
            f"TASK=goals\nIntent: {intent}\nQuery: \"{query}\"\n"
            f"Relevant facts from earlier research (may inform goals):\n{hints}\n"
            f"Compose 2-5 concrete research goals that together answer the query. "
            'JSON schema: {"goals": [{"goal_id": "g1", "description": "..."}]}'}]


def expand_tasks(goals: List[Goal], max_tasks: int) -> List[Message]:
    """Initial task expansion (D-13: model ranks; code caps)."""
    listing = "\n".join(f"- {g.goal_id}: {g.description}" for g in goals)
    return [_SYSTEM, {"role": "user", "content":
            f"TASK=expand\nGoals:\n{listing}\n"
            f"Produce at most {max_tasks} search queries covering these goals, "
            f"highest value first. "
            'JSON schema: {"tasks": [{"query": "...", "goal_id": "g1", "priority": <int>}]}'}]


def generate_gaps(goals: List[Goal], evidence: List[Evidence], depth: int,
                  max_tasks: int) -> List[Message]:
    """Gap analysis for uncovered goals. Same schema as expand_tasks."""
    uncovered = "\n".join(f"- {g.goal_id}: {g.description}"
                          for g in goals if not g.covered) or "(none)"
    have = "\n".join(f"- [{e.goal_id}] {e.content[:120]}" for e in evidence[-10:]) or "(none)"
    return [_SYSTEM, {"role": "user", "content":
            f"TASK=gaps\nUncovered goals:\n{uncovered}\n"
            f"Evidence so far (tail):\n{have}\n"
            f"Iteration depth: {depth}. Produce at most {max_tasks} NEW search "
            f"queries that would cover the uncovered goals, highest value first. "
            'JSON schema: {"tasks": [{"query": "...", "goal_id": "g1", "priority": <int>}]}'}]


def compile_report(query: str, goals: List[Goal], evidence: List[Evidence],
                   critique_notes: List[str]) -> List[Message]:
    """Report composition; grounded rewrite when critique notes exist (D-22)."""
    ev = "\n".join(f"- [{e.goal_id} | {e.source} | score={e.score:.2f}] {e.content}"
                   for e in evidence) or "(no evidence gathered)"
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
    Schema: {"passed": bool, "score": float, "notes": [str]}."""
    gl = "\n".join(f"- {g.goal_id}: {g.description}" for g in goals)
    return [_SYSTEM, {"role": "user", "content":
            f"TASK=critique\nQuestion: \"{query}\"\nGoals:\n{gl}\n"
            f"REPORT:\n{report}\n\n"
            f"Judge ONLY: (a) is the report faithful to its stated evidence, "
            f"(b) does it address every goal. Do NOT judge whether more research "
            f"was needed — that is a different system's job. "
            'JSON schema: {"passed": <bool>, "score": <0..1>, "notes": ["..."]}'}]

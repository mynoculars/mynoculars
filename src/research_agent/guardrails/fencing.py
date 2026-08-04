"""
guardrails/fencing.py — the prompt-injection mitigation for retrieved
evidence text (D-18's defence-in-depth measure).

CALLED BY   prompts/templates.py — every builder that inlines retrieved
            Evidence text into a prompt (compile_report, generate_gaps,
            detect_contradictions).

Moved from prompts/templates.py, unchanged. The <evidence>...</evidence>
tags themselves are still opened by each caller in prompts/templates.py,
right next to the prompt-specific instruction text — only the delimiter-
neutralising helper moved here, since it's a generic string operation with
no dependency on which prompt is calling it.
"""


def fence_untrusted(text: str) -> str:
    """Neutralise the evidence delimiter inside retrieved content.

    WHY THIS EXISTS: Evidence.content comes from an ingested corpus or a
    third-party MCP server, i.e. from OUTSIDE this system's trust
    boundary, and was previously interpolated into prompts verbatim with
    no delimiter and no instruction to treat it as data. A single
    poisoned document could therefore address the model directly — most
    consequentially the contradiction detector, where "return an empty
    contested_goal_ids" defeats D-18 outright and leaves a genuinely
    contested goal looking covered.

    The <evidence> tags (opened by each caller, with this function
    stripping any the content itself contains) plus the shared system
    prompt (prompts/templates.py::_SYSTEM) are the standard RAG
    mitigation: mark the untrusted span, and state that spans so marked
    are never instructions. This is defence in depth, not a guarantee —
    no prompt-level measure is — but it closes the trivially exploitable
    version.
    """
    return (text.replace("<evidence>", "(evidence)")
                .replace("</evidence>", "(/evidence)"))

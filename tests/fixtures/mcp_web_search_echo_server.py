"""
tests/fixtures/mcp_web_search_echo_server.py -- A tiny REAL MCP server
returning canned web-search payloads, used only by
tests/unit/test_mcp_web_search_server.py to prove the WIRE SHAPE of a
`-> list[dict]` FastMCP tool over a real stdio connection.

Sibling of tests/fixtures/mcp_echo_server.py, which does the same job for
the corpus server's `-> list[str]` shape.

WHY A FIXTURE RATHER THAN THE REAL SERVER: scripts/mcp_web_search_server.py
would need `ddgs` installed AND would make a live network call on the first
tool invocation. This repo's suite is entirely offline (see
tests/conftest.py's module docstring), so the real server's own wrapping
logic is tested in-process against a fake provider, and the protocol-level
question -- "what does structuredContent actually look like for a list[dict]
return?" -- is answered here, deterministically, with no network and no
optional dependency beyond mcp itself.

The payload shape below is deliberately identical to
research_agent.websearch.provider.as_payload's output, because that is the
contract the agent side parses.

Not part of the installed package: a test fixture only, launched as a
subprocess by the test itself.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("web-search-echo-server")


@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Deterministic canned results so the calling test can assert exact
    content without depending on any external state.

    Scores descend across the returned set the way rank_to_score would, so a
    test asserting "the agent side preserves per-item ordering" has real
    ordering to preserve.
    """
    n = max(1, min(int(max_results), 3))
    band = [0.75, 0.675, 0.60]
    return [
        {
            "title": f"Result {i + 1} for {query}",
            "url": f"https://example{i + 1}.com/page",
            "snippet": f"canned snippet {i + 1} for: {query}",
            "rank": i + 1,
            "engine": "fixture",
            "domain": f"example{i + 1}.com",
            "score": band[i],
        }
        for i in range(n)
    ]


if __name__ == "__main__":
    mcp.run(transport="stdio")

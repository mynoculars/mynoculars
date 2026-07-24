"""
tests/fixtures/mcp_echo_server.py -- A tiny REAL MCP server, used only by
tests/test_tier3.py's P2-13 integration test to prove tools/mcp_client.py
actually works over a real stdio connection, not just against a mock.

Not part of the installed package or shipped to users -- a test fixture
only, launched as a subprocess by the test itself (see
test_mcp_tool_round_trips_through_a_real_stdio_server in test_tier3.py).
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("echo-test-server")


@mcp.tool()
def search(query: str) -> str:
    """A deterministic canned response so the calling test can assert
    exact content without depending on any external state."""
    return f"canned result for: {query}"


if __name__ == "__main__":
    mcp.run(transport="stdio")

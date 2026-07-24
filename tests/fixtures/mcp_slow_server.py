"""
tests/fixtures/mcp_slow_server.py -- A real MCP server whose tool sleeps
before responding, used only by tests/test_tier3.py to deterministically
trigger MCPBridge.call_tool's timeout path (rather than relying on a real
Qdrant/OpenSearch being slow, which isn't available/controllable in this
test environment).
"""

import time

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("slow-test-server")


@mcp.tool()
def search(query: str) -> str:
    time.sleep(5.0)  # comfortably longer than any short timeout a test uses
    return f"slow result for: {query}"


if __name__ == "__main__":
    mcp.run(transport="stdio")

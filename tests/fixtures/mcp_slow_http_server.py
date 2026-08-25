"""
tests/fixtures/mcp_slow_http_server.py -- D-76's HTTP counterpart to
mcp_slow_server.py, used only by
test_mcp_bridge_timeout_error_is_actually_informative to deterministically
trigger MCPBridge.call_tool's timeout path over a real HTTP connection
(rather than relying on a real Qdrant/OpenSearch being slow, which isn't
available/controllable in this test environment).

Not part of the installed package or shipped to users -- a test fixture
only, launched as a subprocess by the test itself, with --port supplied on
the command line so concurrent test runs don't collide on a fixed port.
"""

import argparse
import time

from mcp.server.fastmcp import FastMCP

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
args, _unknown = parser.parse_known_args()

mcp = FastMCP("slow-http-test-server", host="127.0.0.1", port=args.port)


@mcp.tool()
def search(query: str) -> str:
    time.sleep(5.0)  # comfortably longer than any short timeout a test uses
    return f"slow result for: {query}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")

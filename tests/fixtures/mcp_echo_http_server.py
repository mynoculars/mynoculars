"""
tests/fixtures/mcp_echo_http_server.py -- D-75's HTTP counterpart to
mcp_echo_server.py, used only by
test_mcp_tool_round_trips_through_a_real_streamable_http_server to prove
tools/mcp_client.py's streamable-http path works against a real server, not
just a mock.

Not part of the installed package or shipped to users -- a test fixture
only, launched as a subprocess by the test itself, with --port supplied on
the command line so concurrent test runs don't collide on a fixed port.
"""

import argparse

from mcp.server.fastmcp import FastMCP

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
args, _unknown = parser.parse_known_args()

mcp = FastMCP("echo-http-test-server", host="127.0.0.1", port=args.port)


@mcp.tool()
def search(query: str) -> str:
    """A deterministic canned response so the calling test can assert
    exact content without depending on any external state -- same
    contract as mcp_echo_server.py's stdio tool."""
    return f"canned result for: {query}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")

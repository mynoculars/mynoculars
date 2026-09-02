"""The standalone MCP servers this agent talks to (D-76, D-157).

Both are independent processes: nothing in this codebase ever spawns
one. Start a server yourself, leave it running, point one or more runs
at it, stop it whenever you like. They live here rather than in
`scripts/` so that an installed wheel actually contains the servers its
own documentation tells you to run (D-157) -- `scripts/` keeps a thin
launcher for each.
"""

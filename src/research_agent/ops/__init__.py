"""Operational commands: everything you RUN against a deployment (D-157).

Ingest, health check, store reset, run analysis, memory inspection and
garbage collection, the golden-set harness, and the offline pre-demo
gate. None of it is imported by the graph -- `research_agent.cli` and
`research_agent.api.server` never reach into this package -- which is
why it can depend on argparse, print to stdout, and exit with a code.

These used to live in `scripts/`, outside `src/`, and so appeared in no
wheel: `pip install research-agent[all]` gave you a package that could
not ingest a corpus or be health-checked. `scripts/` keeps a thin
launcher for each, so every documented `python scripts/<name>.py` still
works from a checkout.
"""

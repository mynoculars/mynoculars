"""
scripts/sanity.py -- launcher for research_agent.ops.sanity.

D-157: THE IMPLEMENTATION MOVED INTO THE PACKAGE. It used to live here,
in scripts/, which meant an installed wheel contained the offline pre-demo gate (D-158)
not at all -- `pip install research-agent[all]` produced something that
could not ingest a corpus, could not reach an MCP server, and could not
be health-checked, while pyproject.toml described a consumable package.

This file stays, and stays working, on purpose: every `python
scripts/sanity.py` in README.md and OPERATIONS.md is still correct, and a
checkout needs no install. The three lines below are the whole of it --
put the repo's own `src` on the path (so a checkout works without
PYTHONPATH being set), then hand straight over.

Equivalent, once the package is installed:
    (checkout only -- see the module docstring)
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from research_agent.ops.sanity import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

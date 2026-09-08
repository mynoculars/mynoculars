"""
scripts/package.py -- launcher for research_agent.ops.package.

D-190: builds the distributable archive from an explicit exclude list and
verifies the file that was actually written. The procedure it runs was
documented in README's Packaging section and was still skipped, which is
how an archive of this project shipped `.env` with five live credentials
in it -- the same class of failure scripts/sanity.py exists for, one step
later in the process.

Same launcher shape as every other command in scripts/ (D-157): put the
repo's own `src` on the path so a checkout works with no install and no
PYTHONPATH, then hand straight over.

Equivalent, once the package is installed:
    research-agent-package --dry-run
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from research_agent.ops.package import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

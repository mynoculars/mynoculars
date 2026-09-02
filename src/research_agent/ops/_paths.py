"""
ops/_paths.py -- where the repository is, when there is one (D-157).

Every module in this package used to sit in `scripts/` and could assume a
checkout: `pathlib.Path(__file__).parent.parent` WAS the repo root, and
the sample corpus, the golden set and the test suite were all a known
number of `..` away. Inside an installed wheel none of that is true --
there is no repo root, no `sample_data/`, and no `tests/`.

So the assumption becomes a question with two honest answers, and this
module is the one place that answers it. Callers branch on None rather
than each re-deriving a path that may not exist:

    - `ingest` and `eval_suite` fall back to a `--corpus` / `--golden`
      argument, which is what a real deployment passes anyway: nobody
      ingests THIS repo's ten Redis documents into their own system.
    - `sanity` refuses outright, because linting and testing a
      repository you do not have is not a thing that can be done.
"""

import pathlib
from typing import Optional


def repo_root() -> Optional[pathlib.Path]:
    """The checkout this package is running from, or None if installed.

    Walks up from this file looking for the two markers that only ever
    appear together at the root of THIS repository. Both are required:
    `pyproject.toml` alone would happily match a consuming project that
    has `research-agent` in its dependencies and its own pyproject at the
    top of ITS tree -- and then hand that project's directory back as
    though it were this one.

    Cheap enough to call per invocation (a handful of `.exists()` calls
    on a path already resolved), so it is deliberately not cached: a
    cached "no repository" answer that outlives the process that computed
    it would be a confusing thing to debug, and there is nothing to gain.
    """
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            return parent
    return None


def repo_file(*parts: str) -> Optional[pathlib.Path]:
    """A path INSIDE the checkout, or None when there is no checkout.

    Returns the path whether or not the file exists -- "the repository is
    not here" and "the file is missing from it" are different problems
    with different fixes, and a caller that conflates them reports the
    wrong one.
    """
    root = repo_root()
    return root.joinpath(*parts) if root is not None else None

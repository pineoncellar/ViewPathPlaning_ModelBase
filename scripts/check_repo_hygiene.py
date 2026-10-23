"""Refuse to let customer data become tracked.

Run this before every commit::

    uv run python scripts/check_repo_hygiene.py

Rationale: an ignored file can still be added later with ``git add -f``, but a
pushed one cannot be un-pushed.  So the check is deliberately paranoid and checks
the *index and HEAD*, not the working tree.

Why this exists at all: the customer hands over CAD of the part, the fixture and
the base, and everything derived from it carries real dimensions and coordinates.
That material lives under ``input/``, ``output/``, ``reference/00_Output Collection/``
and ``docs/superpowers/`` and must never be tracked.  The source code and the
public documents under ``docs/``, in contrast, are meant to be public.
"""

from __future__ import annotations

import subprocess
import sys

#: Patterns that must never appear among the tracked files.  Matching is on the
#: whole repo-relative path, forward slashes.
FORBIDDEN_SUBSTRINGS = (
    "数模/",
    "input/",
    "output/",
    "00_Output Collection/",
    "spike/",
    # private design notes quote customer dimensions, so they must not be tracked;
    # the rest of `docs/` is public documentation and is versioned on purpose
    "docs/superpowers/",
)

FORBIDDEN_SUFFIXES = (
    ".igs", ".iges", ".prt", ".stp", ".step", ".npy", ".npz",
)

#: The upstream repository ships public sample meshes; those are legitimate.
ALLOWED_STL_PREFIXES = ("samples/",)

#: Files tracked on purpose inside an otherwise ignored directory. They only
#: exist so the working directories survive a fresh clone and hold no data.
ALLOWED_DATA_DIR_MARKERS = (".gitkeep",)


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line.strip()]


def violations(paths: list[str]) -> list[str]:
    bad = []
    for path in paths:
        if path.endswith(ALLOWED_DATA_DIR_MARKERS):
            continue
        low = path.lower()
        if any(sub in path for sub in FORBIDDEN_SUBSTRINGS):
            bad.append(f"{path}  (inside a data directory)")
        elif low.endswith(FORBIDDEN_SUFFIXES):
            bad.append(f"{path}  (CAD / numerical dump suffix)")
        elif low.endswith(".stl") and not path.startswith(ALLOWED_STL_PREFIXES):
            bad.append(f"{path}  (mesh outside the public samples)")
    return bad


def main() -> int:
    paths = tracked_files()
    bad = violations(paths)
    print(f"tracked files: {len(paths)}")
    if not bad:
        print("OK - no customer data patterns found among the tracked files")
        return 0
    print("\nREFUSING: these tracked paths look like customer data:\n")
    for item in bad:
        print(f"  {item}")
    print(
        "\nRemove them from the index with:\n"
        "  git rm --cached <path>\n"
        "and make sure .gitignore covers them. If a file was already pushed, "
        "untracking it is not enough - the history has to be rewritten."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

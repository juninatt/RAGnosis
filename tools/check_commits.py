"""Check that commit subjects follow the project's convention.

The convention is Conventional Commits with a required scope, for example
``feat(ingest): fetch articles from the Wikipedia API``, kept to 72 characters
so it fits in one line of ``git log``. CI runs this on every push and pull
request; Dependabot's commits are set up to follow it too.

Usage: python tools/check_commits.py <revision range>, e.g. origin/main..HEAD
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence

TYPES = (
    "feat",
    "fix",
    "docs",
    "style",
    "refactor",
    "perf",
    "test",
    "build",
    "ci",
    "chore",
    "revert",
)
SUBJECT = re.compile(rf"({'|'.join(TYPES)})\([a-z0-9][a-z0-9-]*\)!?: \S.*")
MAX_LENGTH = 72
# A push that creates a branch reports this as the commit it came from.
NO_COMMIT = "0" * 40


def problems(subject: str) -> list[str]:
    """What is wrong with one commit subject; empty if nothing is."""
    if subject.startswith("Merge "):
        return []  # merge commits that GitHub writes itself
    found = []
    if not SUBJECT.fullmatch(subject):
        found.append("does not follow 'type(scope): description'")
    if len(subject) > MAX_LENGTH:
        found.append(f"is {len(subject)} characters; keep it to {MAX_LENGTH}")
    return found


def revision_range(base: str, head: str) -> str:
    """The range to check; a new branch has no base, so only its last commit is checked."""
    return f"{head}~1..{head}" if base in ("", NO_COMMIT) else f"{base}..{head}"


def subjects(revisions: str) -> list[tuple[str, str]]:
    """(short hash, subject) for each commit in ``revisions``."""
    out = subprocess.run(
        ["git", "log", "--format=%h%x00%s", revisions],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout
    commits = []
    for line in out.splitlines():
        commit, _, subject = line.partition("\0")
        commits.append((commit, subject))
    return commits


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("base", help="commit before the checked ones; empty or zeros for none")
    parser.add_argument("head", nargs="?", default="HEAD", help="last commit to check")
    args = parser.parse_args(argv)

    try:
        commits = subjects(revision_range(args.base, args.head))
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(f"check_commits: git log failed: {exc.stderr.strip()}\n")
        return 2

    failures = 0
    for commit, subject in commits:
        for problem in problems(subject):
            failures += 1
            sys.stderr.write(f"{commit} {subject!r} {problem}\n")
    sys.stderr.write(f"Checked {len(commits)} commit(s), {failures} problem(s)\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

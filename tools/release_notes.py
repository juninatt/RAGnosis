"""Release notes for a version tag, taken from CHANGELOG.md.

The release workflow runs this when a tag such as ``v0.5.0`` is pushed. It also
checks that the tag matches the version in pyproject.toml, so a release can
never be published for a version the code does not carry.

Usage: python tools/release_notes.py v0.5.0 --output notes.md
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path


class ReleaseError(Exception):
    """The tag, CHANGELOG.md or pyproject.toml is not ready for a release."""


def version_from_tag(tag: str) -> str:
    """``"v0.5.0"`` -> ``"0.5.0"``."""
    match = re.fullmatch(r"v(\d+\.\d+\.\d+)", tag)
    if match is None:
        raise ReleaseError(f"Tag {tag!r} is not of the form vX.Y.Z")
    return match.group(1)


def changelog_section(changelog: str, version: str) -> str:
    """The text under the heading for ``version``, up to the next heading or the link list."""
    heading = re.compile(rf"## \[?{re.escape(version)}\]?(\s|$)")
    lines = changelog.splitlines()
    start = next((i + 1 for i, line in enumerate(lines) if heading.match(line)), None)
    if start is None:
        raise ReleaseError(f"CHANGELOG.md has no section for {version}")

    body: list[str] = []
    for line in lines[start:]:
        if line.startswith("## ") or re.match(r"\[[^\]]+\]: ", line):
            break
        body.append(line)
    text = "\n".join(body).strip()
    if not text:
        raise ReleaseError(f"The CHANGELOG.md section for {version} is empty")
    return text + "\n"


def check_version(pyproject: str, version: str) -> None:
    """Fail unless pyproject.toml declares ``version``."""
    declared = tomllib.loads(pyproject)["project"]["version"]
    if declared != version:
        raise ReleaseError(f"pyproject.toml says version {declared}, the tag says {version}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tag", help="the release tag, e.g. v0.5.0")
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--output", type=Path, required=True, help="file to write the notes to")
    args = parser.parse_args(argv)

    try:
        version = version_from_tag(args.tag)
        check_version(args.pyproject.read_text(encoding="utf-8"), version)
        notes = changelog_section(args.changelog.read_text(encoding="utf-8"), version)
    except (ReleaseError, OSError) as exc:
        sys.stderr.write(f"release_notes: {exc}\n")
        return 1
    args.output.write_text(notes, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

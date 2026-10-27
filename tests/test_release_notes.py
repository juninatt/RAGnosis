"""Tests for tools/release_notes.py, including guards on the committed CHANGELOG."""

from __future__ import annotations

import runpy
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from release_notes import (  # noqa: E402
    ReleaseError,
    changelog_section,
    check_version,
    main,
    version_from_tag,
)

CHANGELOG = """# Changelog

Intro text.

## [Unreleased]

- Something new.

## [0.4.10] - 2026-11-01

- Ten.

## [0.4.1] - 2026-10-30

### Fixed

- One.

## 0.1.0 - 2026-10-23 (not tagged)

- First.

[0.4.1]: https://example.com/v0.4.1
"""

PYPROJECT = '[project]\nname = "ragnosis"\nversion = "0.4.1"\n'


# --- version_from_tag ---


@pytest.mark.parametrize(("tag", "version"), [("v0.5.0", "0.5.0"), ("v1.12.3", "1.12.3")])
def test_version_from_tag(tag: str, version: str) -> None:
    assert version_from_tag(tag) == version


@pytest.mark.parametrize("tag", ["0.5.0", "v0.5", "v0.5.0-rc1", "release-0.5.0", "v0.5.0 "])
def test_rejects_tags_that_are_not_vxyz(tag: str) -> None:
    with pytest.raises(ReleaseError, match="vX.Y.Z"):
        version_from_tag(tag)


# --- changelog_section ---


def test_section_runs_to_the_next_heading() -> None:
    assert changelog_section(CHANGELOG, "0.4.1") == "### Fixed\n\n- One.\n"


def test_similar_versions_are_not_confused() -> None:
    assert changelog_section(CHANGELOG, "0.4.10") == "- Ten.\n"


def test_last_section_stops_at_the_link_list() -> None:
    assert changelog_section(CHANGELOG, "0.1.0") == "- First.\n"


def test_section_at_the_end_of_a_file_without_links() -> None:
    assert changelog_section("# Changelog\n\n## [0.2.0]\n\n- Last.\n", "0.2.0") == "- Last.\n"


def test_missing_version_raises() -> None:
    with pytest.raises(ReleaseError, match="no section for 0.9.0"):
        changelog_section(CHANGELOG, "0.9.0")


def test_empty_section_raises() -> None:
    with pytest.raises(ReleaseError, match="empty"):
        changelog_section("## [0.2.0] - 2026-10-26\n\n## [0.1.0]\n- x\n", "0.2.0")


# --- check_version ---


def test_matching_version_passes() -> None:
    check_version(PYPROJECT, "0.4.1")


def test_mismatched_version_raises() -> None:
    with pytest.raises(ReleaseError, match="says version 0.4.1, the tag says 0.5.0"):
        check_version(PYPROJECT, "0.5.0")


# --- main ---


def write_project(tmp_path: Path) -> tuple[Path, Path]:
    changelog = tmp_path / "CHANGELOG.md"
    pyproject = tmp_path / "pyproject.toml"
    changelog.write_text(CHANGELOG, encoding="utf-8")
    pyproject.write_text(PYPROJECT, encoding="utf-8")
    return changelog, pyproject


def test_main_writes_the_notes(tmp_path: Path) -> None:
    changelog, pyproject = write_project(tmp_path)
    output = tmp_path / "notes.md"

    code = main(
        ["v0.4.1", "--changelog", str(changelog), "--pyproject", str(pyproject),
         "--output", str(output)]
    )  # fmt: skip

    assert code == 0
    assert output.read_text(encoding="utf-8") == "### Fixed\n\n- One.\n"


@pytest.mark.parametrize(
    ("tag", "message"),
    [("v0.5.0", "says version 0.4.1"), ("0.4.1", "vX.Y.Z")],
)
def test_main_reports_problems_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], tag: str, message: str
) -> None:
    changelog, pyproject = write_project(tmp_path)
    output = tmp_path / "notes.md"

    code = main(
        [tag, "--changelog", str(changelog), "--pyproject", str(pyproject),
         "--output", str(output)]
    )  # fmt: skip

    assert code == 1
    assert message in capsys.readouterr().err
    assert not output.exists()


def test_main_reports_a_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["v0.4.1", "--changelog", str(tmp_path / "missing.md"),
                 "--output", str(tmp_path / "notes.md")])  # fmt: skip
    assert code == 1
    assert "release_notes:" in capsys.readouterr().err


def test_runs_as_a_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["release_notes.py", "nope", "--output", "x.md"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(REPO_ROOT / "tools" / "release_notes.py"), run_name="__main__")
    assert exit_info.value.code == 1
    assert "vX.Y.Z" in capsys.readouterr().err


# --- Guard on the committed files ---


def test_committed_changelog_has_notes_for_the_current_version() -> None:
    # Releasing the version in pyproject.toml must find its notes.
    version = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    notes = changelog_section(
        (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), version["project"]["version"]
    )
    assert "### " in notes

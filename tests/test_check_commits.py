"""Tests for tools/check_commits.py, against a throwaway git repository."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from check_commits import (  # noqa: E402
    MAX_LENGTH,
    NO_COMMIT,
    main,
    problems,
    revision_range,
    subjects,
)

# --- problems ---


@pytest.mark.parametrize(
    "subject",
    [
        "feat(ingest): fetch articles from the Wikipedia API",
        "fix(answer): answer Swedish questions in Swedish",
        "build(deps-dev): bump the python-packages group with 3 updates",
        "ci(deps): bump actions/checkout from 4 to 5",
        "feat(cli)!: rename the ask command",
        "chore(release): 0.4.0",
        "docs(readme): " + "a" * (MAX_LENGTH - len("docs(readme): ")),  # exactly the limit
        "Merge pull request #12 from juninatt/dependabot/uv/python-packages",
    ],
)
def test_accepts_conventional_subjects(subject: str) -> None:
    assert problems(subject) == []


@pytest.mark.parametrize(
    "subject",
    [
        "fetch articles",  # no type
        "feat: fetch articles",  # no scope; this project requires one
        "Feat(ingest): fetch articles",  # capitalised type
        "feature(ingest): fetch articles",  # unknown type
        "feat(Ingest): fetch articles",  # capitalised scope
        "feat(ingest) fetch articles",  # no colon
        "feat(ingest):fetch articles",  # no space after the colon
        "feat(ingest): ",  # no description
        "feat(): fetch articles",  # empty scope
    ],
)
def test_rejects_other_subjects(subject: str) -> None:
    assert problems(subject) == ["does not follow 'type(scope): description'"]


def test_rejects_long_subjects() -> None:
    subject = "docs(readme): " + "a" * (MAX_LENGTH - len("docs(readme): ") + 1)
    assert problems(subject) == [f"is {MAX_LENGTH + 1} characters; keep it to {MAX_LENGTH}"]


def test_reports_every_problem_of_a_subject() -> None:
    assert len(problems("x" * (MAX_LENGTH + 5))) == 2


# --- revision_range ---


@pytest.mark.parametrize("base", ["", NO_COMMIT])
def test_new_branch_checks_only_its_last_commit(base: str) -> None:
    assert revision_range(base, "abc123") == "abc123~1..abc123"


def test_range_between_two_commits() -> None:
    assert revision_range("abc", "def") == "abc..def"


# --- Against a real git repository ---


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A git repository with three commits, the middle one badly named."""
    monkeypatch.chdir(tmp_path)

    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *args],
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    for subject in ["chore(project): start", "Added stuff", "feat(cli): add ask"]:
        git("commit", "-q", "--allow-empty", "-m", subject)
    return tmp_path


def git_hash(revision: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", revision], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_subjects_lists_commits_newest_first(repo: Path) -> None:
    assert [s for _, s in subjects("HEAD~2..HEAD")] == ["feat(cli): add ask", "Added stuff"]
    assert subjects("HEAD~1..HEAD")[0][0] == git_hash("HEAD")


def test_main_fails_and_names_the_bad_commit(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([git_hash("HEAD~2"), "HEAD"]) == 1
    err = capsys.readouterr().err
    assert f"{git_hash('HEAD~1')} 'Added stuff' does not follow" in err
    assert "Checked 2 commit(s), 1 problem(s)" in err


def test_main_passes_good_commits(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([git_hash("HEAD~1")]) == 0
    assert "Checked 1 commit(s), 0 problem(s)" in capsys.readouterr().err


def test_main_on_a_new_branch_checks_the_last_commit(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([NO_COMMIT, "HEAD"]) == 0
    assert "Checked 1 commit(s)" in capsys.readouterr().err


def test_main_reports_git_errors(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["no-such-commit"]) == 2
    assert "git log failed" in capsys.readouterr().err


def test_runs_as_a_script(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["check_commits.py", git_hash("HEAD~1")])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(REPO_ROOT / "tools" / "check_commits.py"), run_name="__main__")
    assert exit_info.value.code == 0


# --- Guard on this project's own history ---


def test_this_projects_recent_history_follows_the_convention() -> None:
    # CI checks out one commit; skip when there is no history to look at.
    depth = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-list", "--count", "HEAD"],
        capture_output=True,
        text=True,
    )
    if depth.returncode != 0 or int(depth.stdout) < 2:
        pytest.skip("shallow checkout")
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "log", "--format=%s", "-n", "50"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout
    bad = [s for s in out.splitlines() if problems(s)]
    assert bad == []

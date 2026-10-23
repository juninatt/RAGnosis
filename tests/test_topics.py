"""Tests for reading title lists, including the committed one."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragnosis.settings import DEFAULT_TOPICS_FILE
from ragnosis.topics import load_titles, normalize_title

REPO_ROOT = Path(__file__).resolve().parents[1]


def write_topics(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "topics.txt"
    path.write_text(content, encoding="utf-8")
    return path


def test_reads_one_title_per_line_in_order(tmp_path: Path) -> None:
    path = write_topics(tmp_path, "Blackbeard\nAnne Bonny\nMary Read\n")
    assert load_titles(path) == ["Blackbeard", "Anne Bonny", "Mary Read"]


def test_ignores_comments_and_blank_lines(tmp_path: Path) -> None:
    content = "# Pirates\n\nBlackbeard\n   \n# Rome\nAugustus  # first emperor\n"
    assert load_titles(write_topics(tmp_path, content)) == ["Blackbeard", "Augustus"]


def test_drops_duplicates_keeping_first_occurrence(tmp_path: Path) -> None:
    content = "Blackbeard\nAnne Bonny\nBlackbeard\nAnne_Bonny\n"
    assert load_titles(write_topics(tmp_path, content)) == ["Blackbeard", "Anne Bonny"]


def test_empty_file_gives_no_titles(tmp_path: Path) -> None:
    assert load_titles(write_topics(tmp_path, "")) == []


def test_handles_windows_line_endings_and_non_ascii(tmp_path: Path) -> None:
    content = "Klaus Störtebeker\r\nOruç Reis\r\n"
    assert load_titles(write_topics(tmp_path, content)) == ["Klaus Störtebeker", "Oruç Reis"]


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_titles(tmp_path / "missing.txt")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Anne_Bonny", "Anne Bonny"),
        ("  Anne   Bonny  ", "Anne Bonny"),
        ("Anne\tBonny", "Anne Bonny"),
        ("anne bonny", "anne bonny"),
    ],
)
def test_normalize_title(raw: str, expected: str) -> None:
    assert normalize_title(raw) == expected


# Guards for the committed title list.


def test_committed_topics_file_has_titles() -> None:
    assert len(load_titles(REPO_ROOT / DEFAULT_TOPICS_FILE)) >= 20


def test_committed_topics_file_has_no_accidental_duplicates() -> None:
    path = REPO_ROOT / DEFAULT_TOPICS_FILE
    lines = [
        normalize_title(line.split("#", 1)[0])
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    titles = [line for line in lines if line]
    assert len(titles) == len(set(titles))

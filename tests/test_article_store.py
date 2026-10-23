"""Tests for storing articles as JSON files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from factories import make_article

from ragnosis.article_store import ArticleStore


@pytest.fixture
def store(tmp_path: Path) -> ArticleStore:
    return ArticleStore(tmp_path / "articles")


def test_save_then_load_round_trips(store: ArticleStore) -> None:
    article = make_article()
    store.save("Blackbeard", article)
    assert store.load("Blackbeard") == article


def test_contains_reflects_saved_titles(store: ArticleStore) -> None:
    assert not store.contains("Blackbeard")
    store.save("Blackbeard", make_article())
    assert store.contains("Blackbeard")


def test_creates_missing_directory(store: ArticleStore) -> None:
    assert not store.directory.exists()
    store.save("Blackbeard", make_article())
    assert store.directory.is_dir()


def test_stored_under_requested_title_not_resolved_title(store: ArticleStore) -> None:
    store.save("Edward Teach", make_article(title="Blackbeard"))
    assert store.contains("Edward Teach")
    assert not store.contains("Blackbeard")


@pytest.mark.parametrize(
    "title",
    [
        "AC/DC",
        "What?",
        "Siege of Rhodes (305–304 BC)",
        "Re: Zero",
        'The "Quoted" Pirate',
        "Back\\slash*star<>|",
        "Grace O'Malley",
        "Klaus Störtebeker",
        "..",
        "CON",
    ],
)
def test_any_title_maps_to_a_file_inside_the_directory(store: ArticleStore, title: str) -> None:
    path = store.path_for(title)
    assert path.parent == store.directory

    article = make_article(title=title)
    store.save(title, article)
    assert store.load(title) == article


def test_titles_differing_only_by_unsafe_characters_get_different_files(
    store: ArticleStore,
) -> None:
    assert store.path_for("AC/DC") != store.path_for("AC_DC")
    assert store.path_for("AC/DC") != store.path_for("AC?DC")


def test_spaces_become_underscores_like_wikipedia_urls(store: ArticleStore) -> None:
    assert store.path_for("Anne Bonny").name == "Anne_Bonny.json"


def test_file_is_readable_utf8_json(store: ArticleStore) -> None:
    path = store.save("Klaus Störtebeker", make_article(text="Störtebeker – Vitalienbrüder"))
    raw = path.read_text(encoding="utf-8")
    assert "Störtebeker – Vitalienbrüder" in raw
    assert json.loads(raw)["text"] == "Störtebeker – Vitalienbrüder"


def test_save_overwrites_earlier_version(store: ArticleStore) -> None:
    store.save("Blackbeard", make_article(revision_id=1))
    store.save("Blackbeard", make_article(revision_id=2))
    assert store.load("Blackbeard").revision_id == 2


def test_leaves_no_temporary_files(store: ArticleStore) -> None:
    store.save("Blackbeard", make_article())
    assert [path.name for path in store.directory.iterdir()] == ["Blackbeard.json"]


def test_load_missing_title_raises(store: ArticleStore) -> None:
    with pytest.raises(FileNotFoundError):
        store.load("Nobody")

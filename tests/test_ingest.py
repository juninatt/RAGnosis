"""Tests for the ingest run, with a fake article source and a real store on disk."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from factories import make_article

from ragnosis.article_store import ArticleStore
from ragnosis.ingest import IngestReport, ingest
from ragnosis.wikidata import WikidataError
from ragnosis.wikipedia import Article, ArticleNotFoundError, WikipediaError


class FakeSource:
    """Returns prepared articles or raises prepared errors, and records every request."""

    def __init__(self, outcomes: dict[str, Article | Exception]) -> None:
        self.outcomes = outcomes
        self.requested: list[str] = []

    def fetch_article(self, title: str) -> Article:
        self.requested.append(title)
        outcome = self.outcomes[title]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def store(tmp_path: Path) -> ArticleStore:
    return ArticleStore(tmp_path / "articles")


def test_fetches_and_stores_every_title(store: ArticleStore) -> None:
    source = FakeSource(
        {
            "Blackbeard": make_article(title="Blackbeard", page_id=1),
            "Anne Bonny": make_article(title="Anne Bonny", page_id=2),
        }
    )
    report = ingest(["Blackbeard", "Anne Bonny"], source, store)

    assert report == IngestReport(fetched=["Blackbeard", "Anne Bonny"])
    assert store.load("Anne Bonny").page_id == 2


def test_skips_titles_already_stored(store: ArticleStore) -> None:
    store.save("Blackbeard", make_article(page_id=1))
    source = FakeSource({"Anne Bonny": make_article(title="Anne Bonny", page_id=2)})

    report = ingest(["Blackbeard", "Anne Bonny"], source, store)

    assert source.requested == ["Anne Bonny"]
    assert report.skipped == ["Blackbeard"]
    assert report.fetched == ["Anne Bonny"]


def test_refresh_fetches_stored_titles_again(store: ArticleStore) -> None:
    store.save("Blackbeard", make_article(revision_id=1))
    source = FakeSource({"Blackbeard": make_article(revision_id=2)})

    report = ingest(["Blackbeard"], source, store, refresh=True)

    assert report.fetched == ["Blackbeard"]
    assert store.load("Blackbeard").revision_id == 2


@pytest.mark.parametrize(
    "error",
    [ArticleNotFoundError("No such article"), WikipediaError("Giving up after 4 attempts")],
    ids=["not-found", "api-failure"],
)
def test_failure_is_recorded_and_run_continues(
    store: ArticleStore, error: WikipediaError, caplog: pytest.LogCaptureFixture
) -> None:
    source = FakeSource({"Bad": error, "Blackbeard": make_article()})

    report = ingest(["Bad", "Blackbeard"], source, store)

    assert report.failed == {"Bad": str(error)}
    assert report.fetched == ["Blackbeard"]
    assert not report.ok
    assert not store.contains("Bad")
    assert "Could not fetch 'Bad'" in caplog.text


def test_unexpected_errors_are_not_swallowed(store: ArticleStore) -> None:
    source = FakeSource({"Blackbeard": KeyError("bug")})
    with pytest.raises(KeyError):
        ingest(["Blackbeard"], source, store)


def test_titles_resolving_to_the_same_article_are_stored_once(store: ArticleStore) -> None:
    blackbeard = make_article(title="Blackbeard", page_id=1)
    source = FakeSource({"Blackbeard": blackbeard, "Edward Teach": blackbeard})

    report = ingest(["Blackbeard", "Edward Teach"], source, store)

    assert report.fetched == ["Blackbeard"]
    assert report.duplicates == ["Edward Teach"]
    assert not store.contains("Edward Teach")
    assert report.ok


# --- Other names ---


class FakeNames:
    """Returns prepared names, or raises a prepared error, and records every request."""

    def __init__(self, names: dict[str, list[str]], error: WikidataError | None = None) -> None:
        self.names = names
        self.error = error
        self.requested: list[list[str]] = []

    def fetch_names(self, titles: Sequence[str]) -> dict[str, list[str]]:
        self.requested.append(list(titles))
        if self.error is not None:
            raise self.error
        return {title: self.names.get(title, []) for title in titles}


def test_stores_other_names_for_fetched_articles(store: ArticleStore) -> None:
    source = FakeSource({"Blackbeard": make_article(title="Blackbeard")})
    names = FakeNames({"Blackbeard": ["Svartskägg", "Edward Teach"]})

    report = ingest(["Blackbeard"], source, store, names=names)

    assert store.load("Blackbeard").names == ("Svartskägg", "Edward Teach")
    assert report.named == ["Blackbeard"]
    assert report.ok


def test_asks_by_the_article_title_and_stores_under_the_requested_one(
    store: ArticleStore,
) -> None:
    source = FakeSource({"Calico Jack": make_article(title="John Rackham", page_id=7)})
    names = FakeNames({"John Rackham": ["Calico Jack"]})

    ingest(["Calico Jack"], source, store, names=names)

    assert names.requested == [["John Rackham"]]
    assert store.load("Calico Jack").names == ("Calico Jack",)


def test_gives_stored_articles_without_names_their_names(store: ArticleStore) -> None:
    store.save("Blackbeard", make_article())  # stored before 1.2.0: no names
    source = FakeSource({})
    names = FakeNames({"Blackbeard": ["Svartskägg"]})

    report = ingest(["Blackbeard"], source, store, names=names)

    assert source.requested == []
    assert report.skipped == ["Blackbeard"]
    assert report.named == ["Blackbeard"]
    assert store.load("Blackbeard").names == ("Svartskägg",)


def test_does_not_ask_again_for_articles_that_have_names(store: ArticleStore) -> None:
    store.save("Blackbeard", make_article(names=()))  # asked before, Wikidata had none
    names = FakeNames({"Blackbeard": ["Svartskägg"]})

    report = ingest(["Blackbeard"], FakeSource({}), store, names=names)

    assert names.requested == []
    assert report.named == []
    assert store.load("Blackbeard").names == ()


def test_keeps_articles_without_names_when_wikidata_fails(
    store: ArticleStore, caplog: pytest.LogCaptureFixture
) -> None:
    source = FakeSource({"Blackbeard": make_article()})
    names = FakeNames({}, error=WikidataError("Giving up after 4 attempts"))

    report = ingest(["Blackbeard"], source, store, names=names)

    assert report.fetched == ["Blackbeard"]
    assert store.load("Blackbeard").names is None
    assert report.names_failed == "Giving up after 4 attempts"
    assert not report.ok
    assert "the next run retries" in caplog.text


def test_names_are_not_fetched_for_failed_or_duplicate_titles(store: ArticleStore) -> None:
    blackbeard = make_article(title="Blackbeard", page_id=1)
    source = FakeSource(
        {"Blackbeard": blackbeard, "Edward Teach": blackbeard, "Bad": ArticleNotFoundError("x")}
    )
    names = FakeNames({})

    ingest(["Blackbeard", "Edward Teach", "Bad"], source, store, names=names)

    assert names.requested == [["Blackbeard"]]


def test_empty_title_list_does_nothing(store: ArticleStore) -> None:
    report = ingest([], FakeSource({}), store)
    assert report == IngestReport()
    assert report.ok
    assert not store.directory.exists()

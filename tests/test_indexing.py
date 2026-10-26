"""Tests for keeping the vector index in line with the stored articles."""

from __future__ import annotations

from pathlib import Path

import pytest
from factories import FakeEmbedder, make_article

from ragnosis.article_store import ArticleStore
from ragnosis.indexing import IndexReport, update_index
from ragnosis.vector_store import VectorStore

BLACKBEARD = make_article(
    title="Blackbeard",
    page_id=1,
    revision_id=100,
    text="Teach was a feared pirate.\n== Death ==\nHe was killed at Ocracoke in 1718.",
)
HANNIBAL = make_article(
    title="Hannibal",
    page_id=2,
    revision_id=200,
    url="https://en.wikipedia.org/wiki/Hannibal",
    text="Hannibal led Carthage against Rome and crossed the Alps with elephants.",
)


@pytest.fixture
def articles(tmp_path: Path) -> ArticleStore:
    store = ArticleStore(tmp_path / "articles")
    store.save("Blackbeard", BLACKBEARD)
    store.save("Hannibal", HANNIBAL)
    return store


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store(embedder: FakeEmbedder) -> VectorStore:
    return VectorStore(embedder.model_name, embedder.dimension)


def test_indexes_every_chunk_of_every_listed_article(
    articles: ArticleStore, store: VectorStore, embedder: FakeEmbedder
) -> None:
    report = update_index(["Blackbeard", "Hannibal"], articles, store, embedder)

    assert report == IndexReport(indexed=["Blackbeard", "Hannibal"], chunks_added=3)
    assert len(store) == 3
    assert store.revision_of(1) == 100
    assert store.revision_of(2) == 200


def test_embeds_text_with_article_and_section_header(
    articles: ArticleStore, store: VectorStore, embedder: FakeEmbedder
) -> None:
    update_index(["Blackbeard"], articles, store, embedder)
    assert embedder.document_calls == [
        [
            "Blackbeard > Introduction\nTeach was a feared pirate.",
            "Blackbeard > Death\nHe was killed at Ocracoke in 1718.",
        ]
    ]


def test_indexed_chunks_are_searchable(
    articles: ArticleStore, store: VectorStore, embedder: FakeEmbedder
) -> None:
    update_index(["Blackbeard", "Hannibal"], articles, store, embedder)
    best = store.search(embedder.embed_query("Alps elephants Carthage"), k=1)[0]
    assert best.chunk.article_title == "Hannibal"


def test_unchanged_articles_are_not_embedded_again(
    articles: ArticleStore, store: VectorStore, embedder: FakeEmbedder
) -> None:
    update_index(["Blackbeard", "Hannibal"], articles, store, embedder)
    embedder.document_calls.clear()

    report = update_index(["Blackbeard", "Hannibal"], articles, store, embedder)

    assert report.unchanged == ["Blackbeard", "Hannibal"]
    assert report.indexed == []
    assert embedder.document_calls == []
    assert len(store) == 3


def test_new_revision_is_reindexed_in_place(
    articles: ArticleStore, store: VectorStore, embedder: FakeEmbedder
) -> None:
    update_index(["Blackbeard"], articles, store, embedder)
    articles.save("Blackbeard", make_article(page_id=1, revision_id=101, text="Rewritten."))

    report = update_index(["Blackbeard"], articles, store, embedder)

    assert report.indexed == ["Blackbeard"]
    assert store.revision_of(1) == 101
    assert len(store) == 1


def test_titles_without_stored_article_are_reported_missing(
    articles: ArticleStore, store: VectorStore, embedder: FakeEmbedder
) -> None:
    report = update_index(["Blackbeard", "Anne Bonny"], articles, store, embedder)
    assert report.missing == ["Anne Bonny"]
    assert report.indexed == ["Blackbeard"]
    assert not report.ok


def test_articles_dropped_from_the_list_are_removed(
    articles: ArticleStore, store: VectorStore, embedder: FakeEmbedder
) -> None:
    update_index(["Blackbeard", "Hannibal"], articles, store, embedder)

    report = update_index(["Blackbeard"], articles, store, embedder)

    assert report.removed == [2]
    assert store.page_ids == {1}
    assert all(r.chunk.page_id == 1 for r in store.search(embedder.vector("Hannibal"), k=10))


def test_article_with_only_skipped_sections_is_recorded_without_chunks(
    tmp_path: Path, store: VectorStore, embedder: FakeEmbedder
) -> None:
    articles = ArticleStore(tmp_path)
    articles.save("Stub", make_article(page_id=9, text="== References ==\nSmith 2001."))

    report = update_index(["Stub"], articles, store, embedder)

    assert report.indexed == ["Stub"]
    assert report.chunks_added == 0
    assert store.revision_of(9) == 100


def test_refuses_store_built_with_another_model(
    articles: ArticleStore, embedder: FakeEmbedder
) -> None:
    store = VectorStore("other-model", embedder.dimension)
    with pytest.raises(ValueError, match="other-model"):
        update_index(["Blackbeard"], articles, store, embedder)

"""Tests for the numpy vector store: adding, searching, saving and loading."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from factories import make_chunk
from hypothesis import given
from hypothesis import strategies as st

from ragnosis.chunking import Chunk
from ragnosis.embeddings import Vectors
from ragnosis.vector_store import (
    CHUNKS_FILE,
    MANIFEST_FILE,
    VECTORS_FILE,
    CorruptIndexError,
    VectorStore,
)


def unit(*values: float) -> Vectors:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def chunks_of(page_id: int, count: int, title: str = "Article") -> list[Chunk]:
    return [
        make_chunk(
            chunk_id=f"{page_id}-{i}",
            page_id=page_id,
            article_title=title,
            position=i,
            text=f"{title} passage {i}",
        )
        for i in range(count)
    ]


@pytest.fixture
def store() -> VectorStore:
    """Three articles in a 3-dimensional space, one per axis."""
    store = VectorStore("fake-model", 3)
    store.replace_article(1, 10, chunks_of(1, 1, "Blackbeard"), np.array([unit(1, 0, 0)]))
    store.replace_article(2, 20, chunks_of(2, 1, "Anne Bonny"), np.array([unit(0, 1, 0)]))
    store.replace_article(3, 30, chunks_of(3, 1, "Hannibal"), np.array([unit(0, 0, 1)]))
    return store


# --- Adding and removing ---


def test_new_store_is_empty() -> None:
    store = VectorStore("fake-model", 3)
    assert len(store) == 0
    assert store.page_ids == set()
    assert store.search(unit(1, 0, 0)) == []


def test_tracks_articles_and_revisions(store: VectorStore) -> None:
    assert len(store) == 3
    assert store.page_ids == {1, 2, 3}
    assert store.revision_of(2) == 20
    assert store.revision_of(99) is None


def test_replacing_an_article_does_not_duplicate_it(store: VectorStore) -> None:
    store.replace_article(1, 11, chunks_of(1, 2, "Blackbeard"), np.array([unit(1, 1, 0)] * 2))
    assert len(store) == 4
    assert store.revision_of(1) == 11
    titles = [r.chunk.article_title for r in store.search(unit(1, 0, 0), k=10)]
    assert titles.count("Blackbeard") == 2


def test_article_without_chunks_is_still_recorded() -> None:
    store = VectorStore("fake-model", 3)
    store.replace_article(5, 50, [], np.empty((0, 3), dtype=np.float32))
    assert len(store) == 0
    assert store.revision_of(5) == 50


def test_remove_article(store: VectorStore) -> None:
    store.remove_article(2)
    assert store.page_ids == {1, 3}
    assert [r.chunk.page_id for r in store.search(unit(0, 1, 0), k=10)] == [1, 3]


def test_removing_unknown_article_does_nothing(store: VectorStore) -> None:
    store.remove_article(99)
    assert len(store) == 3


def test_rejects_vectors_of_wrong_shape(store: VectorStore) -> None:
    with pytest.raises(ValueError, match="shape"):
        store.replace_article(4, 40, chunks_of(4, 2), np.array([unit(1, 0, 0)]))
    with pytest.raises(ValueError, match="shape"):
        store.replace_article(4, 40, chunks_of(4, 1), np.ones((1, 4), dtype=np.float32))


def test_rejects_chunks_of_another_page(store: VectorStore) -> None:
    with pytest.raises(ValueError, match="page 4"):
        store.replace_article(4, 40, chunks_of(5, 1), np.array([unit(1, 0, 0)]))


def test_rejects_zero_dimension() -> None:
    with pytest.raises(ValueError, match="dimension"):
        VectorStore("fake-model", 0)


# --- Search ---


def test_search_ranks_by_similarity(store: VectorStore) -> None:
    results = store.search(unit(0.1, 0.9, 0.3), k=3)
    assert [r.chunk.article_title for r in results] == ["Anne Bonny", "Hannibal", "Blackbeard"]
    assert results[0].score == pytest.approx(float(unit(0.1, 0.9, 0.3) @ unit(0, 1, 0)))


def test_search_returns_at_most_k(store: VectorStore) -> None:
    assert len(store.search(unit(1, 0, 0), k=2)) == 2
    assert len(store.search(unit(1, 0, 0), k=10)) == 3


def test_equal_scores_keep_insertion_order() -> None:
    store = VectorStore("fake-model", 2)
    store.replace_article(1, 1, chunks_of(1, 3), np.array([unit(1, 0)] * 3))
    assert [r.chunk.position for r in store.search(unit(1, 0), k=3)] == [0, 1, 2]


@pytest.mark.parametrize("k", [0, -1])
def test_rejects_k_below_one(store: VectorStore, k: int) -> None:
    with pytest.raises(ValueError, match="k must be"):
        store.search(unit(1, 0, 0), k=k)


def test_rejects_query_of_wrong_dimension(store: VectorStore) -> None:
    with pytest.raises(ValueError, match="query of shape"):
        store.search(unit(1, 0))


@given(st.integers(0, 2**32 - 1), st.integers(1, 40), st.integers(1, 10))
def test_search_matches_brute_force_ranking(seed: int, rows: int, k: int) -> None:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(rows, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query = vectors[0] + rng.normal(scale=0.5, size=8).astype(np.float32)
    query /= np.linalg.norm(query)

    store = VectorStore("fake-model", 8)
    store.replace_article(1, 1, chunks_of(1, rows), vectors)
    results = store.search(query, k=k)

    expected = sorted((float(v @ query) for v in vectors), reverse=True)[:k]
    assert [r.score for r in results] == pytest.approx(expected, abs=1e-5)


# --- Saving and loading ---


def test_save_then_load_round_trips(store: VectorStore, tmp_path: Path) -> None:
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)

    assert loaded.embedding_model == "fake-model"
    assert loaded.dimension == 3
    assert loaded.page_ids == {1, 2, 3}
    assert loaded.revision_of(3) == 30
    query = unit(0.2, 0.3, 0.9)
    assert loaded.search(query, k=3) == store.search(query, k=3)


def test_round_trip_keeps_non_ascii_text(tmp_path: Path) -> None:
    store = VectorStore("fake-model", 2)
    chunk = make_chunk(article_title="Klaus Störtebeker", text="Ἀλέξανδρος – Vitalienbrüder")
    store.replace_article(1, 1, [chunk], np.array([unit(1, 0)]))
    store.save(tmp_path)
    assert VectorStore.load(tmp_path).search(unit(1, 0))[0].chunk == chunk


def test_save_replaces_earlier_save_and_leaves_no_temp_files(
    store: VectorStore, tmp_path: Path
) -> None:
    store.save(tmp_path)
    store.remove_article(1)
    store.save(tmp_path)

    assert VectorStore.load(tmp_path).page_ids == {2, 3}
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [MANIFEST_FILE, VECTORS_FILE, CHUNKS_FILE]
    )


def test_exists_only_after_save(store: VectorStore, tmp_path: Path) -> None:
    assert not VectorStore.exists(tmp_path)
    store.save(tmp_path / "index")
    assert VectorStore.exists(tmp_path / "index")


def test_load_without_index_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        VectorStore.load(tmp_path)


def edit_manifest(**changes: object) -> Callable[[Path], None]:
    def corrupt(directory: Path) -> None:
        path = directory / MANIFEST_FILE
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.update(changes)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    return corrupt


def overwrite(name: str, content: str) -> Callable[[Path], None]:
    def corrupt(directory: Path) -> None:
        (directory / name).write_text(content, encoding="utf-8")

    return corrupt


def delete(name: str) -> Callable[[Path], None]:
    return lambda directory: (directory / name).unlink()


@pytest.mark.parametrize(
    "corrupt",
    [
        overwrite(MANIFEST_FILE, "{not json"),
        overwrite(MANIFEST_FILE, "[]"),
        edit_manifest(format_version=99),
        edit_manifest(chunk_count=2),
        edit_manifest(dimension=4),
        edit_manifest(articles={"1": 10, "2": 20}),
        edit_manifest(articles="not a mapping"),
        delete(VECTORS_FILE),
        delete(CHUNKS_FILE),
        overwrite(VECTORS_FILE, "not numpy"),
        overwrite(CHUNKS_FILE, '{"chunk_id": "1-0"}\n'),
        overwrite(CHUNKS_FILE, "not json\n"),
    ],
    ids=[
        "manifest-not-json",
        "manifest-not-object",
        "unknown-format",
        "count-mismatch",
        "dimension-mismatch",
        "unlisted-article",
        "articles-not-mapping",
        "vectors-missing",
        "chunks-missing",
        "vectors-not-numpy",
        "chunk-missing-fields",
        "chunk-not-json",
    ],
)
def test_inconsistent_index_is_reported_as_corrupt(
    store: VectorStore, tmp_path: Path, corrupt: Callable[[Path], None]
) -> None:
    store.save(tmp_path)
    corrupt(tmp_path)
    with pytest.raises(CorruptIndexError):
        VectorStore.load(tmp_path)

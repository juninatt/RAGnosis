"""A small vector store: chunk vectors in a numpy file, chunk metadata in JSON lines.

Search compares the query with every stored vector. For a few thousand chunks
that takes well under a millisecond, so an approximate index (what databases
such as ChromaDB add) would only bring dependencies (see docs/decisions/0005).

On disk, in the index folder:
- ``vectors.npy``: one float32 row per chunk,
- ``chunks.jsonl``: the matching chunks, one JSON object per line, same order,
- ``manifest.json``: embedding model, dimension, chunk count and the revision
  of each indexed article. Written last, so it marks a completed save.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ragnosis.chunking import Chunk
from ragnosis.embeddings import Vectors

FORMAT_VERSION = 1
MANIFEST_FILE = "manifest.json"
VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.jsonl"


class CorruptIndexError(Exception):
    """The files on disk do not form a consistent index; rebuilding fixes it."""


@dataclass(frozen=True)
class SearchResult:
    """A chunk and how similar it is to the query (cosine similarity, -1 to 1)."""

    chunk: Chunk
    score: float


class VectorStore:
    """Chunks with their vectors, tagged with the model that produced the vectors.

    Vectors from different models are not comparable, so a store only ever holds
    vectors from one model, and callers check ``embedding_model`` before adding.
    """

    def __init__(self, embedding_model: str, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("dimension must be at least 1")
        self.embedding_model = embedding_model
        self.dimension = dimension
        self._chunks: list[Chunk] = []
        self._vectors: Vectors = np.empty((0, dimension), dtype=np.float32)
        self._revisions: dict[int, int] = {}

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def page_ids(self) -> set[int]:
        """Articles that are in the index."""
        return set(self._revisions)

    def revision_of(self, page_id: int) -> int | None:
        """The article revision that was indexed, or None if the article is not indexed."""
        return self._revisions.get(page_id)

    def replace_article(
        self, page_id: int, revision_id: int, chunks: list[Chunk], vectors: Vectors
    ) -> None:
        """Index an article's chunks, replacing whatever was indexed for it before."""
        if vectors.shape != (len(chunks), self.dimension):
            raise ValueError(
                f"Expected vectors of shape ({len(chunks)}, {self.dimension}), got {vectors.shape}"
            )
        if any(chunk.page_id != page_id for chunk in chunks):
            raise ValueError(f"All chunks must belong to page {page_id}")
        self.remove_article(page_id)
        self._chunks.extend(chunks)
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float32, copy=False)])
        self._revisions[page_id] = revision_id

    def remove_article(self, page_id: int) -> None:
        """Drop an article's chunks; does nothing if it is not indexed."""
        if page_id not in self._revisions:
            return
        keep = [i for i, chunk in enumerate(self._chunks) if chunk.page_id != page_id]
        self._chunks = [self._chunks[i] for i in keep]
        self._vectors = self._vectors[keep]
        del self._revisions[page_id]

    def search(self, query: Vectors, k: int = 5) -> list[SearchResult]:
        """The ``k`` chunks most similar to ``query``, best first.

        Assumes normalized vectors, so the dot product is the cosine similarity.
        Equal scores keep index order, which makes results reproducible.
        """
        if k < 1:
            raise ValueError("k must be at least 1")
        if query.shape != (self.dimension,):
            raise ValueError(f"Expected a query of shape ({self.dimension},), got {query.shape}")
        scores = self._vectors @ query
        best = np.argsort(-scores, kind="stable")[:k]
        return [SearchResult(self._chunks[i], float(scores[i])) for i in best]

    def save(self, directory: Path) -> None:
        """Write the store to ``directory``, replacing an earlier save.

        Each file is written to a temporary name and then swapped in, and the
        manifest goes last. A crash part-way leaves either the old manifest,
        whose counts then no longer match and ``load`` reports it, or the new one.
        """
        directory.mkdir(parents=True, exist_ok=True)

        vectors_temp = directory / (VECTORS_FILE + ".tmp")
        with vectors_temp.open("wb") as file:
            np.save(file, self._vectors)
        vectors_temp.replace(directory / VECTORS_FILE)

        lines = "".join(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in self._chunks)
        _write_atomic(directory / CHUNKS_FILE, lines)

        manifest = {
            "format_version": FORMAT_VERSION,
            "embedding_model": self.embedding_model,
            "dimension": self.dimension,
            "chunk_count": len(self._chunks),
            "articles": {str(page_id): rev for page_id, rev in sorted(self._revisions.items())},
        }
        _write_atomic(directory / MANIFEST_FILE, json.dumps(manifest, indent=2) + "\n")

    @classmethod
    def exists(cls, directory: Path) -> bool:
        """Whether a completed save is present in ``directory``."""
        return (directory / MANIFEST_FILE).is_file()

    @classmethod
    def load(cls, directory: Path) -> VectorStore:
        """Read a store saved by ``save``.

        Raises:
            FileNotFoundError: no index has been saved in ``directory``.
            CorruptIndexError: the files are unreadable or do not match each other.
        """
        manifest = _read_json(directory / MANIFEST_FILE)
        try:
            if manifest["format_version"] != FORMAT_VERSION:
                raise CorruptIndexError(f"Unsupported index format {manifest['format_version']}")
            store = cls(manifest["embedding_model"], int(manifest["dimension"]))
            revisions = {int(page): int(rev) for page, rev in manifest["articles"].items()}
            expected = int(manifest["chunk_count"])

            vectors = np.load(directory / VECTORS_FILE, allow_pickle=False)
            text = (directory / CHUNKS_FILE).read_text(encoding="utf-8")
            chunks = [Chunk(**json.loads(line)) for line in text.splitlines() if line]
        except (KeyError, TypeError, ValueError, AttributeError, OSError) as exc:
            raise CorruptIndexError(f"Cannot read index in {directory}: {exc}") from exc

        if vectors.shape != (expected, store.dimension) or len(chunks) != expected:
            raise CorruptIndexError(
                f"Index in {directory} is inconsistent: manifest says {expected} chunks, "
                f"found {len(chunks)} chunks and vectors of shape {vectors.shape}"
            )
        if {chunk.page_id for chunk in chunks} - set(revisions):
            raise CorruptIndexError(f"Index in {directory} has chunks of unlisted articles")

        store._chunks = chunks
        store._vectors = vectors.astype(np.float32, copy=False)
        store._revisions = revisions
        return store


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorruptIndexError(f"Cannot parse {path}: {exc}") from exc


def _write_atomic(path: Path, content: str) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)

"""Bring the vector index in line with the stored articles.

Only articles whose revision changed since they were indexed are embedded
again, since embedding is the slow part. Articles that left the title list are
removed, so the index always matches the list.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from ragnosis.article_store import ArticleStore
from ragnosis.chunking import chunk_article
from ragnosis.embeddings import Embedder
from ragnosis.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IndexReport:
    """What happened to each title, so the caller can summarize and pick an exit code."""

    indexed: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    removed: list[int] = field(default_factory=list)
    chunks_added: int = 0

    @property
    def ok(self) -> bool:
        return not self.missing


def update_index(
    titles: Iterable[str],
    articles: ArticleStore,
    store: VectorStore,
    embedder: Embedder,
) -> IndexReport:
    """Index every title's stored article that is new or changed.

    Titles without a stored article are reported as missing (run ``ingest``
    first) rather than failing the run.

    Raises:
        ValueError: ``store`` was built with a different embedding model, so the
            new vectors could not be compared with the old ones.
    """
    if store.embedding_model != embedder.model_name:
        raise ValueError(
            f"Index was built with {store.embedding_model!r}, "
            f"not {embedder.model_name!r}; rebuild it"
        )

    report = IndexReport()
    listed_page_ids: set[int] = set()

    for title in titles:
        if not articles.contains(title):
            logger.warning("Not indexing %r: article not downloaded, run ingest", title)
            report.missing.append(title)
            continue

        article = articles.load(title)
        listed_page_ids.add(article.page_id)
        if store.revision_of(article.page_id) == article.revision_id:
            logger.debug("Skipping %r: already indexed at this revision", title)
            report.unchanged.append(title)
            continue

        chunks = chunk_article(article)
        vectors = embedder.embed_documents([chunk.embedding_text for chunk in chunks])
        store.replace_article(article.page_id, article.revision_id, chunks, vectors)
        logger.info("Indexed %r (%d chunks)", article.title, len(chunks))
        report.indexed.append(title)
        report.chunks_added += len(chunks)

    for page_id in sorted(store.page_ids - listed_page_ids):
        logger.info("Removing page %d from the index: no longer in the title list", page_id)
        store.remove_article(page_id)
        report.removed.append(page_id)

    return report

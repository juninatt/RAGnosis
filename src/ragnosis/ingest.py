"""Download the articles in a title list and store them.

One failing title should not stop the run: failures are collected and reported
at the end, and a rerun only fetches what is still missing.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from ragnosis.article_store import ArticleStore
from ragnosis.wikipedia import Article, WikipediaError

logger = logging.getLogger(__name__)


class ArticleSource(Protocol):
    """Anything that can fetch an article by title; the real one is WikipediaClient."""

    def fetch_article(self, title: str) -> Article: ...


@dataclass
class IngestReport:
    """What happened to each title, so the caller can summarize and pick an exit code."""

    fetched: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failed


def ingest(
    titles: Iterable[str],
    source: ArticleSource,
    store: ArticleStore,
    *,
    refresh: bool = False,
) -> IngestReport:
    """Fetch and store every title, skipping those already stored unless ``refresh``.

    Titles that redirect to an article already fetched in this run are recorded
    as duplicates and not saved again, so aliases in the title list do not end
    up as repeated text in the index.
    """
    report = IngestReport()
    seen_page_ids: set[int] = set()

    for title in titles:
        if not refresh and store.contains(title):
            logger.debug("Skipping %r: already stored", title)
            report.skipped.append(title)
            continue

        try:
            article = source.fetch_article(title)
        except WikipediaError as exc:
            logger.warning("Could not fetch %r: %s", title, exc)
            report.failed[title] = str(exc)
            continue

        if article.page_id in seen_page_ids:
            logger.info("Skipping %r: same article as an earlier title (%r)", title, article.title)
            report.duplicates.append(title)
            continue

        seen_page_ids.add(article.page_id)
        store.save(title, article)
        logger.info("Fetched %r (%d characters)", article.title, len(article.text))
        report.fetched.append(title)

    return report

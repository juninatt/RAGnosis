"""Download the articles in a title list and store them.

One failing title should not stop the run: failures are collected and reported
at the end, and a rerun only fetches what is still missing. Each article's
other names come from Wikidata, for stored articles that do not have them yet
too, so an upgrade does not need the texts fetched again.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

from ragnosis.article_store import ArticleStore
from ragnosis.wikidata import WikidataError
from ragnosis.wikipedia import Article, WikipediaError

logger = logging.getLogger(__name__)


class ArticleSource(Protocol):
    """Anything that can fetch an article by title; the real one is WikipediaClient."""

    def fetch_article(self, title: str) -> Article: ...


class NameSource(Protocol):
    """Anything that can fetch other names by article title; the real one is WikidataClient."""

    def fetch_names(self, titles: Sequence[str]) -> dict[str, list[str]]: ...


@dataclass
class IngestReport:
    """What happened to each title, so the caller can summarize and pick an exit code."""

    fetched: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    named: list[str] = field(default_factory=list)
    names_failed: str | None = None

    @property
    def ok(self) -> bool:
        return not self.failed and self.names_failed is None


def ingest(
    titles: Iterable[str],
    source: ArticleSource,
    store: ArticleStore,
    *,
    names: NameSource | None = None,
    refresh: bool = False,
) -> IngestReport:
    """Fetch and store every title, skipping those already stored unless ``refresh``.

    Titles that redirect to an article already fetched in this run are recorded
    as duplicates and not saved again, so aliases in the title list do not end
    up as repeated text in the index. With ``names``, every stored article
    without other names gets them; if that fails the articles are kept without
    and the next run tries again.
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

    if names is not None:
        _add_names([*report.fetched, *report.skipped], names, store, report)
    return report


def _add_names(
    titles: list[str], names: NameSource, store: ArticleStore, report: IngestReport
) -> None:
    """Fetch and store other names for the stored ``titles`` that have none yet."""
    unnamed = {title: store.load(title) for title in titles}
    unnamed = {title: article for title, article in unnamed.items() if article.names is None}
    if not unnamed:
        return
    try:
        found = names.fetch_names([article.title for article in unnamed.values()])
    except WikidataError as exc:
        logger.warning("Could not fetch other names from Wikidata: %s; the next run retries", exc)
        report.names_failed = str(exc)
        return
    for title, article in unnamed.items():
        store.save(title, replace(article, names=tuple(found.get(article.title, []))))
        report.named.append(title)

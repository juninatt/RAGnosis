"""Command line entry point: ``ragnosis <command>``.

This is the only place that loads env files and configures logging; everything
below it receives settings as arguments. Exit codes: 0 success, 1 some work
failed, 2 bad usage or configuration.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from ragnosis.article_store import ArticleStore
from ragnosis.env import load_env
from ragnosis.ingest import ingest
from ragnosis.logging_config import setup_logging
from ragnosis.settings import Settings
from ragnosis.topics import load_titles
from ragnosis.wikipedia import WikipediaClient

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILURES = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragnosis",
        description="Answer questions from Wikipedia using local retrieval and a local model.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest_parser = commands.add_parser("ingest", help="download the articles in the topics file")
    ingest_parser.add_argument(
        "--topics",
        type=Path,
        help="title list to read (default: RAGNOSIS_TOPICS_FILE)",
    )
    ingest_parser.add_argument(
        "--refresh",
        action="store_true",
        help="download again even if an article is already stored",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    loaded = load_env()
    try:
        setup_logging()
        settings = Settings.from_env()
    except ValueError as exc:
        parser.exit(EXIT_USAGE, f"ragnosis: configuration error: {exc}\n")
    logger.debug("Loaded env files: %s", [str(path) for path in loaded])

    return run_ingest(settings, topics=args.topics, refresh=args.refresh)


def run_ingest(settings: Settings, *, topics: Path | None, refresh: bool) -> int:
    topics_file = topics or settings.topics_file
    try:
        titles = load_titles(topics_file)
    except OSError as exc:
        logger.error("Cannot read topics file %s: %s", topics_file, exc)
        return EXIT_USAGE

    logger.info("Ingesting %d titles from %s", len(titles), topics_file)
    with WikipediaClient(settings.wiki_language) as client:
        report = ingest(titles, client, ArticleStore(settings.articles_dir), refresh=refresh)

    logger.info(
        "Done: %d fetched, %d already stored, %d duplicates, %d failed",
        len(report.fetched),
        len(report.skipped),
        len(report.duplicates),
        len(report.failed),
    )
    for title, reason in report.failed.items():
        logger.error("Failed: %r: %s", title, reason)
    return EXIT_OK if report.ok else EXIT_FAILURES

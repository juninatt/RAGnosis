"""Command line entry point: ``python -m ragnosis <command>``.

This is the only place that loads env files and configures logging; everything
below it receives settings as arguments. Logs go to stderr and results (search
hits) to stdout, so results can be piped without log noise. Exit codes: 0
success, 1 some work failed, 2 bad usage or configuration.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from ragnosis.article_store import ArticleStore
from ragnosis.embeddings import SentenceTransformerEmbedder
from ragnosis.env import load_env
from ragnosis.indexing import update_index
from ragnosis.ingest import ingest
from ragnosis.logging_config import setup_logging
from ragnosis.settings import Settings
from ragnosis.topics import load_titles
from ragnosis.vector_store import CorruptIndexError, SearchResult, VectorStore
from ragnosis.wikipedia import WikipediaClient

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILURES = 1
EXIT_USAGE = 2

PREVIEW_CHARS = 240


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragnosis",
        description="Answer questions from Wikipedia using local retrieval and a local model.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    topics_help = "title list to read (default: RAGNOSIS_TOPICS_FILE)"

    ingest_parser = commands.add_parser("ingest", help="download the articles in the topics file")
    ingest_parser.add_argument("--topics", type=Path, help=topics_help)
    ingest_parser.add_argument(
        "--refresh",
        action="store_true",
        help="download again even if an article is already stored",
    )

    index_parser = commands.add_parser("index", help="embed the downloaded articles for search")
    index_parser.add_argument("--topics", type=Path, help=topics_help)
    index_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="start from an empty index, e.g. after changing the embedding model",
    )

    search_parser = commands.add_parser("search", help="show the passages closest to a question")
    search_parser.add_argument("query", help="question or search text, in any language")
    search_parser.add_argument(
        "-k", type=positive_int, default=5, help="number of passages (default: 5)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Article text can hold characters the console encoding lacks (Greek, for
    # one); show a replacement character rather than crash on them.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    loaded = load_env()
    try:
        setup_logging()
        settings = Settings.from_env()
    except ValueError as exc:
        parser.exit(EXIT_USAGE, f"ragnosis: configuration error: {exc}\n")
    logger.debug("Loaded env files: %s", [str(path) for path in loaded])

    if args.command == "ingest":
        return run_ingest(settings, topics=args.topics, refresh=args.refresh)
    if args.command == "index":
        return run_index(settings, topics=args.topics, rebuild=args.rebuild)
    return run_search(settings, query=args.query, k=args.k)


def run_ingest(settings: Settings, *, topics: Path | None, refresh: bool) -> int:
    titles = _read_titles(topics or settings.topics_file)
    if titles is None:
        return EXIT_USAGE

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


def run_index(settings: Settings, *, topics: Path | None, rebuild: bool) -> int:
    titles = _read_titles(topics or settings.topics_file)
    if titles is None:
        return EXIT_USAGE

    store: VectorStore | None = None
    if not rebuild and VectorStore.exists(settings.index_dir):
        store = _load_store(settings)
        if store is None:
            return EXIT_USAGE

    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    if store is None:
        store = VectorStore(embedder.model_name, embedder.dimension)

    report = update_index(titles, ArticleStore(settings.articles_dir), store, embedder)
    store.save(settings.index_dir)

    logger.info(
        "Done: %d indexed (%d chunks), %d unchanged, %d removed, %d missing; %d chunks in total",
        len(report.indexed),
        report.chunks_added,
        len(report.unchanged),
        len(report.removed),
        len(report.missing),
        len(store),
    )
    for title in report.missing:
        logger.error("Missing: %r is not downloaded; run ingest", title)
    return EXIT_OK if report.ok else EXIT_FAILURES


def run_search(settings: Settings, *, query: str, k: int) -> int:
    if not VectorStore.exists(settings.index_dir):
        logger.error("No index in %s; run index first", settings.index_dir)
        return EXIT_USAGE
    store = _load_store(settings)
    if store is None:
        return EXIT_USAGE

    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    results = store.search(embedder.embed_query(query), k)
    sys.stdout.write(format_results(results))
    return EXIT_OK


def format_results(results: Sequence[SearchResult]) -> str:
    """Human-readable search hits: rank, score, where the passage is from, and a preview."""
    if not results:
        return "No results.\n"
    blocks = []
    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        preview = chunk.text
        if len(preview) > PREVIEW_CHARS:
            preview = preview[:PREVIEW_CHARS].rstrip() + "…"
        blocks.append(
            f"{rank}. [{result.score:.3f}] {chunk.article_title} > {chunk.section}\n"
            f"   {chunk.url}\n"
            f"   {preview}\n"
        )
    return "\n".join(blocks)


def _read_titles(path: Path) -> list[str] | None:
    try:
        titles = load_titles(path)
    except OSError as exc:
        logger.error("Cannot read topics file %s: %s", path, exc)
        return None
    logger.info("Read %d titles from %s", len(titles), path)
    return titles


def _load_store(settings: Settings) -> VectorStore | None:
    """Load the index if it is readable and was built with the configured model."""
    try:
        store = VectorStore.load(settings.index_dir)
    except CorruptIndexError as exc:
        logger.error("%s; run index --rebuild", exc)
        return None
    if store.embedding_model != settings.embedding_model:
        logger.error(
            "Index was built with %r but RAGNOSIS_EMBEDDING_MODEL is %r; run index --rebuild",
            store.embedding_model,
            settings.embedding_model,
        )
        return None
    return store

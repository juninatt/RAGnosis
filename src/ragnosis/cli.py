"""Command line entry point: ``python -m ragnosis <command>``.

This is the only place that loads env files and configures logging; everything
below it receives settings as arguments. Logs go to stderr and results (search
hits, answers) to stdout, so results can be piped without log noise. Exit
codes: 0 success, 1 some work failed, 2 bad usage or configuration (including
a missing index, Ollama not running or the model not downloaded).
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import uvicorn

from ragnosis.answer import DEFAULT_K, Answer, AnswerLength, ask
from ragnosis.article_store import ArticleStore
from ragnosis.embeddings import SentenceTransformerEmbedder
from ragnosis.env import load_env
from ragnosis.evaluation import DEFAULT_CASES_FILE, CaseError, Verdict, check, load_cases
from ragnosis.indexing import update_index
from ragnosis.ingest import ingest
from ragnosis.llm import LLMError, LLMUnavailableError, ModelNotFoundError, OllamaClient
from ragnosis.logging_config import setup_logging
from ragnosis.settings import Settings
from ragnosis.topics import load_titles
from ragnosis.vector_store import CorruptIndexError, SearchResult, VectorStore
from ragnosis.web import create_app
from ragnosis.wikipedia import WikipediaClient

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILURES = 1
EXIT_USAGE = 2

PREVIEW_CHARS = 240
SERVE_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DETAILED_HELP = "answer in a paragraph or two instead of at most 5 sentences"

# Times each evaluated answer; tests replace it with a fake clock.
_clock: Callable[[], float] = time.monotonic


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def port_number(value: str) -> int:
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("must be from 1 to 65535")
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

    ask_parser = commands.add_parser("ask", help="answer a question from the indexed articles")
    ask_parser.add_argument("question", help="the question, in any language")
    ask_parser.add_argument(
        "-k",
        type=positive_int,
        default=DEFAULT_K,
        help=f"number of passages the model reads (default: {DEFAULT_K})",
    )
    ask_parser.add_argument("--detailed", action="store_true", help=DETAILED_HELP)

    serve_parser = commands.add_parser("serve", help="start the web UI on this computer")
    serve_parser.add_argument(
        "--port", type=port_number, default=DEFAULT_PORT, help=f"(default: {DEFAULT_PORT})"
    )

    evaluate_parser = commands.add_parser(
        "evaluate", help="answer known questions and score the answers"
    )
    evaluate_parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_FILE,
        help=f"cases file (default: {DEFAULT_CASES_FILE.as_posix()})",
    )
    evaluate_parser.add_argument(
        "--case",
        dest="only",
        action="append",
        metavar="ID",
        help="run only this case; repeat for several",
    )
    evaluate_parser.add_argument("--detailed", action="store_true", help=DETAILED_HELP)
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
    if args.command == "search":
        return run_search(settings, query=args.query, k=args.k)
    if args.command == "ask":
        return run_ask(settings, question=args.question, k=args.k, length=_length(args))
    if args.command == "serve":
        return run_serve(settings, port=args.port)
    return run_evaluate(settings, cases_file=args.cases, only=args.only, length=_length(args))


def _length(args: argparse.Namespace) -> AnswerLength:
    return "detailed" if args.detailed else "short"


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


def run_ask(settings: Settings, *, question: str, k: int, length: AnswerLength = "short") -> int:
    if not question.strip():
        logger.error("The question must not be blank")
        return EXIT_USAGE
    if not VectorStore.exists(settings.index_dir):
        logger.error("No index in %s; run index first", settings.index_dir)
        return EXIT_USAGE
    store = _load_store(settings)
    if store is None:
        return EXIT_USAGE

    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    try:
        with OllamaClient(
            settings.llm_model, settings.ollama_url, keep_alive=settings.llm_keep_alive
        ) as llm:
            answer = ask(
                question,
                store=store,
                embedder=embedder,
                llm=llm,
                k=k,
                rewrite=settings.query_rewrite,
                length=length,
                on_text=_show,
            )
    except (LLMUnavailableError, ModelNotFoundError) as exc:
        logger.error("%s", exc)
        return EXIT_USAGE
    except LLMError as exc:
        _show("\n")
        logger.error("%s", exc)
        return EXIT_FAILURES

    _show("\n\n" + format_sources(answer))
    return EXIT_OK


def run_serve(settings: Settings, *, port: int) -> int:
    """Start the web UI; runs until stopped with Ctrl+C."""
    if not VectorStore.exists(settings.index_dir):
        logger.error("No index in %s; run index first", settings.index_dir)
        return EXIT_USAGE
    store = _load_store(settings)
    if store is None:
        return EXIT_USAGE

    # Load the models once here, not on every question.
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    with OllamaClient(
        settings.llm_model, settings.ollama_url, keep_alive=settings.llm_keep_alive
    ) as llm:

        def answer_question(
            question: str, k: int, length: AnswerLength, on_text: Callable[[str], None]
        ) -> Answer:
            return ask(
                question,
                store=store,
                embedder=embedder,
                llm=llm,
                k=k,
                rewrite=settings.query_rewrite,
                length=length,
                on_text=on_text,
            )

        _warm_up(llm)
        logger.info("RAGnosis is running at http://localhost:%d (Ctrl+C stops it)", port)
        # Localhost only: the UI is for this computer, not the network (0007).
        uvicorn.run(create_app(answer_question), host=SERVE_HOST, port=port, log_level="warning")
    return EXIT_OK


def run_evaluate(
    settings: Settings,
    *,
    cases_file: Path,
    only: list[str] | None,
    length: AnswerLength = "short",
) -> int:
    """Answer each known question with the real models and print a scorecard.

    Compare scorecards from before and after a change (docs/decisions/0008).
    """
    try:
        cases = load_cases(cases_file)
    except (CaseError, OSError) as exc:
        logger.error("Cannot read cases: %s", exc)
        return EXIT_USAGE
    if only:
        unknown = sorted(set(only) - {case.id for case in cases})
        if unknown:
            logger.error("Unknown case ids: %s", unknown)
            return EXIT_USAGE
        cases = [case for case in cases if case.id in only]

    if not VectorStore.exists(settings.index_dir):
        logger.error("No index in %s; run index first", settings.index_dir)
        return EXIT_USAGE
    store = _load_store(settings)
    if store is None:
        return EXIT_USAGE

    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    results: list[tuple[Verdict, float]] = []
    with OllamaClient(
        settings.llm_model, settings.ollama_url, keep_alive=settings.llm_keep_alive
    ) as llm:
        _warm_up(llm)
        for number, case in enumerate(cases, start=1):
            logger.info("Case %d/%d: %s", number, len(cases), case.id)
            start = _clock()
            try:
                answer = ask(
                    case.question,
                    store=store,
                    embedder=embedder,
                    llm=llm,
                    rewrite=settings.query_rewrite,
                    length=length,
                )
            except (LLMUnavailableError, ModelNotFoundError) as exc:
                logger.error("%s", exc)
                return EXIT_USAGE
            except LLMError as exc:
                answer = Answer(case.question, "", [])
                verdict = Verdict(case, answer, [f"the model failed: {exc}"])
            else:
                verdict = check(case, answer)
            seconds = _clock() - start
            results.append((verdict, seconds))
            _show(format_verdict(verdict, seconds))

    _show(format_summary(results))
    return EXIT_OK if all(verdict.passed for verdict, _ in results) else EXIT_FAILURES


def format_verdict(verdict: Verdict, seconds: float) -> str:
    """One case: PASS or FAIL with its time, and for a failure why and how it began."""
    line = f"{'PASS' if verdict.passed else 'FAIL'}  {verdict.case.id:<28} {seconds:4.0f} s\n"
    if verdict.passed:
        return line
    details = "".join(f"      - {problem}\n" for problem in verdict.problems)
    if verdict.answer.search_query is not None:
        details += f"      searched for: {verdict.answer.search_query}\n"
    preview = " ".join(verdict.answer.text.split())
    if len(preview) > PREVIEW_CHARS:
        preview = preview[:PREVIEW_CHARS].rstrip() + "…"
    return line + details + f"      answer: {preview or '(none)'}\n"


def format_summary(results: Sequence[tuple[Verdict, float]]) -> str:
    """How many passed, and typical and total time."""
    if not results:
        return "\nNo cases were run.\n"
    passed = sum(verdict.passed for verdict, _ in results)
    seconds = [s for _, s in results]
    return (
        f"\n{passed} of {len(results)} passed · median {statistics.median(seconds):.0f} s"
        f" · total {sum(seconds) / 60:.1f} min\n"
    )


def _warm_up(llm: OllamaClient) -> None:
    """Load the language model now, so the first question does not wait for it.

    A failure is only a warning: Ollama may be started after the server, and
    each question reports clearly if the model is still unavailable then.
    """
    logger.info("Loading the language model %s", llm.model)
    try:
        llm.load()
    except LLMError as exc:
        logger.warning("Could not load the language model yet: %s", exc)


def format_sources(answer: Answer) -> str:
    """The cited passages with their numbers, so each [n] in the answer can be checked."""
    if not answer.cited:
        return "No sources cited.\n"
    lines = ["Sources:"]
    for number, result in answer.cited:
        chunk = result.chunk
        lines.append(f"[{number}] {chunk.article_title} > {chunk.section}")
        lines.append(f"    {chunk.url}")
    return "\n".join(lines) + "\n"


def _show(text: str) -> None:
    """Write to stdout at once; answers arrive slowly and should appear as written."""
    sys.stdout.write(text)
    sys.stdout.flush()


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

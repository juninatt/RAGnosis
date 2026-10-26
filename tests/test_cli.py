"""Tests for the command line, with Wikipedia and the models replaced by fakes."""

from __future__ import annotations

import io
import runpy
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import TracebackType

import pytest
from factories import FakeChatModel, FakeEmbedder, make_article, make_chunk

from ragnosis import cli
from ragnosis.answer import Answer
from ragnosis.article_store import ArticleStore
from ragnosis.llm import LLMError, LLMUnavailableError, Message, ModelNotFoundError
from ragnosis.vector_store import MANIFEST_FILE, SearchResult, VectorStore
from ragnosis.wikipedia import Article, ArticleNotFoundError


class FakeWikipediaClient:
    """Stands in for WikipediaClient; knows a fixed set of articles."""

    articles: dict[str, Article] = {}
    languages: list[str] = []

    def __init__(self, language: str = "en") -> None:
        self.languages.append(language)

    def __enter__(self) -> FakeWikipediaClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

    def fetch_article(self, title: str) -> Article:
        if title not in self.articles:
            raise ArticleNotFoundError(f"No Wikipedia article titled {title!r}")
        return self.articles[title]


@pytest.fixture(autouse=True)
def project_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run each test in an empty project folder with a fake Wikipedia and embedder."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "WikipediaClient", FakeWikipediaClient)
    monkeypatch.setattr(cli, "SentenceTransformerEmbedder", lambda name: FakeEmbedder(name))
    monkeypatch.setattr(
        FakeWikipediaClient,
        "articles",
        {
            "Blackbeard": make_article(title="Blackbeard", page_id=1),
            "Anne Bonny": make_article(
                title="Anne Bonny",
                page_id=2,
                url="https://en.wikipedia.org/wiki/Anne_Bonny",
                text="Anne Bonny was an Irish pirate who sailed with Calico Jack.",
            ),
        },
    )
    monkeypatch.setattr(FakeWikipediaClient, "languages", [])
    return tmp_path


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_ingest_stores_articles_and_exits_ok(project_dir: Path) -> None:
    topics = write(project_dir / "topics.txt", "Blackbeard\nAnne Bonny\n")

    assert cli.main(["ingest", "--topics", str(topics)]) == cli.EXIT_OK

    store = ArticleStore(project_dir / "data" / "articles")
    assert store.contains("Blackbeard")
    assert store.contains("Anne Bonny")


def test_ingest_uses_settings_from_env_file(project_dir: Path) -> None:
    write(project_dir / "lists" / "mine.txt", "Blackbeard\n")
    write(
        project_dir / ".env",
        "RAGNOSIS_TOPICS_FILE=lists/mine.txt\nRAGNOSIS_DATA_DIR=store\nRAGNOSIS_WIKI_LANGUAGE=sv\n",
    )

    assert cli.main(["ingest"]) == cli.EXIT_OK

    assert ArticleStore(project_dir / "store" / "articles").contains("Blackbeard")
    assert FakeWikipediaClient.languages == ["sv"]


def test_topics_option_overrides_settings(project_dir: Path) -> None:
    write(project_dir / ".env", "RAGNOSIS_TOPICS_FILE=does-not-exist.txt\n")
    topics = write(project_dir / "topics.txt", "Blackbeard\n")

    assert cli.main(["ingest", "--topics", str(topics)]) == cli.EXIT_OK


def test_ingest_exits_with_failure_code_when_a_title_fails(
    project_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    topics = write(project_dir / "topics.txt", "Blackbeard\nNo such pirate\n")

    assert cli.main(["ingest", "--topics", str(topics)]) == cli.EXIT_FAILURES
    assert "Failed: 'No such pirate'" in caplog.text
    assert "1 fetched, 0 already stored, 0 duplicates, 1 failed" in caplog.text


def test_second_run_skips_stored_articles_and_refresh_refetches(
    project_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    topics = write(project_dir / "topics.txt", "Blackbeard\n")
    cli.main(["ingest", "--topics", str(topics)])

    caplog.clear()
    cli.main(["ingest", "--topics", str(topics)])
    assert "0 fetched, 1 already stored" in caplog.text

    caplog.clear()
    cli.main(["ingest", "--topics", str(topics), "--refresh"])
    assert "1 fetched, 0 already stored" in caplog.text


def test_missing_topics_file_exits_with_usage_code(
    project_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert cli.main(["ingest", "--topics", "missing.txt"]) == cli.EXIT_USAGE
    assert "Cannot read topics file" in caplog.text


def test_invalid_configuration_exits_with_usage_code(
    project_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write(project_dir / ".env", "RAGNOSIS_WIKI_LANGUAGE=not a language\n")

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["ingest"])

    assert exit_info.value.code == cli.EXIT_USAGE
    assert "configuration error" in capsys.readouterr().err


def test_invalid_log_level_exits_with_usage_code(project_dir: Path) -> None:
    write(project_dir / ".env", "RAGNOSIS_LOG_LEVEL=chatty\n")
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["ingest"])
    assert exit_info.value.code == cli.EXIT_USAGE


def test_command_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main([])
    assert exit_info.value.code == cli.EXIT_USAGE
    assert "required" in capsys.readouterr().err


def test_runs_as_python_module(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["ragnosis", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("ragnosis", run_name="__main__")
    assert exit_info.value.code == 0
    assert "ingest" in capsys.readouterr().out


def test_logs_go_to_stderr_not_stdout(
    project_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    topics = write(project_dir / "topics.txt", "Blackbeard\n")
    cli.main(["ingest", "--topics", str(topics)])

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Fetched 'Blackbeard'" in captured.err


# --- index ---


def ingest_both(project_dir: Path) -> Path:
    topics = write(project_dir / "topics.txt", "Blackbeard\nAnne Bonny\n")
    assert cli.main(["ingest", "--topics", str(topics)]) == cli.EXIT_OK
    return topics


def test_index_builds_index_from_ingested_articles(project_dir: Path) -> None:
    topics = ingest_both(project_dir)

    assert cli.main(["index", "--topics", str(topics)]) == cli.EXIT_OK

    store = VectorStore.load(project_dir / "data" / "index")
    assert store.page_ids == {1, 2}
    assert store.embedding_model == "intfloat/multilingual-e5-small"


def test_second_index_run_finds_nothing_to_do(
    project_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    topics = ingest_both(project_dir)
    cli.main(["index", "--topics", str(topics)])
    caplog.clear()

    assert cli.main(["index", "--topics", str(topics)]) == cli.EXIT_OK
    assert "0 indexed (0 chunks), 2 unchanged" in caplog.text


def test_index_reports_titles_not_yet_ingested(
    project_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ingest_both(project_dir)
    topics = write(project_dir / "more.txt", "Blackbeard\nHannibal\n")

    assert cli.main(["index", "--topics", str(topics)]) == cli.EXIT_FAILURES
    assert "Missing: 'Hannibal' is not downloaded" in caplog.text


def test_index_refuses_index_built_with_another_model(
    project_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    topics = ingest_both(project_dir)
    cli.main(["index", "--topics", str(topics)])
    write(project_dir / ".env", "RAGNOSIS_EMBEDDING_MODEL=other/model\n")

    assert cli.main(["index", "--topics", str(topics)]) == cli.EXIT_USAGE
    assert "run index --rebuild" in caplog.text


def test_rebuild_replaces_index_built_with_another_model(project_dir: Path) -> None:
    topics = ingest_both(project_dir)
    cli.main(["index", "--topics", str(topics)])
    write(project_dir / ".env", "RAGNOSIS_EMBEDDING_MODEL=other/model\n")

    assert cli.main(["index", "--topics", str(topics), "--rebuild"]) == cli.EXIT_OK
    assert VectorStore.load(project_dir / "data" / "index").embedding_model == "other/model"


def test_index_refuses_corrupt_index(project_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    topics = ingest_both(project_dir)
    cli.main(["index", "--topics", str(topics)])
    write(project_dir / "data" / "index" / MANIFEST_FILE, "{broken")

    assert cli.main(["index", "--topics", str(topics)]) == cli.EXIT_USAGE
    assert "run index --rebuild" in caplog.text


def test_index_with_missing_topics_file_exits_with_usage_code() -> None:
    assert cli.main(["index", "--topics", "missing.txt"]) == cli.EXIT_USAGE


# --- search ---


def test_search_prints_best_matches_to_stdout(
    project_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    topics = ingest_both(project_dir)
    cli.main(["index", "--topics", str(topics)])
    capsys.readouterr()

    assert cli.main(["search", "Irish pirate Calico Jack", "-k", "1"]) == cli.EXIT_OK

    out = capsys.readouterr().out
    assert out.startswith("1. [")
    assert "Anne Bonny > Introduction" in out
    assert "https://en.wikipedia.org/wiki/Anne_Bonny" in out
    assert "2. [" not in out


def test_search_without_index_exits_with_usage_code(caplog: pytest.LogCaptureFixture) -> None:
    assert cli.main(["search", "pirates"]) == cli.EXIT_USAGE
    assert "run index first" in caplog.text


def test_search_refuses_index_built_with_another_model(project_dir: Path) -> None:
    topics = ingest_both(project_dir)
    cli.main(["index", "--topics", str(topics)])
    write(project_dir / ".env", "RAGNOSIS_EMBEDDING_MODEL=other/model\n")

    assert cli.main(["search", "pirates"]) == cli.EXIT_USAGE


@pytest.mark.parametrize("k", ["0", "-2", "many"])
def test_search_rejects_invalid_k(k: str) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["search", "pirates", "-k", k])
    assert exit_info.value.code == cli.EXIT_USAGE


def test_works_when_stdout_cannot_be_reconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    # Embedding hosts may replace stdout with a plain stream without reconfigure().
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    assert cli.main(["search", "pirates"]) == cli.EXIT_USAGE


def test_format_results_without_hits() -> None:
    assert cli.format_results([]) == "No results.\n"


def test_format_results_shows_rank_score_source_and_preview() -> None:
    result = SearchResult(make_chunk(section="Death", text="He was killed in 1718."), 0.8123)
    assert cli.format_results([result]) == (
        "1. [0.812] Blackbeard > Death\n"
        "   https://en.wikipedia.org/wiki/Blackbeard\n"
        "   He was killed in 1718.\n"
    )


def test_format_results_shortens_long_passages() -> None:
    text = "word " * 100
    output = cli.format_results([SearchResult(make_chunk(text=text), 0.5)])
    preview = output.splitlines()[2].strip()
    assert preview.endswith("…")
    assert len(preview) <= cli.PREVIEW_CHARS + 1


# --- ask ---


class FakeOllama(FakeChatModel):
    """Stands in for OllamaClient: replies with ``reply``, or raises ``error``."""

    reply: tuple[str, ...] = ("Anne Bonny ", "was an Irish pirate [1].")
    error: Exception | None = None
    created: list[tuple[str, str]] = []

    def __init__(self, model: str, base_url: str) -> None:
        super().__init__(*self.reply)
        self.created.append((model, base_url))

    def __enter__(self) -> FakeOllama:
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass

    def stream_chat(
        self, messages: Sequence[Message], *, temperature: float = 0.2
    ) -> Iterator[str]:
        yield from super().stream_chat(messages, temperature=temperature)
        if self.error:
            raise self.error


@pytest.fixture
def indexed(project_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project with both articles ingested and indexed, and a fake Ollama."""
    topics = ingest_both(project_dir)
    assert cli.main(["index", "--topics", str(topics)]) == cli.EXIT_OK
    monkeypatch.setattr(cli, "OllamaClient", FakeOllama)
    monkeypatch.setattr(FakeOllama, "created", [])
    return project_dir


def test_ask_streams_answer_then_cited_sources(
    indexed: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()

    assert cli.main(["ask", "Irish pirate Calico Jack", "-k", "1"]) == cli.EXIT_OK

    assert capsys.readouterr().out == (
        "Anne Bonny was an Irish pirate [1].\n\n"
        "Sources:\n"
        "[1] Anne Bonny > Introduction\n"
        "    https://en.wikipedia.org/wiki/Anne_Bonny\n"
    )


def test_ask_uses_model_and_address_from_settings(indexed: Path) -> None:
    write(
        indexed / ".env",
        "RAGNOSIS_LLM_MODEL=gemma4:e4b\nRAGNOSIS_OLLAMA_URL=http://gpu-box:11434\n",
    )
    cli.main(["ask", "pirates"])
    assert FakeOllama.created == [("gemma4:e4b", "http://gpu-box:11434")]


def test_ask_says_when_nothing_is_cited(
    indexed: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(FakeOllama, "reply", ("I cannot answer from the sources.",))
    capsys.readouterr()

    assert cli.main(["ask", "Capital of Japan?"]) == cli.EXIT_OK
    assert capsys.readouterr().out.endswith("\n\nNo sources cited.\n")


@pytest.mark.parametrize(
    "error",
    [
        LLMUnavailableError("Ollama is not reachable; start Ollama"),
        ModelNotFoundError("run: ollama pull qwen3.5:4b"),
    ],
    ids=["ollama-not-running", "model-missing"],
)
def test_ask_setup_problems_exit_with_usage_code(
    indexed: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
) -> None:
    monkeypatch.setattr(FakeOllama, "reply", ())
    monkeypatch.setattr(FakeOllama, "error", error)

    assert cli.main(["ask", "pirates"]) == cli.EXIT_USAGE
    assert str(error) in caplog.text


def test_ask_failure_mid_answer_exits_with_failure_code(
    indexed: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(FakeOllama, "reply", ("Anne Bonny was",))
    monkeypatch.setattr(FakeOllama, "error", LLMError("Lost connection to Ollama"))
    capsys.readouterr()

    assert cli.main(["ask", "pirates"]) == cli.EXIT_FAILURES
    assert capsys.readouterr().out == "Anne Bonny was\n"
    assert "Lost connection to Ollama" in caplog.text


def test_ask_without_index_exits_with_usage_code(caplog: pytest.LogCaptureFixture) -> None:
    assert cli.main(["ask", "pirates"]) == cli.EXIT_USAGE
    assert "run index first" in caplog.text


def test_ask_refuses_index_built_with_another_model(indexed: Path) -> None:
    write(indexed / ".env", "RAGNOSIS_EMBEDDING_MODEL=other/model\n")
    assert cli.main(["ask", "pirates"]) == cli.EXIT_USAGE
    assert FakeOllama.created == []


@pytest.mark.parametrize("question", ["", "   "])
def test_ask_rejects_blank_question(question: str, caplog: pytest.LogCaptureFixture) -> None:
    assert cli.main(["ask", question]) == cli.EXIT_USAGE
    assert "must not be blank" in caplog.text


def test_format_sources_lists_cited_passages_in_citation_order() -> None:
    passages = [
        SearchResult(make_chunk(article_title="Blackbeard", section="Death"), 0.9),
        SearchResult(
            make_chunk(article_title="Anne Bonny", url="https://en.wikipedia.org/wiki/Anne_Bonny"),
            0.8,
        ),
    ]
    answer = Answer("q", "Bonny [2] and Teach [1].", passages)
    assert cli.format_sources(answer) == (
        "Sources:\n"
        "[2] Anne Bonny > Introduction\n"
        "    https://en.wikipedia.org/wiki/Anne_Bonny\n"
        "[1] Blackbeard > Death\n"
        "    https://en.wikipedia.org/wiki/Blackbeard\n"
    )

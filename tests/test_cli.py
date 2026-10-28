"""Tests for the command line, with Wikipedia and the models replaced by fakes."""

from __future__ import annotations

import io
import json
import runpy
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

import pytest
from factories import FakeChatModel, FakeEmbedder, make_article, make_chunk
from starlette.testclient import TestClient

from ragnosis import cli
from ragnosis.answer import LENGTH_RULES, Answer, AnswerLength
from ragnosis.article_store import ArticleStore
from ragnosis.evaluation import Case, Verdict
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
    keep_alives: list[str] = []
    load_error: Exception | None = None
    # Shared with FakeUvicorn, so tests can see what happened in which order.
    events: list[str] = []
    # The system message of every chat, to see which instructions were sent.
    systems: list[str] = []

    def __init__(self, model: str, base_url: str, *, keep_alive: str = "30m") -> None:
        super().__init__(*self.reply)
        self.model = model
        self.created.append((model, base_url))
        self.keep_alives.append(keep_alive)

    def load(self) -> None:
        self.events.append("load")
        if self.load_error:
            raise self.load_error

    def __enter__(self) -> FakeOllama:
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass

    def stream_chat(
        self, messages: Sequence[Message], *, temperature: float = 0.2
    ) -> Iterator[str]:
        self.events.append("chat")
        self.systems.append(messages[0].content)
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
    monkeypatch.setattr(FakeOllama, "keep_alives", [])
    monkeypatch.setattr(FakeOllama, "events", [])
    monkeypatch.setattr(FakeOllama, "systems", [])
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


@pytest.mark.parametrize(
    ("setting", "chats"),
    [("", 2), ("RAGNOSIS_QUERY_REWRITE=never\n", 1), ("RAGNOSIS_QUERY_REWRITE=always\n", 2)],
    ids=["default", "never", "always"],
)
def test_query_rewrite_setting_decides_whether_a_swedish_question_is_rewritten(
    indexed: Path, setting: str, chats: int
) -> None:
    write(indexed / ".env", setting)
    cli.main(["ask", "Vem var Anne Bonny?"])
    assert FakeOllama.events.count("chat") == chats


@pytest.mark.parametrize(
    ("options", "length"),
    [([], "short"), (["--detailed"], "detailed")],
    ids=["default", "detailed"],
)
def test_ask_answers_short_unless_detailed_is_asked_for(
    indexed: Path, options: list[str], length: AnswerLength
) -> None:
    cli.main(["ask", "Who was Anne Bonny?", *options])
    assert LENGTH_RULES[length] in FakeOllama.systems[-1]


@pytest.mark.parametrize(
    ("options", "length"),
    [([], "short"), (["--detailed"], "detailed")],
    ids=["default", "detailed"],
)
def test_evaluate_answers_at_the_chosen_length(
    cases_file: Path, options: list[str], length: AnswerLength
) -> None:
    cli.main(["evaluate", "--cases", str(cases_file), "--case", "bonny-en", *options])
    assert LENGTH_RULES[length] in FakeOllama.systems[-1]


@pytest.mark.parametrize(("setting", "chats"), [("", 2), ("RAGNOSIS_QUERY_REWRITE=swedish\n", 1)])
def test_english_question_is_rewritten_unless_only_swedish_ones_are(
    indexed: Path, setting: str, chats: int
) -> None:
    write(indexed / ".env", setting)
    cli.main(["ask", "Who was Anne Bonny?"])
    assert FakeOllama.events.count("chat") == chats


def test_served_app_rewrites_by_the_setting(indexed: Path, fake_uvicorn: FakeUvicorn) -> None:
    cli.main(["serve"])
    client = TestClient(fake_uvicorn.calls[0]["app"], base_url="http://localhost")
    client.post("/api/ask", json={"question": "Vem var Anne Bonny?"})
    assert FakeOllama.events.count("chat") == 2


@pytest.mark.parametrize("command", [["ask", "pirates"], ["serve"]])
def test_keep_alive_setting_reaches_ollama(
    indexed: Path, monkeypatch: pytest.MonkeyPatch, command: list[str]
) -> None:
    monkeypatch.setattr(cli, "uvicorn", FakeUvicorn())
    write(indexed / ".env", "RAGNOSIS_LLM_KEEP_ALIVE=1h\n")
    cli.main(command)
    assert FakeOllama.keep_alives == ["1h"]


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
    # Without rewriting, the first model call is the answer itself.
    write(indexed / ".env", "RAGNOSIS_QUERY_REWRITE=never\n")
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


# --- serve ---


class FakeUvicorn:
    """Records the app instead of starting a server."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, app: Any, **options: Any) -> None:
        self.calls.append({"app": app, **options})
        FakeOllama.events.append("run")


@pytest.fixture
def fake_uvicorn(monkeypatch: pytest.MonkeyPatch) -> FakeUvicorn:
    fake = FakeUvicorn()
    monkeypatch.setattr(cli, "uvicorn", fake)
    return fake


def test_serve_starts_the_web_app_on_localhost(indexed: Path, fake_uvicorn: FakeUvicorn) -> None:
    assert cli.main(["serve", "--port", "8123"]) == cli.EXIT_OK

    (call,) = fake_uvicorn.calls
    assert call["host"] == "127.0.0.1"
    assert call["port"] == 8123


def test_serve_loads_the_model_before_serving(indexed: Path, fake_uvicorn: FakeUvicorn) -> None:
    cli.main(["serve"])
    assert FakeOllama.events == ["load", "run"]


def test_serve_still_starts_when_the_model_cannot_be_loaded(
    indexed: Path,
    fake_uvicorn: FakeUvicorn,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(FakeOllama, "load_error", LLMUnavailableError("start Ollama"))

    assert cli.main(["serve"]) == cli.EXIT_OK

    assert FakeOllama.events == ["load", "run"]
    assert "Could not load the language model yet: start Ollama" in caplog.text


def test_serve_defaults_to_port_8000(indexed: Path, fake_uvicorn: FakeUvicorn) -> None:
    cli.main(["serve"])
    assert fake_uvicorn.calls[0]["port"] == 8000


def test_served_app_answers_with_the_configured_models(
    indexed: Path, fake_uvicorn: FakeUvicorn
) -> None:
    write(indexed / ".env", "RAGNOSIS_LLM_MODEL=gemma4:e4b\n")
    cli.main(["serve"])
    client = TestClient(fake_uvicorn.calls[0]["app"], base_url="http://localhost")

    response = client.post("/api/ask", json={"question": "Irish pirate Calico Jack", "k": 1})

    lines = [json.loads(line) for line in response.text.splitlines()]
    assert "".join(e["text"] for e in lines if e["type"] == "text") == (
        "Anne Bonny was an Irish pirate [1]."
    )
    assert lines[-1]["sources"][0]["title"] == "Anne Bonny"
    assert FakeOllama.created == [("gemma4:e4b", "http://localhost:11434")]


def test_serve_without_index_exits_with_usage_code(
    fake_uvicorn: FakeUvicorn, caplog: pytest.LogCaptureFixture
) -> None:
    assert cli.main(["serve"]) == cli.EXIT_USAGE
    assert "run index first" in caplog.text
    assert fake_uvicorn.calls == []


def test_serve_refuses_index_built_with_another_model(
    indexed: Path, fake_uvicorn: FakeUvicorn
) -> None:
    write(indexed / ".env", "RAGNOSIS_EMBEDDING_MODEL=other/model\n")
    assert cli.main(["serve"]) == cli.EXIT_USAGE
    assert fake_uvicorn.calls == []


@pytest.mark.parametrize("port", ["0", "65536", "-1", "http"])
def test_serve_rejects_invalid_port(port: str, fake_uvicorn: FakeUvicorn) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["serve", "--port", port])
    assert exit_info.value.code == cli.EXIT_USAGE


# --- evaluate ---

CASES = """
[[case]]
id = "bonny-en"
question = "Who was Anne Bonny?"
language = "en"
sources = ["Anne Bonny"]
mentions = ["Irish"]

[[case]]
id = "bonny-sv"
question = "Vem var Anne Bonny?"
language = "sv"
sources = ["Anne Bonny"]
mentions = ["irländsk"]
"""


@pytest.fixture
def cases_file(indexed: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Two cases: the fake model's English answer passes one and fails the other."""
    readings = iter([0.0, 31.0, 100.0, 133.0])
    monkeypatch.setattr(cli, "_clock", lambda: next(readings))
    monkeypatch.setattr(
        FakeOllama, "reply", ("Anne Bonny was an Irish pirate ", "and she sailed with Jack [1].")
    )
    return write(indexed / "cases.toml", CASES)


def test_evaluate_prints_a_scorecard_and_fails_when_a_case_fails(
    cases_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()

    assert cli.main(["evaluate", "--cases", str(cases_file)]) == cli.EXIT_FAILURES

    out = capsys.readouterr().out
    assert "PASS  bonny-en" in out
    assert "  31 s" in out
    assert "FAIL  bonny-sv" in out
    assert "- is in en, expected sv" in out
    assert "- mentions none of ['irländsk']" in out
    assert "answer: Anne Bonny was an Irish pirate and she sailed with Jack [1]." in out
    assert out.endswith("\n1 of 2 passed, 2 required: NOT met · median 32 s · total 1.1 min\n")


def test_full_run_passes_when_it_reaches_the_required_level(
    cases_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write(cases_file, "required = 1\n" + CASES)

    assert cli.main(["evaluate", "--cases", str(cases_file)]) == cli.EXIT_OK

    assert "1 of 2 passed, 1 required: met" in capsys.readouterr().out


def test_chosen_cases_must_all_pass_whatever_the_level(
    cases_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write(cases_file, "required = 1\n" + CASES)

    code = cli.main(["evaluate", "--cases", str(cases_file), "--case", "bonny-sv"])

    assert code == cli.EXIT_FAILURES
    assert "0 of 1 passed, 1 required: NOT met" in capsys.readouterr().out


def test_evaluate_passes_when_every_case_passes(
    cases_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["evaluate", "--cases", str(cases_file), "--case", "bonny-en"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "bonny-sv" not in out
    assert "1 of 1 passed" in out


def test_evaluate_loads_the_model_before_the_first_case(cases_file: Path) -> None:
    cli.main(["evaluate", "--cases", str(cases_file), "--case", "bonny-en"])
    assert FakeOllama.events[0] == "load"


def test_evaluate_records_a_model_failure_and_carries_on(
    cases_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(FakeOllama, "error", LLMError("Lost connection to Ollama"))

    assert cli.main(["evaluate", "--cases", str(cases_file)]) == cli.EXIT_FAILURES

    out = capsys.readouterr().out
    assert out.count("- the model failed: Lost connection to Ollama") == 2
    assert "answer: (none)" in out
    assert "0 of 2 passed" in out


@pytest.mark.parametrize(
    "error",
    [LLMUnavailableError("start Ollama"), ModelNotFoundError("run: ollama pull")],
    ids=["ollama-down", "model-missing"],
)
def test_evaluate_stops_when_the_model_cannot_be_reached(
    cases_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
) -> None:
    monkeypatch.setattr(FakeOllama, "error", error)
    assert cli.main(["evaluate", "--cases", str(cases_file)]) == cli.EXIT_USAGE
    assert str(error) in caplog.text


def test_evaluate_rejects_unknown_case_ids(
    cases_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    code = cli.main(["evaluate", "--cases", str(cases_file), "--case", "bonny-en", "--case", "x"])
    assert code == cli.EXIT_USAGE
    assert "Unknown case ids: ['x']" in caplog.text


@pytest.mark.parametrize("content", [None, "not = [toml"], ids=["missing", "malformed"])
def test_evaluate_rejects_an_unreadable_cases_file(
    indexed: Path, caplog: pytest.LogCaptureFixture, content: str | None
) -> None:
    path = indexed / "cases.toml"
    if content is not None:
        write(path, content)
    assert cli.main(["evaluate", "--cases", str(path)]) == cli.EXIT_USAGE
    assert "Cannot read cases" in caplog.text


def test_evaluate_without_index_exits_with_usage_code(
    project_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = write(project_dir / "cases.toml", CASES)
    assert cli.main(["evaluate", "--cases", str(path)]) == cli.EXIT_USAGE
    assert "run index first" in caplog.text


def test_evaluate_refuses_index_built_with_another_model(cases_file: Path, indexed: Path) -> None:
    write(indexed / ".env", "RAGNOSIS_EMBEDDING_MODEL=other/model\n")
    assert cli.main(["evaluate", "--cases", str(cases_file)]) == cli.EXIT_USAGE


def test_evaluate_reads_the_committed_cases_by_default(
    indexed: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo_cases = Path(__file__).resolve().parents[1] / "evaluation" / "answers.toml"
    write(indexed / "evaluation" / "answers.toml", repo_cases.read_text(encoding="utf-8"))
    monkeypatch.setattr(cli, "_clock", lambda: 0.0)

    cli.main(["evaluate", "--case", "japan-capital-en"])

    assert "japan-capital-en" in capsys.readouterr().out


def test_failed_case_shows_the_rewritten_query(
    cases_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.main(["evaluate", "--cases", str(cases_file), "--case", "bonny-sv"])
    out = capsys.readouterr().out
    assert "searched for: Anne Bonny was an Irish pirate and she sailed with Jack [1]." in out


def test_format_summary_without_cases() -> None:
    assert cli.format_summary([], required=1) == "\nNo cases were run.\n"


def test_format_verdict_shortens_long_answers() -> None:
    case = Case(id="long", question="Q?", language="en", mentions=("x",))
    verdict = Verdict(case, Answer("Q?", "word " * 100, []), ["mentions none of ['x']"])
    preview = cli.format_verdict(verdict, 5.0).splitlines()[-1]
    assert preview.endswith("…")
    assert len(preview) <= len("      answer: ") + cli.PREVIEW_CHARS + 1


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

"""Tests for the command line, with the Wikipedia client replaced by a fake."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import TracebackType

import pytest
from factories import make_article

from ragnosis import cli
from ragnosis.article_store import ArticleStore
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
    """Run each test in an empty project folder with a fake Wikipedia."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "WikipediaClient", FakeWikipediaClient)
    monkeypatch.setattr(
        FakeWikipediaClient,
        "articles",
        {
            "Blackbeard": make_article(title="Blackbeard", page_id=1),
            "Anne Bonny": make_article(title="Anne Bonny", page_id=2),
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

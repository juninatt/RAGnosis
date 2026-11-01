"""Tests for reading settings from the environment."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from ragnosis.settings import (
    DATA_DIR_VAR,
    DEFAULT_DATA_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_KEEP_ALIVE,
    DEFAULT_LLM_MODEL,
    DEFAULT_MIN_SCORE,
    DEFAULT_NAME_LANGUAGES,
    DEFAULT_OLLAMA_URL,
    DEFAULT_QUERY_REWRITE,
    DEFAULT_TOPICS_FILE,
    DEFAULT_WIKI_LANGUAGE,
    EMBEDDING_MODEL_VAR,
    LLM_KEEP_ALIVE_VAR,
    LLM_MODEL_VAR,
    MIN_SCORE_VAR,
    NAME_LANGUAGES_VAR,
    OLLAMA_URL_VAR,
    QUERY_REWRITE_VAR,
    TOPICS_FILE_VAR,
    WIKI_LANGUAGE_VAR,
    Settings,
    is_valid_language,
)

ALL_VARS = (
    DATA_DIR_VAR,
    TOPICS_FILE_VAR,
    WIKI_LANGUAGE_VAR,
    EMBEDDING_MODEL_VAR,
    OLLAMA_URL_VAR,
    LLM_MODEL_VAR,
    LLM_KEEP_ALIVE_VAR,
    QUERY_REWRITE_VAR,
    MIN_SCORE_VAR,
    NAME_LANGUAGES_VAR,
)


def test_query_rewrite_defaults_to_always() -> None:
    # The scorecard passed 15 of 16 with "always", 13 with "swedish" (docs/evaluation.md).
    assert Settings.from_env({}).query_rewrite == DEFAULT_QUERY_REWRITE == "always"


@pytest.mark.parametrize("value", ["always", "swedish", "never"])
def test_query_rewrite_accepts_its_modes(value: str) -> None:
    assert Settings.from_env({QUERY_REWRITE_VAR: value}).query_rewrite == value


@pytest.mark.parametrize("value", ["yes", "Swedish", "sometimes", " never"])
def test_query_rewrite_rejects_other_values(value: str) -> None:
    with pytest.raises(ValueError, match="Query rewrite must be one of always, swedish, never"):
        Settings.from_env({QUERY_REWRITE_VAR: value})


def test_min_score_defaults_to_the_measured_limit() -> None:
    # Between the known questions' best scores (docs/evaluation.md, decision 0010).
    assert Settings.from_env({}).min_score == DEFAULT_MIN_SCORE == 0.83


@pytest.mark.parametrize(("value", "score"), [("0", 0.0), ("0.9", 0.9), ("1", 1.0), (".5", 0.5)])
def test_min_score_accepts_numbers_from_0_to_1(value: str, score: float) -> None:
    assert Settings.from_env({MIN_SCORE_VAR: value}).min_score == score


@pytest.mark.parametrize("value", ["-0.1", "1.5", "nan", "inf"])
def test_min_score_rejects_numbers_outside_0_to_1(value: str) -> None:
    with pytest.raises(ValueError, match="Minimum score must be from 0 to 1"):
        Settings.from_env({MIN_SCORE_VAR: value})


@pytest.mark.parametrize("value", ["high", "0,83", "83%"])
def test_min_score_rejects_text_that_is_not_a_number(value: str) -> None:
    with pytest.raises(ValueError, match="Minimum score must be a number from 0 to 1"):
        Settings.from_env({MIN_SCORE_VAR: value})


def test_name_languages_default_to_english_and_swedish() -> None:
    assert Settings.from_env({}).name_languages == DEFAULT_NAME_LANGUAGES == ("en", "sv")


@pytest.mark.parametrize(
    ("value", "languages"),
    [("sv", ("sv",)), ("en,sv,de", ("en", "sv", "de")), (" sv , en ", ("sv", "en"))],
)
def test_name_languages_are_read_as_a_list(value: str, languages: tuple[str, ...]) -> None:
    assert Settings.from_env({NAME_LANGUAGES_VAR: value}).name_languages == languages


@pytest.mark.parametrize("value", ["EN", "en;sv", "en,sv.evil.com"])
def test_name_languages_reject_other_values(value: str) -> None:
    with pytest.raises(ValueError, match="Name languages must be language codes"):
        Settings.from_env({NAME_LANGUAGES_VAR: value})


def test_no_name_languages_are_rejected() -> None:
    with pytest.raises(ValueError, match="Name languages"):
        Settings(name_languages=())


def test_keep_alive_defaults_to_thirty_minutes() -> None:
    assert Settings.from_env({}).llm_keep_alive == DEFAULT_LLM_KEEP_ALIVE == "30m"


@pytest.mark.parametrize("value", ["30m", "1h", "1h30m", "90s", "500ms", "2.5h", "0", "-1"])
def test_keep_alive_accepts_ollama_durations(value: str) -> None:
    assert Settings.from_env({LLM_KEEP_ALIVE_VAR: value}).llm_keep_alive == value


@pytest.mark.parametrize("value", ["30", "forever", "30 m", "-5m", "1d", "m", " 30m"])
def test_keep_alive_rejects_other_values(value: str) -> None:
    with pytest.raises(ValueError, match="Keep-alive"):
        Settings.from_env({LLM_KEEP_ALIVE_VAR: value})


def test_defaults_when_environment_is_empty() -> None:
    settings = Settings.from_env({})
    assert settings.data_dir == DEFAULT_DATA_DIR
    assert settings.topics_file == DEFAULT_TOPICS_FILE
    assert settings.wiki_language == DEFAULT_WIKI_LANGUAGE
    assert settings.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert settings.ollama_url == DEFAULT_OLLAMA_URL
    assert settings.llm_model == DEFAULT_LLM_MODEL


def test_reads_values_from_environment() -> None:
    settings = Settings.from_env(
        {
            DATA_DIR_VAR: "/tmp/rag",
            TOPICS_FILE_VAR: "my-topics.txt",
            WIKI_LANGUAGE_VAR: "sv",
            EMBEDDING_MODEL_VAR: "BAAI/bge-m3",
            OLLAMA_URL_VAR: "https://ollama.example:8443",
            LLM_MODEL_VAR: "gemma4:e4b",
        }
    )
    assert settings.data_dir == Path("/tmp/rag")
    assert settings.topics_file == Path("my-topics.txt")
    assert settings.wiki_language == "sv"
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.ollama_url == "https://ollama.example:8443"
    assert settings.llm_model == "gemma4:e4b"


def test_blank_language_model_raises() -> None:
    with pytest.raises(ValueError, match="Language model"):
        Settings.from_env({LLM_MODEL_VAR: " "})


@pytest.mark.parametrize(
    "url", ["localhost:11434", "ftp://localhost:11434", "http://", "file:///tmp/ollama"]
)
def test_ollama_url_must_be_http_with_host(url: str) -> None:
    with pytest.raises(ValueError, match="Ollama URL"):
        Settings.from_env({OLLAMA_URL_VAR: url})


def test_empty_values_fall_back_to_defaults() -> None:
    settings = Settings.from_env(dict.fromkeys(ALL_VARS, ""))
    assert settings == Settings()


def test_blank_embedding_model_raises() -> None:
    with pytest.raises(ValueError, match="Embedding model"):
        Settings.from_env({EMBEDDING_MODEL_VAR: "   "})


def test_index_dir_is_inside_data_dir() -> None:
    assert Settings(data_dir=Path("store")).index_dir == Path("store/index")


def test_reads_os_environ_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(WIKI_LANGUAGE_VAR, "de")
    assert Settings.from_env().wiki_language == "de"


def test_articles_dir_is_inside_data_dir() -> None:
    assert Settings(data_dir=Path("store")).articles_dir == Path("store/articles")


def test_settings_are_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        Settings().wiki_language = "sv"  # type: ignore[misc]


def test_invalid_language_from_environment_raises() -> None:
    with pytest.raises(ValueError, match="language code"):
        Settings.from_env({WIKI_LANGUAGE_VAR: "en.evil.com"})


@pytest.mark.parametrize("code", ["en", "sv", "als", "zh-yue", "be-tarask"])
def test_accepts_real_language_codes(code: str) -> None:
    assert is_valid_language(code)


@pytest.mark.parametrize(
    "code", ["", "e", "EN", "engl", "en.evil.com", "en/x", "en wiki", "-en", "en-"]
)
def test_rejects_malformed_language_codes(code: str) -> None:
    assert not is_valid_language(code)

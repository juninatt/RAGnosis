"""Tests for reading settings from the environment."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from ragnosis.settings import (
    DATA_DIR_VAR,
    DEFAULT_DATA_DIR,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOPICS_FILE,
    DEFAULT_WIKI_LANGUAGE,
    EMBEDDING_MODEL_VAR,
    TOPICS_FILE_VAR,
    WIKI_LANGUAGE_VAR,
    Settings,
    is_valid_language,
)

ALL_VARS = (DATA_DIR_VAR, TOPICS_FILE_VAR, WIKI_LANGUAGE_VAR, EMBEDDING_MODEL_VAR)


def test_defaults_when_environment_is_empty() -> None:
    settings = Settings.from_env({})
    assert settings.data_dir == DEFAULT_DATA_DIR
    assert settings.topics_file == DEFAULT_TOPICS_FILE
    assert settings.wiki_language == DEFAULT_WIKI_LANGUAGE
    assert settings.embedding_model == DEFAULT_EMBEDDING_MODEL


def test_reads_values_from_environment() -> None:
    settings = Settings.from_env(
        {
            DATA_DIR_VAR: "/tmp/rag",
            TOPICS_FILE_VAR: "my-topics.txt",
            WIKI_LANGUAGE_VAR: "sv",
            EMBEDDING_MODEL_VAR: "BAAI/bge-m3",
        }
    )
    assert settings.data_dir == Path("/tmp/rag")
    assert settings.topics_file == Path("my-topics.txt")
    assert settings.wiki_language == "sv"
    assert settings.embedding_model == "BAAI/bge-m3"


def test_empty_values_fall_back_to_defaults() -> None:
    settings = Settings.from_env(dict.fromkeys(ALL_VARS, ""))
    assert settings == Settings()


def test_blank_embedding_model_raises() -> None:
    with pytest.raises(ValueError, match="Embedding model"):
        Settings.from_env({EMBEDDING_MODEL_VAR: "   "})


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

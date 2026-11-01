"""Application settings read from the environment.

Settings are read once into an immutable object and handed to the code that
needs them, instead of every module reading ``os.environ`` on its own. That keeps
all settings in one place and lets tests build any configuration directly.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from urllib.parse import urlsplit

from ragnosis.llm import DEFAULT_KEEP_ALIVE

DATA_DIR_VAR = "RAGNOSIS_DATA_DIR"
TOPICS_FILE_VAR = "RAGNOSIS_TOPICS_FILE"
WIKI_LANGUAGE_VAR = "RAGNOSIS_WIKI_LANGUAGE"
EMBEDDING_MODEL_VAR = "RAGNOSIS_EMBEDDING_MODEL"
OLLAMA_URL_VAR = "RAGNOSIS_OLLAMA_URL"
LLM_MODEL_VAR = "RAGNOSIS_LLM_MODEL"
LLM_KEEP_ALIVE_VAR = "RAGNOSIS_LLM_KEEP_ALIVE"
QUERY_REWRITE_VAR = "RAGNOSIS_QUERY_REWRITE"
MIN_SCORE_VAR = "RAGNOSIS_MIN_SCORE"
NAME_LANGUAGES_VAR = "RAGNOSIS_NAME_LANGUAGES"

# When to rewrite a question into an English search query before retrieval
# (docs/decisions/0008): every question, only Swedish ones, or never.
RewriteMode = Literal["always", "swedish", "never"]
REWRITE_MODES: tuple[RewriteMode, ...] = ("always", "swedish", "never")

DEFAULT_DATA_DIR = Path("data")
DEFAULT_TOPICS_FILE = Path("topics/pirates-and-antiquity-europe.txt")
DEFAULT_WIKI_LANGUAGE = "en"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "qwen3.5:4b"
DEFAULT_LLM_KEEP_ALIVE = DEFAULT_KEEP_ALIVE
DEFAULT_QUERY_REWRITE: RewriteMode = "always"
# Below this search score no passage is about the question, so it is declined
# without the model. Set between the known questions' scores (0010); it
# depends on the embedding model and the passages.
DEFAULT_MIN_SCORE = 0.83
# The languages of the other names fetched from Wikidata (0011): those the
# questions are asked in.
DEFAULT_NAME_LANGUAGES = ("en", "sv")

# The language code becomes part of the API host name, so only accept what real
# Wikipedia codes look like ("en", "sv", "zh-yue") and nothing that could change the URL.
_LANGUAGE_PATTERN = re.compile(r"[a-z]{2,3}(-[a-z]+)*")
# Durations as Ollama reads them: "30m", "1h30m", "90s", or "-1" for "keep forever".
_DURATION_PATTERN = re.compile(r"-1|0|(\d+(\.\d+)?(ms|s|m|h))+")


def is_valid_language(code: str) -> bool:
    """Return whether ``code`` looks like a Wikipedia language code."""
    return _LANGUAGE_PATTERN.fullmatch(code) is not None


@dataclass(frozen=True)
class Settings:
    """Where data lives, which Wikipedia to read from and which models to use."""

    data_dir: Path = DEFAULT_DATA_DIR
    topics_file: Path = DEFAULT_TOPICS_FILE
    wiki_language: str = DEFAULT_WIKI_LANGUAGE
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    ollama_url: str = DEFAULT_OLLAMA_URL
    llm_model: str = DEFAULT_LLM_MODEL
    llm_keep_alive: str = DEFAULT_LLM_KEEP_ALIVE
    query_rewrite: RewriteMode = DEFAULT_QUERY_REWRITE
    min_score: float = DEFAULT_MIN_SCORE
    name_languages: tuple[str, ...] = DEFAULT_NAME_LANGUAGES

    def __post_init__(self) -> None:
        if not is_valid_language(self.wiki_language):
            raise ValueError(f"Invalid Wikipedia language code: {self.wiki_language!r}")
        if not self.embedding_model.strip():
            raise ValueError("Embedding model name must not be blank")
        if not self.llm_model.strip():
            raise ValueError("Language model name must not be blank")
        url = urlsplit(self.ollama_url)
        if url.scheme not in ("http", "https") or not url.hostname:
            raise ValueError(f"Ollama URL must be an http(s) address: {self.ollama_url!r}")
        if not _DURATION_PATTERN.fullmatch(self.llm_keep_alive):
            raise ValueError(
                f"Keep-alive must be a duration like 30m, 1h or -1: {self.llm_keep_alive!r}"
            )
        if self.query_rewrite not in REWRITE_MODES:
            raise ValueError(
                f"Query rewrite must be one of {', '.join(REWRITE_MODES)}: {self.query_rewrite!r}"
            )
        if not 0 <= self.min_score <= 1:
            raise ValueError(f"Minimum score must be from 0 to 1: {self.min_score!r}")
        if not self.name_languages or not all(map(is_valid_language, self.name_languages)):
            raise ValueError(
                f"Name languages must be language codes like en,sv: {self.name_languages!r}"
            )

    @property
    def articles_dir(self) -> Path:
        """Folder for fetched articles, kept apart from later derived data."""
        return self.data_dir / "articles"

    @property
    def index_dir(self) -> Path:
        """Folder for the vector index, which can always be rebuilt from the articles."""
        return self.data_dir / "index"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Build settings from environment variables; empty values mean "use the default"."""
        env = os.environ if environ is None else environ
        return cls(
            data_dir=Path(env.get(DATA_DIR_VAR) or DEFAULT_DATA_DIR),
            topics_file=Path(env.get(TOPICS_FILE_VAR) or DEFAULT_TOPICS_FILE),
            wiki_language=env.get(WIKI_LANGUAGE_VAR) or DEFAULT_WIKI_LANGUAGE,
            embedding_model=env.get(EMBEDDING_MODEL_VAR) or DEFAULT_EMBEDDING_MODEL,
            ollama_url=env.get(OLLAMA_URL_VAR) or DEFAULT_OLLAMA_URL,
            llm_model=env.get(LLM_MODEL_VAR) or DEFAULT_LLM_MODEL,
            llm_keep_alive=env.get(LLM_KEEP_ALIVE_VAR) or DEFAULT_LLM_KEEP_ALIVE,
            # Checked in __post_init__; the cast only tells the type checker.
            query_rewrite=cast(RewriteMode, env.get(QUERY_REWRITE_VAR) or DEFAULT_QUERY_REWRITE),
            min_score=_score(env.get(MIN_SCORE_VAR) or str(DEFAULT_MIN_SCORE)),
            name_languages=_languages(env.get(NAME_LANGUAGES_VAR)) or DEFAULT_NAME_LANGUAGES,
        )


def _languages(text: str | None) -> tuple[str, ...]:
    """Comma-separated language codes, ignoring spaces: "en, sv" is ("en", "sv")."""
    return tuple(code.strip() for code in (text or "").split(",") if code.strip())


def _score(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"Minimum score must be a number from 0 to 1: {text!r}") from None

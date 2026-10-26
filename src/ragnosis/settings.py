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

DATA_DIR_VAR = "RAGNOSIS_DATA_DIR"
TOPICS_FILE_VAR = "RAGNOSIS_TOPICS_FILE"
WIKI_LANGUAGE_VAR = "RAGNOSIS_WIKI_LANGUAGE"
EMBEDDING_MODEL_VAR = "RAGNOSIS_EMBEDDING_MODEL"

DEFAULT_DATA_DIR = Path("data")
DEFAULT_TOPICS_FILE = Path("topics/pirates-and-antiquity-europe.txt")
DEFAULT_WIKI_LANGUAGE = "en"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# The language code becomes part of the API host name, so only accept what real
# Wikipedia codes look like ("en", "sv", "zh-yue") and nothing that could change the URL.
_LANGUAGE_PATTERN = re.compile(r"[a-z]{2,3}(-[a-z]+)*")


def is_valid_language(code: str) -> bool:
    """Return whether ``code`` looks like a Wikipedia language code."""
    return _LANGUAGE_PATTERN.fullmatch(code) is not None


@dataclass(frozen=True)
class Settings:
    """Where data lives, which Wikipedia to read from and which model embeds text."""

    data_dir: Path = DEFAULT_DATA_DIR
    topics_file: Path = DEFAULT_TOPICS_FILE
    wiki_language: str = DEFAULT_WIKI_LANGUAGE
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    def __post_init__(self) -> None:
        if not is_valid_language(self.wiki_language):
            raise ValueError(f"Invalid Wikipedia language code: {self.wiki_language!r}")
        if not self.embedding_model.strip():
            raise ValueError("Embedding model name must not be blank")

    @property
    def articles_dir(self) -> Path:
        """Folder for fetched articles, kept apart from later derived data."""
        return self.data_dir / "articles"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Build settings from environment variables; empty values mean "use the default"."""
        env = os.environ if environ is None else environ
        return cls(
            data_dir=Path(env.get(DATA_DIR_VAR) or DEFAULT_DATA_DIR),
            topics_file=Path(env.get(TOPICS_FILE_VAR) or DEFAULT_TOPICS_FILE),
            wiki_language=env.get(WIKI_LANGUAGE_VAR) or DEFAULT_WIKI_LANGUAGE,
            embedding_model=env.get(EMBEDDING_MODEL_VAR) or DEFAULT_EMBEDDING_MODEL,
        )

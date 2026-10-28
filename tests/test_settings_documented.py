"""Guards that every setting is documented, with the same default everywhere.

Settings are part of what 1.0 promises (docs/decisions/0009), so the code, the
committed .env and the README table must name the same variables and agree on
their defaults.
"""

from __future__ import annotations

import re
from pathlib import Path

from dotenv import dotenv_values

from ragnosis import settings
from ragnosis.logging_config import DEFAULT_LEVEL, LOG_LEVEL_ENV_VAR

REPO_ROOT = Path(__file__).resolve().parents[1]

# Each setting the code reads, with its default as written in .env and README.
CODE_DEFAULTS = {
    LOG_LEVEL_ENV_VAR: DEFAULT_LEVEL,
    settings.DATA_DIR_VAR: settings.DEFAULT_DATA_DIR.as_posix(),
    settings.TOPICS_FILE_VAR: settings.DEFAULT_TOPICS_FILE.as_posix(),
    settings.WIKI_LANGUAGE_VAR: settings.DEFAULT_WIKI_LANGUAGE,
    settings.EMBEDDING_MODEL_VAR: settings.DEFAULT_EMBEDDING_MODEL,
    settings.OLLAMA_URL_VAR: settings.DEFAULT_OLLAMA_URL,
    settings.LLM_MODEL_VAR: settings.DEFAULT_LLM_MODEL,
    settings.LLM_KEEP_ALIVE_VAR: settings.DEFAULT_LLM_KEEP_ALIVE,
    settings.QUERY_REWRITE_VAR: settings.DEFAULT_QUERY_REWRITE,
}


def readme_defaults() -> dict[str, str]:
    """The README settings table as {variable: default}."""
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| `(RAGNOSIS_[A-Z_]+)` \| `([^`]*)` \|", text, re.MULTILINE)
    return dict(rows)


def env_defaults() -> dict[str, str | None]:
    return dotenv_values(REPO_ROOT / ".env")


def test_every_setting_in_the_code_is_known_to_this_test() -> None:
    # A new *_VAR in settings.py must be added above, and so to the docs.
    in_module = {
        value
        for name, value in vars(settings).items()
        if name.endswith("_VAR") and isinstance(value, str)
    }
    assert in_module | {LOG_LEVEL_ENV_VAR} == set(CODE_DEFAULTS)


def test_readme_documents_every_setting_with_the_code_default() -> None:
    assert readme_defaults() == CODE_DEFAULTS


def test_env_file_lists_every_setting_with_the_code_default() -> None:
    assert env_defaults() == CODE_DEFAULTS

"""Tests for env file loading and for the committed ``.env`` itself."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from dotenv import dotenv_values

from ragnosis.env import LOCAL_ENV_FILE, SHARED_ENV_FILE, load_env

VAR = "RAGNOSIS_TEST_SETTING"
REPO_ROOT = Path(__file__).resolve().parents[1]
SECRET_LIKE = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL", re.IGNORECASE)


@pytest.fixture(autouse=True)
def restore_environ() -> Iterator[None]:
    """load_env writes to os.environ directly, so undo it after every test."""
    saved = os.environ.copy()
    os.environ.pop(VAR, None)
    yield
    os.environ.clear()
    os.environ.update(saved)


def write(directory: Path, name: str, content: str) -> Path:
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_shared_file(tmp_path: Path) -> None:
    write(tmp_path, SHARED_ENV_FILE, f"{VAR}=shared\n")
    load_env(tmp_path)
    assert os.environ[VAR] == "shared"


def test_local_file_overrides_shared_file(tmp_path: Path) -> None:
    write(tmp_path, SHARED_ENV_FILE, f"{VAR}=shared\n")
    write(tmp_path, LOCAL_ENV_FILE, f"{VAR}=local\n")
    load_env(tmp_path)
    assert os.environ[VAR] == "local"


def test_real_environment_overrides_both_files(tmp_path: Path) -> None:
    write(tmp_path, SHARED_ENV_FILE, f"{VAR}=shared\n")
    write(tmp_path, LOCAL_ENV_FILE, f"{VAR}=local\n")
    os.environ[VAR] = "real"
    load_env(tmp_path)
    assert os.environ[VAR] == "real"


def test_local_file_alone_is_loaded(tmp_path: Path) -> None:
    local = write(tmp_path, LOCAL_ENV_FILE, f"{VAR}=local\n")
    assert load_env(tmp_path) == [local]
    assert os.environ[VAR] == "local"


def test_returns_loaded_files_highest_precedence_first(tmp_path: Path) -> None:
    shared = write(tmp_path, SHARED_ENV_FILE, "")
    local = write(tmp_path, LOCAL_ENV_FILE, "")
    assert load_env(tmp_path) == [local, shared]


def test_missing_files_are_skipped(tmp_path: Path) -> None:
    assert load_env(tmp_path) == []
    assert VAR not in os.environ


def test_directory_named_like_env_file_is_ignored(tmp_path: Path) -> None:
    (tmp_path / SHARED_ENV_FILE).mkdir()
    assert load_env(tmp_path) == []


def test_defaults_to_current_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write(tmp_path, SHARED_ENV_FILE, f"{VAR}=from-cwd\n")
    monkeypatch.chdir(tmp_path)
    load_env()
    assert os.environ[VAR] == "from-cwd"


# Guards for the committed .env: it is shared through git, so it must stay free of secrets.


def committed_env() -> dict[str, str | None]:
    return dotenv_values(REPO_ROOT / SHARED_ENV_FILE)


def test_committed_env_file_exists() -> None:
    assert (REPO_ROOT / SHARED_ENV_FILE).is_file()


def test_committed_env_has_no_secret_like_keys() -> None:
    offending = [key for key in committed_env() if SECRET_LIKE.search(key)]
    assert offending == [], f"Move these to {LOCAL_ENV_FILE}: {offending}"


def test_committed_env_keys_use_project_prefix() -> None:
    unprefixed = [key for key in committed_env() if not key.startswith("RAGNOSIS_")]
    assert unprefixed == []

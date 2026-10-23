"""Tests for the central logging setup."""

from __future__ import annotations

import logging

import pytest

from ragnosis.logging_config import (
    LOG_LEVEL_ENV_VAR,
    PACKAGE_LOGGER,
    setup_logging,
)


def test_defaults_to_info_without_argument_or_env() -> None:
    assert setup_logging().level == logging.INFO


def test_reads_level_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOG_LEVEL_ENV_VAR, "DEBUG")
    assert setup_logging().level == logging.DEBUG


def test_argument_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOG_LEVEL_ENV_VAR, "DEBUG")
    assert setup_logging("ERROR").level == logging.ERROR


def test_empty_env_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOG_LEVEL_ENV_VAR, "")
    assert setup_logging().level == logging.INFO


@pytest.mark.parametrize("name", ["debug", "Warning", "eRrOr"])
def test_level_names_are_case_insensitive(name: str) -> None:
    assert setup_logging(name).level == logging.getLevelNamesMapping()[name.upper()]


@pytest.mark.parametrize("name", ["verbose", "LOUD", "10"])
def test_unknown_level_argument_raises(name: str) -> None:
    with pytest.raises(ValueError, match="Unknown log level"):
        setup_logging(name)


def test_unknown_level_in_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOG_LEVEL_ENV_VAR, "chatty")
    with pytest.raises(ValueError, match="CHATTY"):
        setup_logging()


def test_repeated_setup_does_not_duplicate_handlers() -> None:
    setup_logging()
    logger = setup_logging()
    assert len(logger.handlers) == 1


def test_keeps_handlers_added_by_others() -> None:
    foreign = logging.NullHandler()
    logging.getLogger(PACKAGE_LOGGER).addHandler(foreign)
    logger = setup_logging()
    assert foreign in logger.handlers
    assert len(logger.handlers) == 2


def test_writes_formatted_records_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging()
    logging.getLogger(f"{PACKAGE_LOGGER}.ingest").info("fetched %d articles", 3)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "INFO" in captured.err
    assert "ragnosis.ingest: fetched 3 articles" in captured.err


def test_filters_records_below_level(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("WARNING")
    log = logging.getLogger(f"{PACKAGE_LOGGER}.ingest")
    log.info("hidden")
    log.warning("shown")

    err = capsys.readouterr().err
    assert "hidden" not in err
    assert "shown" in err


def test_does_not_change_root_logger() -> None:
    root = logging.getLogger()
    handlers_before = list(root.handlers)
    level_before = root.level

    setup_logging("DEBUG")

    assert root.handlers == handlers_before
    assert root.level == level_before

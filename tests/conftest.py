"""Shared fixtures that keep tests independent of each other and of the machine."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator

import pytest

from ragnosis.logging_config import PACKAGE_LOGGER


@pytest.fixture(autouse=True)
def isolated_environ() -> Iterator[None]:
    """Start every test without RAGNOSIS_* variables and undo any changes afterwards.

    Code under test (``load_env``) writes to ``os.environ`` directly, which
    monkeypatch cannot undo, and a developer's own settings must not leak in.
    """
    saved = os.environ.copy()
    for key in [key for key in os.environ if key.startswith("RAGNOSIS_")]:
        del os.environ[key]
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture(autouse=True)
def reset_package_logger() -> Iterator[None]:
    """Remove handlers and levels that ``setup_logging`` added during a test."""
    yield
    logger = logging.getLogger(PACKAGE_LOGGER)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)

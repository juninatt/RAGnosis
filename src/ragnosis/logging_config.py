"""Logging setup for RAGnosis.

Every module logs through ``logging.getLogger(__name__)``. This module is the one
place that decides where those records go and what they look like, so verbosity
can be changed through configuration without touching application code.
"""

from __future__ import annotations

import logging
import os
import sys

LOG_LEVEL_ENV_VAR = "RAGNOSIS_LOG_LEVEL"
DEFAULT_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

PACKAGE_LOGGER = "ragnosis"
_HANDLER_NAME = "ragnosis-stderr"


def setup_logging(level: str | None = None) -> logging.Logger:
    """Configure the package logger and return it.

    The level is taken from ``level``, then the ``RAGNOSIS_LOG_LEVEL`` environment
    variable, then ``INFO``. Records go to stderr so stdout stays free for output
    other programs may consume.

    Calling this more than once is safe: our handler is replaced rather than
    duplicated, otherwise re-running setup (tests, notebooks) would print every
    line several times. Only the ``ragnosis`` logger is touched, so third-party
    libraries keep their own settings.

    Raises:
        ValueError: if the level name is unknown. A typo in configuration should
            fail loudly instead of silently logging at the wrong level.
    """
    name = (level or os.environ.get(LOG_LEVEL_ENV_VAR) or DEFAULT_LEVEL).upper()
    numeric_level = logging.getLevelNamesMapping().get(name)
    if numeric_level is None:
        raise ValueError(f"Unknown log level: {name!r}")

    logger = logging.getLogger(PACKAGE_LOGGER)
    for handler in list(logger.handlers):
        if handler.get_name() == _HANDLER_NAME:
            logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(numeric_level)
    return logger

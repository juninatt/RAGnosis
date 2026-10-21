"""Load configuration from env files.

Non-sensitive settings live in the committed ``.env`` so every clone runs with
the same defaults. Secrets and personal overrides go in ``.env.local``, which git
ignores. Real environment variables always win, so CI or a one-off terminal
session can override any setting without editing files.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

SHARED_ENV_FILE = ".env"
LOCAL_ENV_FILE = ".env.local"


def load_env(directory: Path | None = None) -> list[Path]:
    """Load the env files in ``directory`` (default: current directory) into ``os.environ``.

    Precedence, highest first: real environment, ``.env.local``, ``.env``.
    Missing files are skipped, since a fresh clone has no ``.env.local``.

    Returns the files that were loaded, highest precedence first. They are
    returned rather than logged because logging reads its level from these
    files and cannot be configured until they are loaded.
    """
    base = directory if directory is not None else Path.cwd()
    loaded: list[Path] = []
    # Load the highest precedence first: load_dotenv never replaces a variable
    # that is already set, so earlier files win over later ones.
    for name in (LOCAL_ENV_FILE, SHARED_ENV_FILE):
        path = base / name
        if path.is_file():
            load_dotenv(path, override=False)
            loaded.append(path)
    return loaded

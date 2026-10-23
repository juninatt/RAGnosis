"""Keep fetched articles on disk, one JSON file per article.

Plain JSON files are easy to inspect, diff and delete by hand, and later steps
(splitting, indexing) can be rerun from them without downloading again.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote

from ragnosis.wikipedia import Article


class ArticleStore:
    """Saves and loads articles, keyed by the title they were requested under.

    Keying on the requested title (not the title after redirects) lets a rerun
    see that a title is already stored without asking Wikipedia first.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def path_for(self, title: str) -> Path:
        """File for ``title``. Percent-encoding makes any title a safe, unique file name."""
        return self.directory / f"{quote(title.replace(' ', '_'), safe='')}.json"

    def contains(self, title: str) -> bool:
        return self.path_for(title).is_file()

    def save(self, title: str, article: Article) -> Path:
        """Write ``article`` under ``title``, replacing any earlier version.

        Writes to a temporary file first and then swaps it in, so an interrupted
        run never leaves a half-written file that would later fail to load.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.path_for(title)
        temp = path.with_suffix(".tmp")
        temp.write_text(
            json.dumps(asdict(article), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp.replace(path)
        return path

    def load(self, title: str) -> Article:
        """Load the article stored under ``title``. Raises FileNotFoundError if absent."""
        data = json.loads(self.path_for(title).read_text(encoding="utf-8"))
        return Article(**data)

"""Read the list of Wikipedia articles to ingest.

The corpus is defined by a plain text file of titles rather than by Wikipedia
categories, so exactly what goes in is visible, reviewable in a diff and the
same on every run (see docs/decisions/0003).
"""

from __future__ import annotations

from pathlib import Path


def normalize_title(title: str) -> str:
    """Normalize a title the way Wikipedia does for whitespace and underscores.

    Capitalization is left alone: Wikipedia only treats the first letter as
    case-insensitive, and that rule varies between languages.
    """
    return " ".join(title.replace("_", " ").split())


def load_titles(path: Path) -> list[str]:
    """Return the titles in ``path``, one per line, in file order.

    ``#`` starts a comment (it cannot occur in a Wikipedia title), blank lines are
    skipped and duplicates are dropped, so the file can be grouped and annotated
    freely without causing repeat downloads.
    """
    titles: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        title = normalize_title(line.split("#", 1)[0])
        if title and title not in seen:
            seen.add(title)
            titles.append(title)
    return titles

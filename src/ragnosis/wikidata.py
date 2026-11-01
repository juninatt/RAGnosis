"""Fetch other names of Wikipedia articles from Wikidata.

Every Wikipedia article has a Wikidata item with its name in many languages
and its aliases, such as "Svartskägg" and "Edward Teach" for Blackbeard. The
names are only used to find passages; answers still come from Wikipedia text
(decision 0011).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from types import TracebackType
from typing import Any

import httpx

from ragnosis.api_client import ApiClient, ApiError
from ragnosis.settings import is_valid_language

API_URL = "https://www.wikidata.org/w/api.php"
# The most titles Wikidata accepts in one request.
BATCH_SIZE = 50
# Some items have dozens of aliases, including fragments such as "Din" for
# Hayreddin Barbarossa; a few good names help search more than many.
MAX_NAMES = 10
MIN_NAME_LENGTH = 4


class WikidataError(RuntimeError):
    """Wikidata could not deliver the names."""


def clean_names(title: str, candidates: Iterable[str]) -> list[str]:
    """Up to ``MAX_NAMES`` distinct names other than ``title``, in the given order."""
    seen = {title.casefold()}
    names: list[str] = []
    for candidate in candidates:
        name = " ".join(candidate.split())
        key = name.casefold()
        if len(name) < MIN_NAME_LENGTH or key in seen:
            continue
        seen.add(key)
        names.append(name)
        if len(names) == MAX_NAMES:
            break
    return names


class WikidataClient:
    """Fetches other names for articles of one Wikipedia language edition."""

    def __init__(
        self,
        wiki_language: str = "en",
        name_languages: Sequence[str] = ("en", "sv"),
        *,
        transport: httpx.BaseTransport | None = None,
        max_attempts: int = 4,
        backoff_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create a client.

        ``name_languages`` are the languages whose names and aliases are kept,
        in order of preference. ``transport`` and ``sleep`` exist so tests can
        run without network access or real waiting.
        """
        for language in (wiki_language, *name_languages):
            if not is_valid_language(language):
                raise ValueError(f"Invalid language code: {language!r}")
        if not name_languages:
            raise ValueError("At least one name language is needed")
        self.site = f"{wiki_language.replace('-', '_')}wiki"
        self.name_languages = tuple(name_languages)
        self._api = ApiClient(
            API_URL,
            transport=transport,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            sleep=sleep,
        )

    def __enter__(self) -> WikidataClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying HTTP connections."""
        self._api.close()

    def fetch_names(self, titles: Sequence[str]) -> dict[str, list[str]]:
        """Other names for each Wikipedia article title, asking for 50 at a time.

        Titles must be the articles' own titles, as Wikipedia returns them.
        A title without a Wikidata item gets an empty list.

        Raises:
            WikidataError: a request failed or returned something unusable.
        """
        names: dict[str, list[str]] = {}
        for start in range(0, len(titles), BATCH_SIZE):
            names.update(self._fetch_batch(titles[start : start + BATCH_SIZE]))
        return names

    def _fetch_batch(self, titles: Sequence[str]) -> dict[str, list[str]]:
        try:
            data = self._api.get(
                {
                    "action": "wbgetentities",
                    "format": "json",
                    "formatversion": "2",
                    "sites": self.site,
                    "titles": "|".join(titles),
                    "props": "labels|aliases|sitelinks",
                    "languages": "|".join(self.name_languages),
                    "sitefilter": self.site,
                }
            )
        except ApiError as exc:
            raise WikidataError(str(exc)) from exc
        if "error" in data:
            raise WikidataError(f"API error: {data['error']}")

        found: dict[str, list[str]] = {title: [] for title in titles}
        try:
            for entity in data["entities"].values():
                if "missing" in entity:
                    continue
                title = entity["sitelinks"][self.site]["title"]
                if title in found:
                    found[title] = clean_names(title, self._candidates(entity))
        except (KeyError, TypeError, AttributeError) as exc:
            raise WikidataError("Unexpected Wikidata response") from exc
        return found

    def _candidates(self, entity: dict[str, Any]) -> list[str]:
        """Labels first, then aliases, each in the order of ``name_languages``."""
        labels = entity.get("labels", {})
        aliases = entity.get("aliases", {})
        candidates = [labels[lang]["value"] for lang in self.name_languages if lang in labels]
        for lang in self.name_languages:
            candidates += [alias["value"] for alias in aliases.get(lang, [])]
        return candidates

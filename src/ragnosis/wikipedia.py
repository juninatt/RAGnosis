"""Fetch articles as plain text from the Wikipedia API.

Uses the TextExtracts API, which returns article text without markup, so no
wikitext or HTML parsing is needed. Section headings remain as ``== Heading ==``
lines, which later steps can use to split articles sensibly.

Requests go through ApiClient, which is polite to Wikimedia's servers.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from types import TracebackType
from typing import Any

import httpx

from ragnosis.api_client import ApiClient, ApiError
from ragnosis.settings import is_valid_language


class WikipediaError(RuntimeError):
    """The API could not deliver a usable article."""


class ArticleNotFoundError(WikipediaError):
    """No article exists under the requested title."""


class DisambiguationPageError(ArticleNotFoundError):
    """The title leads to a list of meanings, not to an article worth indexing."""


@dataclass(frozen=True)
class Article:
    """A fetched article and where it came from, so answers can cite their source."""

    title: str
    page_id: int
    revision_id: int
    url: str
    language: str
    text: str
    fetched_at: str


def _utc_now() -> datetime:
    return datetime.now(UTC)


class WikipediaClient:
    """Fetches single articles from one language edition of Wikipedia."""

    def __init__(
        self,
        language: str = "en",
        *,
        transport: httpx.BaseTransport | None = None,
        max_attempts: int = 4,
        backoff_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """Create a client.

        ``transport``, ``sleep`` and ``clock`` exist so tests can run without
        network access, real waiting or a changing time.
        """
        if not is_valid_language(language):
            raise ValueError(f"Invalid Wikipedia language code: {language!r}")
        self.language = language
        self._api = ApiClient(
            f"https://{language}.wikipedia.org/w/api.php",
            transport=transport,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            sleep=sleep,
        )
        self._clock = clock

    def __enter__(self) -> WikipediaClient:
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

    def fetch_article(self, title: str) -> Article:
        """Fetch the article ``title``, following redirects.

        Raises:
            ArticleNotFoundError: the title does not exist or is invalid.
            DisambiguationPageError: the title is a disambiguation page.
            WikipediaError: the API failed or returned something unusable.
        """
        data = self._get(
            {
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "prop": "extracts|info|pageprops",
                "explaintext": "1",
                "inprop": "url",
                "ppprop": "disambiguation",
                "redirects": "1",
                "titles": title,
            }
        )
        if "error" in data:
            raise WikipediaError(f"API error for {title!r}: {data['error']}")

        try:
            page = data["query"]["pages"][0]
            if page.get("missing") or page.get("invalid"):
                raise ArticleNotFoundError(f"No Wikipedia article titled {title!r}")
            if "disambiguation" in page.get("pageprops", {}):
                raise DisambiguationPageError(f"{title!r} is a disambiguation page")
            text = page.get("extract") or ""
            if not text.strip():
                raise WikipediaError(f"Article {title!r} has no text")
            return Article(
                title=page["title"],
                page_id=page["pageid"],
                revision_id=page["lastrevid"],
                url=page["fullurl"],
                language=self.language,
                text=text,
                fetched_at=self._clock().isoformat(),
            )
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise WikipediaError(f"Unexpected API response for {title!r}") from exc

    def _get(self, params: dict[str, str]) -> dict[str, Any]:
        try:
            return self._api.get(params)
        except ApiError as exc:
            raise WikipediaError(str(exc)) from exc

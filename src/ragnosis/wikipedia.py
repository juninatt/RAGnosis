"""Fetch articles as plain text from the Wikipedia API.

Uses the TextExtracts API, which returns article text without markup, so no
wikitext or HTML parsing is needed. Section headings remain as ``== Heading ==``
lines, which later steps can use to split articles sensibly.

The client is deliberately polite: it identifies itself with a User-Agent (which
Wikimedia requires), sends one request at a time and backs off when the servers
ask it to.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from types import TracebackType
from typing import Any

import httpx

from ragnosis.settings import is_valid_language

logger = logging.getLogger(__name__)

# The version comes from the installed package so it follows every release.
USER_AGENT = f"RAGnosis/{version('ragnosis')} (https://github.com/juninatt/RAGnosis)"
TIMEOUT_SECONDS = 30.0
# Rate limiting and temporary server trouble; anything else will not improve by retrying.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


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
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.language = language
        self._api_url = f"https://{language}.wikipedia.org/w/api.php"
        self._http = httpx.Client(
            transport=transport,
            timeout=TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
        )
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds
        self._sleep = sleep
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
        self._http.close()

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
        """GET the API, retrying with exponential backoff on temporary failures."""
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            delay = self._backoff_seconds * 2 ** (attempt - 1)
            try:
                response = self._http.get(self._api_url, params=params)
            except httpx.TransportError as exc:
                last_error = exc
            else:
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    return self._parse(response)
                last_error = WikipediaError(f"HTTP {response.status_code}")
                delay = max(delay, _retry_after_seconds(response))

            if attempt < self._max_attempts:
                logger.warning(
                    "Request failed (%s), retry %d/%d in %.1fs",
                    last_error,
                    attempt,
                    self._max_attempts - 1,
                    delay,
                )
                self._sleep(delay)

        raise WikipediaError(
            f"Giving up after {self._max_attempts} attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _parse(response: httpx.Response) -> dict[str, Any]:
        if response.is_error:
            raise WikipediaError(f"HTTP {response.status_code} from {response.url}")
        try:
            data = response.json()
        except ValueError as exc:
            raise WikipediaError("Response was not valid JSON") from exc
        if not isinstance(data, dict):
            raise WikipediaError("Response was not a JSON object")
        return data


def _retry_after_seconds(response: httpx.Response) -> float:
    """Seconds the server asked us to wait, or 0 if it did not say (in a form we read)."""
    try:
        return max(0.0, float(response.headers.get("Retry-After", "0")))
    except ValueError:
        return 0.0

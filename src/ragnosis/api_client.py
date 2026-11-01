"""Polite GET requests to Wikimedia's web APIs.

Wikimedia asks clients to identify themselves with a User-Agent, to send one
request at a time and to back off when the servers ask them to. Wikipedia and
Wikidata are both reached this way, so the rules live here once.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from importlib.metadata import version
from types import TracebackType
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# The version comes from the installed package so it follows every release.
USER_AGENT = f"RAGnosis/{version('ragnosis')} (https://github.com/juninatt/RAGnosis)"
TIMEOUT_SECONDS = 30.0
# Rate limiting and temporary server trouble; anything else will not improve by retrying.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class ApiError(RuntimeError):
    """The API could not deliver a usable response."""


class ApiClient:
    """GETs JSON objects from one API address, retrying temporary failures."""

    def __init__(
        self,
        url: str,
        *,
        transport: httpx.BaseTransport | None = None,
        max_attempts: int = 4,
        backoff_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create a client for ``url``.

        ``transport`` and ``sleep`` exist so tests can run without network
        access or real waiting.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.url = url
        self._http = httpx.Client(
            transport=transport,
            timeout=TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
        )
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds
        self._sleep = sleep

    def __enter__(self) -> ApiClient:
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

    def get(self, params: dict[str, str]) -> dict[str, Any]:
        """GET the API, retrying with exponential backoff on temporary failures.

        Raises:
            ApiError: the request failed for good, or the answer was not a JSON object.
        """
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            delay = self._backoff_seconds * 2 ** (attempt - 1)
            try:
                response = self._http.get(self.url, params=params)
            except httpx.TransportError as exc:
                last_error = exc
            else:
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    return self._parse(response)
                last_error = ApiError(f"HTTP {response.status_code}")
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

        raise ApiError(
            f"Giving up after {self._max_attempts} attempts: {last_error}"
        ) from last_error

    @staticmethod
    def _parse(response: httpx.Response) -> dict[str, Any]:
        if response.is_error:
            raise ApiError(f"HTTP {response.status_code} from {response.url}")
        try:
            data = response.json()
        except ValueError as exc:
            raise ApiError("Response was not valid JSON") from exc
        if not isinstance(data, dict):
            raise ApiError("Response was not a JSON object")
        return data


def _retry_after_seconds(response: httpx.Response) -> float:
    """Seconds the server asked us to wait, or 0 if it did not say (in a form we read)."""
    try:
        return max(0.0, float(response.headers.get("Retry-After", "0")))
    except ValueError:
        return 0.0

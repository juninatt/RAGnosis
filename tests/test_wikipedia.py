"""Tests for the Wikipedia client, using a mock transport instead of the network."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from factories import api_page, api_response

from ragnosis.api_client import USER_AGENT
from ragnosis.wikipedia import (
    ArticleNotFoundError,
    DisambiguationPageError,
    WikipediaClient,
    WikipediaError,
)

FIXED_TIME = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
Handler = Callable[[httpx.Request], httpx.Response]


def make_client(
    handler: Handler,
    *,
    language: str = "en",
    max_attempts: int = 3,
    sleeps: list[float] | None = None,
) -> WikipediaClient:
    return WikipediaClient(
        language,
        transport=httpx.MockTransport(handler),
        max_attempts=max_attempts,
        backoff_seconds=1.0,
        sleep=(sleeps if sleeps is not None else []).append,
        clock=lambda: FIXED_TIME,
    )


def returning(payload: Any, status: int = 200) -> Handler:
    return lambda request: httpx.Response(status, json=payload)


def in_sequence(*outcomes: httpx.Response | Exception) -> Handler:
    """Answer each request with the next outcome; exceptions are raised like network errors."""
    queue = list(outcomes)

    def handler(request: httpx.Request) -> httpx.Response:
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return handler


OK = httpx.Response(200, json=api_response(api_page()))


# --- Successful fetches ---


def test_maps_api_page_to_article() -> None:
    with make_client(returning(api_response(api_page()))) as client:
        article = client.fetch_article("Blackbeard")

    assert article.title == "Blackbeard"
    assert article.page_id == 1
    assert article.revision_id == 100
    assert article.url == "https://en.wikipedia.org/wiki/Blackbeard"
    assert article.language == "en"
    assert article.text.startswith("Edward Teach")
    assert article.fetched_at == "2026-10-21T12:00:00+00:00"


def test_sends_expected_request() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=api_response(api_page()))

    with make_client(handler, language="sv") as client:
        client.fetch_article("Anne Bonny")

    (request,) = requests
    assert request.url.host == "sv.wikipedia.org"
    assert request.url.path == "/w/api.php"
    assert request.headers["User-Agent"] == USER_AGENT
    params = request.url.params
    assert params["titles"] == "Anne Bonny"
    assert params["redirects"] == "1"
    assert params["explaintext"] == "1"
    assert params["formatversion"] == "2"


def test_redirect_returns_the_target_title() -> None:
    payload = api_response(api_page(title="Blackbeard"))
    payload["query"]["redirects"] = [{"from": "Edward Teach", "to": "Blackbeard"}]
    with make_client(returning(payload)) as client:
        assert client.fetch_article("Edward Teach").title == "Blackbeard"


def test_keeps_non_ascii_text() -> None:
    page = api_page(title="Klaus Störtebeker", extract="Klaus Störtebeker – Vitalienbrüder")
    with make_client(returning(api_response(page))) as client:
        article = client.fetch_article("Klaus Störtebeker")
    assert article.text == "Klaus Störtebeker – Vitalienbrüder"


# --- Titles that are not usable articles ---


@pytest.mark.parametrize(
    "page",
    [
        {"ns": 0, "title": "No such pirate", "missing": True},
        {"title": "Bad|title", "invalid": True, "invalidreason": "illegal character"},
    ],
    ids=["missing", "invalid"],
)
def test_missing_or_invalid_title_raises_not_found(page: dict[str, Any]) -> None:
    with make_client(returning(api_response(page))) as client, pytest.raises(ArticleNotFoundError):
        client.fetch_article("No such pirate")


def test_disambiguation_page_raises() -> None:
    page = api_page(title="Mercury", pageprops={"disambiguation": ""})
    client = make_client(returning(api_response(page)))
    with client, pytest.raises(DisambiguationPageError, match="disambiguation"):
        client.fetch_article("Mercury")


def test_disambiguation_is_a_kind_of_not_found() -> None:
    assert issubclass(DisambiguationPageError, ArticleNotFoundError)


@pytest.mark.parametrize("extract", ["", "   \n", None])
def test_empty_text_raises(extract: str | None) -> None:
    page = api_page(extract=extract)
    client = make_client(returning(api_response(page)))
    with client, pytest.raises(WikipediaError, match="no text"):
        client.fetch_article("Blackbeard")


# --- Malformed or error responses ---


def test_api_error_object_raises() -> None:
    payload = {"error": {"code": "badvalue", "info": "Unrecognized value"}}
    client = make_client(returning(payload))
    with client, pytest.raises(WikipediaError, match="API error"):
        client.fetch_article("Blackbeard")


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": {}},
        {"query": {"pages": []}},
        {"query": {"pages": ["not a page"]}},
        api_response({"title": "Blackbeard", "extract": "text"}),
    ],
    ids=["empty", "no-pages", "empty-pages", "page-not-object", "missing-fields"],
)
def test_unexpected_structure_raises(payload: dict[str, Any]) -> None:
    client = make_client(returning(payload))
    with client, pytest.raises(WikipediaError, match="Unexpected API response"):
        client.fetch_article("Blackbeard")


def test_failed_requests_raise_wikipedia_errors() -> None:
    # Retries and parsing are tested with ApiClient; here only the error type matters.
    sleeps: list[float] = []
    handler = in_sequence(*(httpx.Response(503) for _ in range(3)))
    client = make_client(handler, max_attempts=3, sleeps=sleeps)
    with client, pytest.raises(WikipediaError, match="Giving up after 3 attempts"):
        client.fetch_article("Blackbeard")
    assert sleeps == [1.0, 2.0]


def test_responses_that_are_not_json_objects_raise_wikipedia_errors() -> None:
    client = make_client(returning([1, 2, 3]))
    with client, pytest.raises(WikipediaError, match="not a JSON object"):
        client.fetch_article("Blackbeard")


# --- Construction ---


@pytest.mark.parametrize("language", ["", "EN", "en.evil.com", "en/../x", "e"])
def test_rejects_invalid_language(language: str) -> None:
    with pytest.raises(ValueError, match="language code"):
        WikipediaClient(language)


def test_rejects_zero_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        WikipediaClient(max_attempts=0)


def test_default_clock_stamps_current_utc_time() -> None:
    client = WikipediaClient(transport=httpx.MockTransport(returning(api_response(api_page()))))
    before = datetime.now(UTC)
    with client:
        fetched_at = datetime.fromisoformat(client.fetch_article("Blackbeard").fetched_at)
    assert fetched_at.tzinfo is not None
    assert before <= fetched_at <= datetime.now(UTC)


def test_close_releases_http_client() -> None:
    client = make_client(returning(api_response(api_page())))
    with client:
        pass
    with pytest.raises(RuntimeError):
        client.fetch_article("Blackbeard")

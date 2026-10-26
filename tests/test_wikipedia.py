"""Tests for the Wikipedia client, using a mock transport instead of the network."""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from factories import api_page, api_response

from ragnosis.wikipedia import (
    USER_AGENT,
    ArticleNotFoundError,
    DisambiguationPageError,
    WikipediaClient,
    WikipediaError,
)

FIXED_TIME = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)
REPO_ROOT = Path(__file__).resolve().parents[1]
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


def test_user_agent_carries_the_released_version_and_contact() -> None:
    # Wikimedia asks clients to identify themselves and how to reach the operator.
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert USER_AGENT.startswith(f"RAGnosis/{pyproject['project']['version']} ")
    assert "https://github.com/juninatt/RAGnosis" in USER_AGENT


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


def test_non_json_body_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>maintenance</html>")

    with make_client(handler) as client, pytest.raises(WikipediaError, match="not valid JSON"):
        client.fetch_article("Blackbeard")


def test_json_that_is_not_an_object_raises() -> None:
    client = make_client(returning([1, 2, 3]))
    with client, pytest.raises(WikipediaError, match="not a JSON object"):
        client.fetch_article("Blackbeard")


# --- Retries ---


def test_retries_temporary_failures_with_exponential_backoff() -> None:
    sleeps: list[float] = []
    handler = in_sequence(httpx.Response(503), httpx.Response(502), OK)
    with make_client(handler, max_attempts=3, sleeps=sleeps) as client:
        assert client.fetch_article("Blackbeard").title == "Blackbeard"
    assert sleeps == [1.0, 2.0]


def test_retries_network_errors() -> None:
    sleeps: list[float] = []
    handler = in_sequence(httpx.ConnectError("connection refused"), OK)
    with make_client(handler, sleeps=sleeps) as client:
        assert client.fetch_article("Blackbeard").page_id == 1
    assert sleeps == [1.0]


def test_honours_longer_retry_after_header() -> None:
    sleeps: list[float] = []
    handler = in_sequence(httpx.Response(429, headers={"Retry-After": "7"}), OK)
    with make_client(handler, sleeps=sleeps) as client:
        client.fetch_article("Blackbeard")
    assert sleeps == [7.0]


@pytest.mark.parametrize("value", ["Wed, 21 Oct 2026 07:28:00 GMT", "-5"])
def test_unusable_retry_after_falls_back_to_backoff(value: str) -> None:
    sleeps: list[float] = []
    handler = in_sequence(httpx.Response(429, headers={"Retry-After": value}), OK)
    with make_client(handler, sleeps=sleeps) as client:
        client.fetch_article("Blackbeard")
    assert sleeps == [1.0]


def test_gives_up_after_max_attempts() -> None:
    sleeps: list[float] = []
    handler = in_sequence(*(httpx.Response(503) for _ in range(3)))
    client = make_client(handler, max_attempts=3, sleeps=sleeps)
    with client, pytest.raises(WikipediaError, match="Giving up after 3 attempts"):
        client.fetch_article("Blackbeard")
    assert sleeps == [1.0, 2.0]


def test_logs_each_retry(caplog: pytest.LogCaptureFixture) -> None:
    handler = in_sequence(httpx.Response(503), OK)
    with make_client(handler) as client:
        client.fetch_article("Blackbeard")
    assert "retry 1/2" in caplog.text


@pytest.mark.parametrize("status", [400, 403, 404])
def test_client_errors_are_not_retried(status: int) -> None:
    sleeps: list[float] = []
    client = make_client(returning({}, status=status), sleeps=sleeps)
    with client, pytest.raises(WikipediaError, match=f"HTTP {status}"):
        client.fetch_article("Blackbeard")
    assert sleeps == []


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

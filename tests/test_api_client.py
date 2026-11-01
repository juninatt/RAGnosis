"""Tests for polite API requests, using a mock transport instead of the network."""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from ragnosis.api_client import USER_AGENT, ApiClient, ApiError

URL = "https://en.wikipedia.org/w/api.php"
REPO_ROOT = Path(__file__).resolve().parents[1]
Handler = Callable[[httpx.Request], httpx.Response]


def make_client(
    handler: Handler, *, max_attempts: int = 3, sleeps: list[float] | None = None
) -> ApiClient:
    return ApiClient(
        URL,
        transport=httpx.MockTransport(handler),
        max_attempts=max_attempts,
        backoff_seconds=1.0,
        sleep=(sleeps if sleeps is not None else []).append,
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


OK = httpx.Response(200, json={"ok": True})


def test_sends_the_parameters_and_user_agent() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True})

    with make_client(handler) as client:
        assert client.get({"action": "query", "titles": "Anne Bonny"}) == {"ok": True}

    (request,) = requests
    assert str(request.url).startswith(URL)
    assert request.url.params["titles"] == "Anne Bonny"
    assert request.headers["User-Agent"] == USER_AGENT


def test_user_agent_carries_the_released_version_and_contact() -> None:
    # Wikimedia asks clients to identify themselves and how to reach the operator.
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert USER_AGENT.startswith(f"RAGnosis/{pyproject['project']['version']} ")
    assert "https://github.com/juninatt/RAGnosis" in USER_AGENT


# --- Malformed responses ---


def test_non_json_body_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>maintenance</html>")

    with make_client(handler) as client, pytest.raises(ApiError, match="not valid JSON"):
        client.get({})


def test_json_that_is_not_an_object_raises() -> None:
    client = make_client(returning([1, 2, 3]))
    with client, pytest.raises(ApiError, match="not a JSON object"):
        client.get({})


# --- Retries ---


def test_retries_temporary_failures_with_exponential_backoff() -> None:
    sleeps: list[float] = []
    handler = in_sequence(httpx.Response(503), httpx.Response(502), OK)
    with make_client(handler, max_attempts=3, sleeps=sleeps) as client:
        assert client.get({}) == {"ok": True}
    assert sleeps == [1.0, 2.0]


def test_retries_network_errors() -> None:
    sleeps: list[float] = []
    handler = in_sequence(httpx.ConnectError("connection refused"), OK)
    with make_client(handler, sleeps=sleeps) as client:
        assert client.get({}) == {"ok": True}
    assert sleeps == [1.0]


def test_honours_longer_retry_after_header() -> None:
    sleeps: list[float] = []
    handler = in_sequence(httpx.Response(429, headers={"Retry-After": "7"}), OK)
    with make_client(handler, sleeps=sleeps) as client:
        client.get({})
    assert sleeps == [7.0]


@pytest.mark.parametrize("value", ["Wed, 21 Oct 2026 07:28:00 GMT", "-5"])
def test_unusable_retry_after_falls_back_to_backoff(value: str) -> None:
    sleeps: list[float] = []
    handler = in_sequence(httpx.Response(429, headers={"Retry-After": value}), OK)
    with make_client(handler, sleeps=sleeps) as client:
        client.get({})
    assert sleeps == [1.0]


def test_gives_up_after_max_attempts() -> None:
    sleeps: list[float] = []
    handler = in_sequence(*(httpx.Response(503) for _ in range(3)))
    client = make_client(handler, max_attempts=3, sleeps=sleeps)
    with client, pytest.raises(ApiError, match="Giving up after 3 attempts"):
        client.get({})
    assert sleeps == [1.0, 2.0]


def test_logs_each_retry(caplog: pytest.LogCaptureFixture) -> None:
    handler = in_sequence(httpx.Response(503), OK)
    with make_client(handler) as client:
        client.get({})
    assert "retry 1/2" in caplog.text


@pytest.mark.parametrize("status", [400, 403, 404])
def test_client_errors_are_not_retried(status: int) -> None:
    sleeps: list[float] = []
    client = make_client(returning({}, status=status), sleeps=sleeps)
    with client, pytest.raises(ApiError, match=f"HTTP {status}"):
        client.get({})
    assert sleeps == []


# --- Construction ---


def test_rejects_zero_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        ApiClient(URL, max_attempts=0)


def test_close_releases_http_client() -> None:
    client = make_client(returning({}))
    with client:
        pass
    with pytest.raises(RuntimeError):
        client.get({})

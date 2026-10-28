"""Tests for the Ollama client, using a mock transport instead of a running Ollama."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from ragnosis.llm import (
    SEED,
    LLMError,
    LLMUnavailableError,
    Message,
    ModelNotFoundError,
    OllamaClient,
)

Handler = Callable[[httpx.Request], httpx.Response]
MESSAGES = [Message("system", "Answer briefly."), Message("user", "Who was Blackbeard?")]


def ndjson(*lines: dict[str, Any]) -> bytes:
    return "".join(json.dumps(line) + "\n" for line in lines).encode()


def piece(text: str) -> dict[str, Any]:
    return {"model": "qwen3.5:4b", "message": {"role": "assistant", "content": text}, "done": False}


DONE = {"model": "qwen3.5:4b", "message": {"role": "assistant", "content": ""}, "done": True}


def streaming(*lines: dict[str, Any], status: int = 200) -> Handler:
    return lambda request: httpx.Response(status, content=ndjson(*lines))


def make_client(handler: Handler, base_url: str = "http://localhost:11434") -> OllamaClient:
    return OllamaClient("qwen3.5:4b", base_url, transport=httpx.MockTransport(handler))


# --- Successful replies ---


def test_streams_reply_pieces_in_order() -> None:
    client = make_client(streaming(piece("Edward "), piece("Teach"), piece("."), DONE))
    with client:
        assert list(client.stream_chat(MESSAGES)) == ["Edward ", "Teach", "."]


def test_chat_joins_the_pieces() -> None:
    with make_client(streaming(piece("Edward "), piece("Teach."), DONE)) as client:
        assert client.chat(MESSAGES) == "Edward Teach."


def test_ignores_blank_lines_and_empty_pieces() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\n" + ndjson(piece(""), piece("Hi"), DONE) + b"\n")

    with make_client(handler) as client:
        assert list(client.stream_chat(MESSAGES)) == ["Hi"]


def test_stops_at_done_even_if_more_follows() -> None:
    with make_client(streaming(piece("Hi"), DONE, piece("ignored"))) as client:
        assert client.chat(MESSAGES) == "Hi"


def test_text_on_the_final_line_is_kept() -> None:
    final = {**DONE, "message": {"role": "assistant", "content": "!"}}
    with make_client(streaming(piece("Hi"), final)) as client:
        assert client.chat(MESSAGES) == "Hi!"


def test_sends_model_messages_and_disables_thinking() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=ndjson(DONE))

    with make_client(handler, base_url="http://gpu-box:11434/") as client:
        client.chat(MESSAGES, temperature=0.5)

    (request,) = requests
    assert str(request.url) == "http://gpu-box:11434/api/chat"
    body = json.loads(request.content)
    assert body["model"] == "qwen3.5:4b"
    assert body["stream"] is True
    assert body["think"] is False
    assert body["keep_alive"] == "30m"
    assert body["options"] == {"temperature": 0.5, "seed": SEED}
    assert body["messages"] == [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "Who was Blackbeard?"},
    ]


def test_every_request_uses_the_same_seed() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=ndjson(DONE))

    with make_client(handler) as client:
        client.chat(MESSAGES)
        client.chat(MESSAGES, temperature=0.0)
    assert [body["options"]["seed"] for body in bodies] == [SEED, SEED]


def test_sends_the_configured_keep_alive() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, content=ndjson(DONE))

    client = OllamaClient("qwen3.5:4b", keep_alive="-1", transport=httpx.MockTransport(handler))
    with client:
        client.chat(MESSAGES)
    assert bodies[0]["keep_alive"] == "-1"


# --- Failures, each with an error that says what to do ---


def test_missing_model_tells_how_to_download_it() -> None:
    handler = streaming({"error": "model 'qwen3.5:4b' not found"}, status=404)
    with make_client(handler) as client, pytest.raises(ModelNotFoundError, match="ollama pull"):
        client.chat(MESSAGES)


@pytest.mark.parametrize(
    "error",
    [httpx.ConnectError("refused"), httpx.ConnectTimeout("timed out")],
    ids=["refused", "timeout"],
)
def test_unreachable_server_tells_to_start_ollama(error: Exception) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    with make_client(handler) as client, pytest.raises(LLMUnavailableError, match="start Ollama"):
        client.chat(MESSAGES)


def test_unavailable_and_missing_model_are_llm_errors() -> None:
    assert issubclass(LLMUnavailableError, LLMError)
    assert issubclass(ModelNotFoundError, LLMError)


def test_connection_lost_mid_reply() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("no data")

    with make_client(handler) as client, pytest.raises(LLMError, match="Lost connection"):
        client.chat(MESSAGES)


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (500, b'{"error": "out of memory"}', "HTTP 500: out of memory"),
        (400, b"plain text failure", "HTTP 400: plain text failure"),
        (404, b'{"error": "page missing"}', "HTTP 404: page missing"),
    ],
)
def test_other_http_errors_name_status_and_reason(status: int, body: bytes, expected: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=body)

    with make_client(handler) as client, pytest.raises(LLMError, match=expected):
        client.chat(MESSAGES)


def test_error_reported_mid_stream() -> None:
    handler = streaming(piece("Edward"), {"error": "model crashed"})
    with make_client(handler) as client, pytest.raises(LLMError, match="model crashed"):
        client.chat(MESSAGES)


@pytest.mark.parametrize(
    "line",
    [b"not json\n", b"[1, 2]\n", b'{"done": false}\n', b'{"message": "text"}\n'],
    ids=["not-json", "not-object", "no-message", "message-not-object"],
)
def test_malformed_lines_raise(line: bytes) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=line)

    with make_client(handler) as client, pytest.raises(LLMError, match="Unexpected reply"):
        client.chat(MESSAGES)


def test_reply_cut_off_before_done_raises() -> None:
    client = make_client(streaming(piece("Edward")))
    with client, pytest.raises(LLMError, match="without finishing"):
        client.chat(MESSAGES)


# --- Loading the model ahead of the first question ---


def test_load_asks_ollama_to_load_the_model_without_writing() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, json={"model": "qwen3.5:4b", "done": True, "done_reason": "load"}
        )

    client = OllamaClient("qwen3.5:4b", keep_alive="1h", transport=httpx.MockTransport(handler))
    with client:
        client.load()

    (request,) = requests
    assert request.url.path == "/api/chat"
    assert json.loads(request.content) == {
        "model": "qwen3.5:4b",
        "messages": [],
        "stream": False,
        "keep_alive": "1h",
    }


def test_load_reports_a_missing_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'qwen3.5:4b' not found"})

    with make_client(handler) as client, pytest.raises(ModelNotFoundError, match="ollama pull"):
        client.load()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (httpx.ConnectError("refused"), LLMUnavailableError),
        (httpx.ConnectTimeout("timed out"), LLMUnavailableError),
        (httpx.ReadTimeout("slow"), LLMError),
    ],
    ids=["refused", "connect-timeout", "read-timeout"],
)
def test_load_translates_connection_errors(error: Exception, expected: type[LLMError]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error

    with make_client(handler) as client, pytest.raises(expected):
        client.load()


def test_load_reports_other_http_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "out of memory"})

    with make_client(handler) as client, pytest.raises(LLMError, match="HTTP 500: out of memory"):
        client.load()


def test_close_releases_http_client() -> None:
    client = make_client(streaming(DONE))
    with client:
        pass
    with pytest.raises(RuntimeError):
        client.chat(MESSAGES)

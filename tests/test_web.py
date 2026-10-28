"""Tests for the web UI's HTTP layer, with a fake answer function instead of models."""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Callable
from typing import Any

import pytest
from factories import make_chunk
from starlette.applications import Starlette
from starlette.testclient import TestClient

from ragnosis.answer import DEFAULT_K, Answer, AnswerLength
from ragnosis.llm import LLMError, LLMUnavailableError, ModelNotFoundError
from ragnosis.vector_store import SearchResult
from ragnosis.web import MAX_BODY_BYTES, MAX_K, MAX_QUESTION_CHARS, AnswerFn, create_app

PASSAGES = [
    SearchResult(make_chunk(article_title="Blackbeard", section="Death"), 0.9),
    SearchResult(
        make_chunk(
            article_title="Anne Bonny",
            url="https://en.wikipedia.org/wiki/Anne_Bonny",
        ),
        0.8,
    ),
]


class Recorder:
    """An answer function that streams fixed pieces and records how it was called."""

    def __init__(self, *pieces: str, error: Exception | None = None) -> None:
        self.pieces = pieces or ("Bonny [2] ", "and Teach [1].")
        self.error = error
        self.calls: list[tuple[str, int, AnswerLength]] = []

    def __call__(
        self, question: str, k: int, length: AnswerLength, on_text: Callable[[str], None]
    ) -> Answer:
        self.calls.append((question, k, length))
        for piece in self.pieces:
            on_text(piece)
        if self.error:
            raise self.error
        return Answer(question, "".join(self.pieces), PASSAGES)


def ticking(*times: float) -> Callable[[], float]:
    """A clock that returns ``times`` in order, then keeps returning the last one."""
    remaining = list(times)
    return lambda: remaining.pop(0) if len(remaining) > 1 else remaining[0]


def client_for(answer: AnswerFn, clock: Callable[[], float] | None = None) -> TestClient:
    app = create_app(answer, clock=clock or ticking(0.0))
    return TestClient(app, base_url="http://localhost")


def ask(client: TestClient, body: Any = None, **kwargs: Any) -> Any:
    return client.post("/api/ask", json={"question": "Who?"} if body is None else body, **kwargs)


def events(response: Any) -> list[dict[str, Any]]:
    return [json.loads(line) for line in response.text.splitlines() if line]


# --- Health and page ---


def test_health() -> None:
    response = client_for(Recorder()).get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_page_is_served() -> None:
    response = client_for(Recorder()).get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<form id="ask-form">' in response.text


# --- Asking ---


def test_streams_text_pieces_then_cited_sources() -> None:
    response = ask(client_for(Recorder(), clock=ticking(100.0, 112.34, 141.06)))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert events(response) == [
        {"type": "text", "text": "Bonny [2] "},
        {"type": "text", "text": "and Teach [1]."},
        {
            "type": "sources",
            "sources": [
                {
                    "number": 2,
                    "title": "Anne Bonny",
                    "section": "Introduction",
                    "url": "https://en.wikipedia.org/wiki/Anne_Bonny",
                },
                {
                    "number": 1,
                    "title": "Blackbeard",
                    "section": "Death",
                    "url": "https://en.wikipedia.org/wiki/Blackbeard",
                },
            ],
            "timing": {"first_text_seconds": 12.3, "total_seconds": 41.1},
        },
    ]


def test_answer_without_citations_has_empty_sources() -> None:
    response = ask(client_for(Recorder("I cannot answer from the sources.")))
    assert events(response)[-1]["sources"] == []


def test_timing_counts_from_the_start_of_answering() -> None:
    # Clock readings: answer starts, first text arrives, answer ends.
    response = ask(client_for(Recorder("One piece."), clock=ticking(5.0, 7.0, 9.5)))
    assert events(response)[-1]["timing"] == {"first_text_seconds": 2.0, "total_seconds": 4.5}


def test_only_the_first_piece_sets_the_time_to_first_text() -> None:
    clock = ticking(0.0, 3.0, 4.0, 5.0, 6.0)
    response = ask(client_for(Recorder("a", "b", "c"), clock=clock))
    assert events(response)[-1]["timing"] == {"first_text_seconds": 3.0, "total_seconds": 4.0}


def test_answer_without_any_text_has_no_time_to_first_text() -> None:
    silent = Recorder()
    silent.pieces = ()
    response = ask(client_for(silent, clock=ticking(0.0, 8.0)))
    assert events(response)[-1]["timing"] == {"first_text_seconds": None, "total_seconds": 8.0}


def test_errors_carry_no_timing() -> None:
    response = ask(client_for(Recorder("Partial", error=LLMError("failed"))))
    assert "timing" not in events(response)[-1]


def test_default_clock_measures_real_time() -> None:
    client = TestClient(create_app(Recorder()), base_url="http://localhost")
    timing = events(ask(client))[-1]["timing"]
    assert 0 <= timing["first_text_seconds"] <= timing["total_seconds"] < 5


def test_passes_trimmed_question_and_default_k() -> None:
    recorder = Recorder()
    ask(client_for(recorder), {"question": "  Vem var Anne Bonny?  "})
    assert recorder.calls == [("Vem var Anne Bonny?", DEFAULT_K, "short")]


def test_passes_k() -> None:
    recorder = Recorder()
    ask(client_for(recorder), {"question": "Who?", "k": 3})
    assert recorder.calls == [("Who?", 3, "short")]


@pytest.mark.parametrize("length", ["short", "detailed"])
def test_passes_the_answer_length(length: str) -> None:
    recorder = Recorder()
    ask(client_for(recorder), {"question": "Who?", "length": length})
    assert recorder.calls == [("Who?", DEFAULT_K, length)]


def test_non_ascii_text_is_kept() -> None:
    response = ask(client_for(Recorder("Störtebeker ", "Ἀλέξανδρος")))
    assert "Störtebeker" in response.text
    assert [e["text"] for e in events(response)[:2]] == ["Störtebeker ", "Ἀλέξανδρος"]


# --- Errors while answering ---


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (LLMUnavailableError("Ollama is not reachable; start Ollama"), "setup"),
        (ModelNotFoundError("run: ollama pull qwen3.5:4b"), "setup"),
        (LLMError("Lost connection to Ollama"), "failure"),
    ],
    ids=["ollama-down", "model-missing", "model-failed"],
)
def test_model_errors_end_the_stream_with_an_error_event(error: Exception, kind: str) -> None:
    response = ask(client_for(Recorder("Partial", error=error)))

    assert response.status_code == 200
    assert events(response) == [
        {"type": "text", "text": "Partial"},
        {"type": "error", "kind": kind, "message": str(error)},
    ]


def test_unexpected_errors_are_logged_but_not_shown(caplog: pytest.LogCaptureFixture) -> None:
    response = ask(client_for(Recorder(error=RuntimeError("secret internals"))))

    last = events(response)[-1]
    assert last == {
        "type": "error",
        "kind": "failure",
        "message": "Something went wrong; see the server log",
    }
    assert "secret internals" not in response.text
    assert "secret internals" in caplog.text
    assert any(record.levelno == logging.ERROR for record in caplog.records)


# --- One question at a time ---


def test_second_question_is_refused_while_one_is_running() -> None:
    recorder = Recorder()
    app = create_app(recorder)
    busy: threading.Lock = app.state.busy
    client = TestClient(app, base_url="http://localhost")

    busy.acquire()
    try:
        response = ask(client)
    finally:
        busy.release()

    assert response.status_code == 429
    assert "try again" in response.json()["error"]
    assert recorder.calls == []


@pytest.mark.parametrize("error", [None, LLMError("failed")], ids=["success", "error"])
def test_lock_is_released_after_each_answer(error: Exception | None) -> None:
    app: Starlette = create_app(Recorder(error=error))
    client = TestClient(app, base_url="http://localhost")

    assert ask(client).status_code == 200
    assert ask(client).status_code == 200
    assert app.state.busy.acquire(blocking=False)


# --- Rejected requests ---


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ([1, 2], "JSON object"),
        ({}, "non-empty string"),
        ({"question": ""}, "non-empty string"),
        ({"question": "   "}, "non-empty string"),
        ({"question": 42}, "non-empty string"),
        ({"question": "x" * (MAX_QUESTION_CHARS + 1)}, "longer than"),
        ({"question": "Who?", "k": 0}, "k must be"),
        ({"question": "Who?", "k": MAX_K + 1}, "k must be"),
        ({"question": "Who?", "k": "3"}, "k must be"),
        ({"question": "Who?", "k": 2.5}, "k must be"),
        ({"question": "Who?", "k": True}, "k must be"),
        ({"question": "Who?", "length": "long"}, "length must be one of short, detailed"),
        ({"question": "Who?", "length": 2}, "length must be one of"),
        ({"question": "Who?", "length": None}, "length must be one of"),
    ],
)
def test_invalid_questions_are_rejected(body: Any, message: str) -> None:
    recorder = Recorder()
    response = ask(client_for(recorder), body)

    assert response.status_code == 400
    assert message in response.json()["error"]
    assert recorder.calls == []


def test_longest_allowed_question_is_accepted() -> None:
    response = ask(client_for(Recorder()), {"question": "x" * MAX_QUESTION_CHARS})
    assert response.status_code == 200


def test_body_that_is_not_json_is_rejected() -> None:
    response = client_for(Recorder()).post(
        "/api/ask", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400
    assert "not valid JSON" in response.json()["error"]


@pytest.mark.parametrize("content_type", ["text/plain", "application/x-www-form-urlencoded", ""])
def test_only_json_requests_are_accepted(content_type: str) -> None:
    # Other sites can send these types without a CORS check; JSON they cannot.
    response = client_for(Recorder()).post(
        "/api/ask", content=b'{"question": "Who?"}', headers={"Content-Type": content_type}
    )
    assert response.status_code == 415


def test_json_with_charset_is_accepted() -> None:
    response = client_for(Recorder()).post(
        "/api/ask",
        content=b'{"question": "Who?"}',
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    assert response.status_code == 200


def test_oversized_body_is_rejected() -> None:
    body = json.dumps({"question": "Who?", "padding": "x" * MAX_BODY_BYTES}).encode()
    response = client_for(Recorder()).post(
        "/api/ask", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413


def test_ask_requires_post() -> None:
    assert client_for(Recorder()).get("/api/ask").status_code == 405


@pytest.mark.parametrize("host", ["evil.example", "192.168.1.20", "localhost.evil.example"])
def test_requests_for_other_host_names_are_refused(host: str) -> None:
    # Guards against DNS rebinding: a site whose name resolves to 127.0.0.1.
    client = TestClient(create_app(Recorder()), base_url=f"http://{host}")
    assert client.get("/api/health").status_code == 400


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1"])
def test_local_host_names_are_accepted(host: str) -> None:
    client = TestClient(create_app(Recorder()), base_url=f"http://{host}:8000")
    assert client.get("/api/health").status_code == 200


# --- The page itself ---


def page() -> str:
    return str(client_for(Recorder()).get("/").text)


def test_page_loads_nothing_from_elsewhere() -> None:
    # Works offline and sends no question anywhere else.
    assert not re.search(r"""(src|href)\s*=\s*["']?\s*(https?:)?//""", page(), re.IGNORECASE)
    assert "@import" not in page()


def test_page_never_inserts_model_output_as_html() -> None:
    assert not re.search(r"\.(innerHTML|outerHTML)\s*=|insertAdjacentHTML|document\.write", page())


def test_page_talks_to_the_ask_api_with_json() -> None:
    text = page()
    assert 'fetch("/api/ask"' in text
    assert '"Content-Type": "application/json"' in text


def test_page_shows_the_answer_time_as_text() -> None:
    text = page()
    assert '<p id="timing" hidden></p>' in text
    assert "timingLine.textContent = " in text
    assert "Answered in" in text


def test_page_offers_short_and_detailed_answers_with_short_preselected() -> None:
    text = page()
    assert '<input type="radio" name="length" value="short" checked>' in text
    assert '<input type="radio" name="length" value="detailed">' in text
    assert "length: chosenLength()" in text


def test_page_remembers_the_length_only_locally_and_survives_blocked_storage() -> None:
    text = page()
    assert text.count("localStorage.") == 2
    # Every use of storage sits in a try block, so blocked storage cannot break the page.
    assert text.count("try {") >= 2
    assert "sessionStorage" not in text
    assert "document.cookie" not in text


def test_page_only_links_sources_to_wikipedia_over_https() -> None:
    assert r"/^https:\/\/[a-z-]+\.wikipedia\.org\//" in page()
    assert 'rel = "noopener noreferrer"' in page()

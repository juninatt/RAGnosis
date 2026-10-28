"""The web UI's HTTP layer: one page and a small JSON API (docs/decisions/0007).

``POST /api/ask`` streams its answer as JSON lines, one object per line:
``{"type": "text", "text": ...}`` while the model writes, then either
``{"type": "sources", "sources": [...], "timing": {...}}`` or
``{"type": "error", ...}``. The timing (seconds to the first words and in
total) lets the page show how long the answer took.

Answering is slow and CPU-bound, so only one question runs at a time; a second
request is turned away with HTTP 429 instead of slowing both down. The answer
runs in a worker thread that feeds a queue, so the lock is released when the
answer is finished even if the browser has stopped listening.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from importlib.resources import files
from typing import Any, NamedTuple

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from ragnosis.answer import ANSWER_LENGTHS, DEFAULT_K, Answer, AnswerLength
from ragnosis.llm import LLMError, LLMUnavailableError, ModelNotFoundError

logger = logging.getLogger(__name__)

# ask(question, k, length, on_text) -> Answer; the CLI binds the index and models to it.
AnswerFn = Callable[[str, int, AnswerLength, Callable[[str], None]], Answer]

ALLOWED_HOSTS = ["localhost", "127.0.0.1"]
MAX_QUESTION_CHARS = 1000
MAX_K = 20
MAX_BODY_BYTES = 16 * 1024


def create_app(
    answer_question: AnswerFn, *, clock: Callable[[], float] = time.monotonic
) -> Starlette:
    """Build the web app around ``answer_question``.

    ``clock`` times each answer; tests pass a fake one.
    """
    busy = threading.Lock()
    page = files("ragnosis").joinpath("static", "index.html").read_text(encoding="utf-8")

    async def index(request: Request) -> Response:
        return HTMLResponse(page)

    async def health(request: Request) -> Response:
        return JSONResponse({"status": "ok"})

    async def ask_endpoint(request: Request) -> Response:
        if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
            return _error(415, "Send the question as JSON (Content-Type: application/json)")
        try:
            body = await request.json()
        except ValueError:
            return _error(400, "The request body is not valid JSON")
        parsed = _parse_question(body)
        if isinstance(parsed, str):
            return _error(400, parsed)
        question = parsed

        if not busy.acquire(blocking=False):
            return _error(429, "Another question is being answered; try again when it is done")
        events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        worker = threading.Thread(
            target=_answer_in_background,
            args=(answer_question, question, events, busy, clock),
            daemon=True,
        )
        worker.start()
        return StreamingResponse(_json_lines(events), media_type="application/x-ndjson")

    app = Starlette(
        routes=[
            Route("/", index),
            Route("/api/health", health),
            Route("/api/ask", ask_endpoint, methods=["POST"]),
        ],
        middleware=[Middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)],
        max_body_size=MAX_BODY_BYTES,
    )
    # Exposed so tests can hold it and check that a second question is refused.
    app.state.busy = busy
    return app


class Question(NamedTuple):
    """A validated request: the question, how many passages and how long an answer."""

    text: str
    k: int
    length: AnswerLength


def _parse_question(body: Any) -> Question | str:
    """The question from a request body, or a message saying what is wrong."""
    if not isinstance(body, dict):
        return "The body must be a JSON object"
    question = body.get("question")
    if not isinstance(question, str) or not question.strip():
        return "Give a question as a non-empty string"
    if len(question) > MAX_QUESTION_CHARS:
        return f"The question is longer than {MAX_QUESTION_CHARS} characters"
    k = body.get("k", DEFAULT_K)
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= MAX_K:
        return f"k must be a whole number from 1 to {MAX_K}"
    length = body.get("length", "short")
    if length not in ANSWER_LENGTHS:
        return f"length must be one of {', '.join(ANSWER_LENGTHS)}"
    return Question(question.strip(), k, length)


def _answer_in_background(
    answer_question: AnswerFn,
    question: Question,
    events: queue.Queue[dict[str, Any] | None],
    busy: threading.Lock,
    clock: Callable[[], float],
) -> None:
    """Run one answer, turning text pieces, sources and errors into events."""
    start = clock()
    first_text_at: list[float] = []

    def on_text(text: str) -> None:
        if not first_text_at:
            first_text_at.append(clock() - start)
        events.put({"type": "text", "text": text})

    try:
        answer = answer_question(question.text, question.k, question.length, on_text)
        timing = {
            "first_text_seconds": round(first_text_at[0], 1) if first_text_at else None,
            "total_seconds": round(clock() - start, 1),
        }
        events.put({"type": "sources", "sources": _sources(answer), "timing": timing})
    except (LLMUnavailableError, ModelNotFoundError) as exc:
        logger.error("%s", exc)
        events.put({"type": "error", "kind": "setup", "message": str(exc)})
    except LLMError as exc:
        logger.error("%s", exc)
        events.put({"type": "error", "kind": "failure", "message": str(exc)})
    except Exception:
        logger.exception("Answering failed")
        events.put(
            {
                "type": "error",
                "kind": "failure",
                "message": "Something went wrong; see the server log",
            }
        )
    finally:
        events.put(None)
        busy.release()


def _json_lines(events: queue.Queue[dict[str, Any] | None]) -> Iterator[str]:
    while (event := events.get()) is not None:
        yield json.dumps(event, ensure_ascii=False) + "\n"


def _sources(answer: Answer) -> list[dict[str, Any]]:
    return [
        {
            "number": number,
            "title": result.chunk.article_title,
            "section": result.chunk.section,
            "url": result.chunk.url,
        }
        for number, result in answer.cited
    ]


def _error(status: int, message: str) -> Response:
    return JSONResponse({"error": message}, status_code=status)

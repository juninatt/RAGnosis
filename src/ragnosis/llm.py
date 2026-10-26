"""Talk to a language model served by a local Ollama.

Replies are streamed: on a CPU a full answer takes the better part of a minute,
and showing it as it is written is the difference between slow and broken.
Errors are translated into ones that say what to do, since "connection refused"
does not tell a user to start Ollama.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import httpx

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 600.0
# Ollama's own default is 5 minutes, after which the next question pays for
# loading the model again.
DEFAULT_KEEP_ALIVE = "30m"


class LLMError(RuntimeError):
    """The model could not produce a reply."""


class LLMUnavailableError(LLMError):
    """Nothing answers at the configured Ollama address."""


class ModelNotFoundError(LLMError):
    """Ollama runs, but the configured model has not been downloaded."""


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


class OllamaClient:
    """Chats with one model on one Ollama server."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        *,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        transport: httpx.BaseTransport | None = None,
        timeout: float = TIMEOUT_SECONDS,
    ) -> None:
        """Create a client.

        ``keep_alive`` is how long Ollama keeps the model in memory after a
        reply ("30m", "1h"; "-1" for always). Loading takes about 9 seconds on a
        laptop CPU, so a longer time spares that wait between questions.
        ``transport`` exists so tests can run without an Ollama server.
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive
        self._http = httpx.Client(transport=transport, timeout=timeout)

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying HTTP connections."""
        self._http.close()

    def stream_chat(
        self, messages: Sequence[Message], *, temperature: float = 0.2
    ) -> Iterator[str]:
        """Yield the reply piece by piece as the model writes it.

        Thinking is turned off: it multiplies answer time on a CPU and adds
        little when the answer is in the given text (docs/decisions/0006).

        Raises:
            LLMUnavailableError: Ollama is not running at ``base_url``.
            ModelNotFoundError: the model is not downloaded.
            LLMError: any other failure, including errors reported mid-stream.
        """
        payload = {
            "model": self.model,
            "stream": True,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": temperature},
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        try:
            with self._http.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                if response.is_error:
                    response.read()
                    raise self._http_error(response)
                for line in response.iter_lines():
                    if line.strip():
                        piece, done = self._parse_line(line)
                        if piece:
                            yield piece
                        if done:
                            return
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise self._unavailable() from exc
        except httpx.TransportError as exc:
            raise LLMError(f"Lost connection to Ollama: {exc}") from exc
        raise LLMError("Ollama ended the reply without finishing it")

    def chat(self, messages: Sequence[Message], *, temperature: float = 0.2) -> str:
        """The whole reply at once."""
        return "".join(self.stream_chat(messages, temperature=temperature))

    def load(self) -> None:
        """Load the model into memory without writing anything.

        Ollama loads a model when it gets a chat request with no messages.
        Doing this before the first question spares that question the load time.

        Raises:
            LLMUnavailableError, ModelNotFoundError, LLMError: as for ``stream_chat``.
        """
        payload = {
            "model": self.model,
            "messages": [],
            "stream": False,
            "keep_alive": self.keep_alive,
        }
        try:
            response = self._http.post(f"{self.base_url}/api/chat", json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise self._unavailable() from exc
        except httpx.TransportError as exc:
            raise LLMError(f"Lost connection to Ollama: {exc}") from exc
        if response.is_error:
            raise self._http_error(response)

    def _unavailable(self) -> LLMUnavailableError:
        return LLMUnavailableError(
            f"Ollama is not reachable at {self.base_url}; start Ollama and try again"
        )

    def _http_error(self, response: httpx.Response) -> LLMError:
        reason = _error_text(response)
        if response.status_code == 404 and "not found" in reason:
            return ModelNotFoundError(
                f"Model {self.model!r} is not downloaded; run: ollama pull {self.model}"
            )
        return LLMError(f"Ollama returned HTTP {response.status_code}: {reason}")

    @staticmethod
    def _parse_line(line: str) -> tuple[str, bool]:
        try:
            data: Any = json.loads(line)
            if "error" in data:
                raise LLMError(f"Ollama reported an error: {data['error']}")
            return str(data["message"]["content"]), bool(data.get("done"))
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError(f"Unexpected reply from Ollama: {line[:200]!r}") from exc


def _error_text(response: httpx.Response) -> str:
    try:
        return str(response.json()["error"])
    except (ValueError, KeyError, TypeError):
        return response.text[:200]

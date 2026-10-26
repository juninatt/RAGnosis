"""Checks against a real, running Ollama. Opt-in: ``uv run pytest -m integration``.

Needs Ollama running and the default model downloaded (``ollama pull qwen3.5:4b``).
"""

from __future__ import annotations

import pytest

from ragnosis.llm import Message, ModelNotFoundError, OllamaClient
from ragnosis.settings import DEFAULT_LLM_MODEL, DEFAULT_OLLAMA_URL

pytestmark = pytest.mark.integration


def test_real_model_streams_a_reply() -> None:
    messages = [
        Message("system", "Reply with exactly one word."),
        Message("user", "What colour was the classic pirate flag's background?"),
    ]
    with OllamaClient(DEFAULT_LLM_MODEL, DEFAULT_OLLAMA_URL) as client:
        pieces = list(client.stream_chat(messages, temperature=0))
    assert pieces
    assert "black" in "".join(pieces).lower()


def test_real_server_reports_missing_model() -> None:
    client = OllamaClient("ragnosis-no-such-model", DEFAULT_OLLAMA_URL)
    with client, pytest.raises(ModelNotFoundError):
        client.chat([Message("user", "hi")])

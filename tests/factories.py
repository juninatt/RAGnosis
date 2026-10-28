"""Builders and fakes for test data, so each test only spells out what it cares about."""

from __future__ import annotations

import re
import zlib
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from ragnosis.chunking import Chunk
from ragnosis.embeddings import Vectors
from ragnosis.llm import Message
from ragnosis.wikipedia import Article


def make_article(**overrides: Any) -> Article:
    fields: dict[str, Any] = {
        "title": "Blackbeard",
        "page_id": 1,
        "revision_id": 100,
        "url": "https://en.wikipedia.org/wiki/Blackbeard",
        "language": "en",
        "text": "Edward Teach, better known as Blackbeard, was an English pirate.",
        "fetched_at": "2026-10-21T12:00:00+00:00",
    }
    fields.update(overrides)
    return Article(**fields)


def make_chunk(**overrides: Any) -> Chunk:
    fields: dict[str, Any] = {
        "chunk_id": "1-0",
        "page_id": 1,
        "article_title": "Blackbeard",
        "section": "Introduction",
        "position": 0,
        "text": "Edward Teach, better known as Blackbeard, was an English pirate.",
        "url": "https://en.wikipedia.org/wiki/Blackbeard",
    }
    fields.update(overrides)
    return Chunk(**fields)


def api_page(**overrides: Any) -> dict[str, Any]:
    """One page as the API returns it with formatversion=2."""
    page: dict[str, Any] = {
        "pageid": 1,
        "ns": 0,
        "title": "Blackbeard",
        "extract": "Edward Teach, better known as Blackbeard, was an English pirate.",
        "lastrevid": 100,
        "fullurl": "https://en.wikipedia.org/wiki/Blackbeard",
    }
    page.update(overrides)
    return page


def api_response(*pages: dict[str, Any]) -> dict[str, Any]:
    return {"batchcomplete": True, "query": {"pages": list(pages)}}


class FakeEmbedder:
    """Bag-of-words vectors: texts that share words come out similar.

    Deterministic and instant, so tests of indexing and search need no model.
    Words are hashed with crc32 because the built-in ``hash`` changes per run.
    """

    def __init__(self, model_name: str = "fake-model", dimension: int = 64) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.document_calls: list[list[str]] = []

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        self.document_calls.append(list(texts))
        rows = [self.vector(text) for text in texts]
        return np.array(rows, dtype=np.float32).reshape(len(rows), self.dimension)

    def embed_query(self, text: str) -> Vectors:
        return self.vector(text)

    def vector(self, text: str) -> Vectors:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            vector[zlib.crc32(word.encode()) % self.dimension] += 1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector


class FakeChatModel:
    """Replies with fixed pieces of text and records the prompts it was sent."""

    def __init__(self, *pieces: str) -> None:
        self.pieces = list(pieces) or ["Blackbeard was killed in 1718 [1]."]
        self.prompts: list[list[Message]] = []

    def stream_chat(
        self, messages: Sequence[Message], *, temperature: float = 0.2
    ) -> Iterator[str]:
        self.prompts.append(list(messages))
        yield from self.pieces


class ScriptedChatModel:
    """Gives each call the next reply in order, e.g. a rewritten query and then an answer."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[list[Message]] = []
        self.temperatures: list[float] = []

    def stream_chat(
        self, messages: Sequence[Message], *, temperature: float = 0.2
    ) -> Iterator[str]:
        self.prompts.append(list(messages))
        self.temperatures.append(temperature)
        yield self.replies.pop(0)

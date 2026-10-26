"""Tests for the embedding wrapper, with sentence-transformers replaced by a fake module.

The real model is exercised in test_embeddings_integration.py.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from ragnosis.embeddings import SentenceTransformerEmbedder, prefixes_for


class FakeSentenceTransformer:
    """Records how it is called and returns unnormalized, float64 vectors."""

    instances: list[FakeSentenceTransformer] = []

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self.encoded: list[list[str]] = []
        self.encode_kwargs: dict[str, Any] = {}
        self.instances.append(self)

    def get_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts: list[str], **kwargs: Any) -> Any:
        self.encoded.append(texts)
        self.encode_kwargs = kwargs
        return np.ones((len(texts), 3), dtype=np.float64)


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> type[FakeSentenceTransformer]:
    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(FakeSentenceTransformer, "instances", [])
    return FakeSentenceTransformer


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("intfloat/multilingual-e5-small", ("query: ", "passage: ")),
        ("intfloat/e5-large-v2", ("query: ", "passage: ")),
        ("E5-base", ("query: ", "passage: ")),
        ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", ("", "")),
        ("BAAI/bge-m3", ("", "")),
        ("someone/model-e50", ("", "")),
        ("e5org/plain-model", ("", "")),
    ],
)
def test_prefixes_depend_on_model_family(model_name: str, expected: tuple[str, str]) -> None:
    assert prefixes_for(model_name) == expected


def test_loads_model_on_cpu(fake_model: type[FakeSentenceTransformer]) -> None:
    embedder = SentenceTransformerEmbedder("intfloat/multilingual-e5-small")
    (model,) = fake_model.instances
    assert model.model_name == "intfloat/multilingual-e5-small"
    assert model.device == "cpu"
    assert embedder.model_name == "intfloat/multilingual-e5-small"
    assert embedder.dimension == 3


def test_documents_get_passage_prefix_and_normalization(
    fake_model: type[FakeSentenceTransformer],
) -> None:
    embedder = SentenceTransformerEmbedder("intfloat/multilingual-e5-small", batch_size=8)
    vectors = embedder.embed_documents(["Blackbeard died.", "Rome fell."])

    (model,) = fake_model.instances
    assert model.encoded == [["passage: Blackbeard died.", "passage: Rome fell."]]
    assert model.encode_kwargs["normalize_embeddings"] is True
    assert model.encode_kwargs["batch_size"] == 8
    assert vectors.shape == (2, 3)
    assert vectors.dtype == np.float32


def test_query_gets_query_prefix_and_is_one_vector(
    fake_model: type[FakeSentenceTransformer],
) -> None:
    embedder = SentenceTransformerEmbedder("intfloat/multilingual-e5-small")
    vector = embedder.embed_query("How did Blackbeard die?")

    (model,) = fake_model.instances
    assert model.encoded == [["query: How did Blackbeard die?"]]
    assert vector.shape == (3,)
    assert vector.dtype == np.float32


def test_models_without_prefixes_get_text_unchanged(
    fake_model: type[FakeSentenceTransformer],
) -> None:
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3")
    embedder.embed_query("Who was Hannibal?")
    assert fake_model.instances[0].encoded == [["Who was Hannibal?"]]


def test_no_documents_gives_empty_matrix_without_calling_model(
    fake_model: type[FakeSentenceTransformer],
) -> None:
    embedder = SentenceTransformerEmbedder("intfloat/multilingual-e5-small")
    vectors = embedder.embed_documents([])
    assert vectors.shape == (0, 3)
    assert fake_model.instances[0].encoded == []

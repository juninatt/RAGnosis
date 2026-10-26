"""Checks against the real embedding model. Opt-in: ``uv run pytest -m integration``.

The first run downloads the model (about 0.5 GB) to the Hugging Face cache.
"""

from __future__ import annotations

import numpy as np
import pytest

from ragnosis.embeddings import SentenceTransformerEmbedder
from ragnosis.settings import DEFAULT_EMBEDDING_MODEL

pytestmark = pytest.mark.integration

PASSAGES = [
    "Blackbeard > Death\nTeach was killed in a battle near Ocracoke Island in 1718.",
    "Roman Senate > Introduction\nThe Senate was the governing assembly of ancient Rome.",
    "Anne Bonny > Introduction\nAnne Bonny was an Irish pirate operating in the Caribbean.",
]


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(DEFAULT_EMBEDDING_MODEL)


def test_vectors_are_normalized_and_sized(embedder: SentenceTransformerEmbedder) -> None:
    vectors = embedder.embed_documents(PASSAGES)
    assert vectors.shape == (3, embedder.dimension)
    assert np.linalg.norm(vectors, axis=1) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How did Blackbeard die?", 0),
        ("Hur dog Svartskägg?", 0),
        ("What did the senate do in Rome?", 1),
        ("kvinnliga pirater", 2),
    ],
)
def test_questions_in_english_and_swedish_find_the_right_passage(
    embedder: SentenceTransformerEmbedder, question: str, expected: int
) -> None:
    scores = embedder.embed_documents(PASSAGES) @ embedder.embed_query(question)
    assert int(np.argmax(scores)) == expected

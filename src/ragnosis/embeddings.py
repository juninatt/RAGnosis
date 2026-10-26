"""Turn text into vectors whose closeness reflects closeness in meaning.

Questions and passages are embedded with the same model, so finding passages
that answer a question becomes finding the nearest vectors. Vectors are
normalized to length 1, which makes cosine similarity a plain dot product.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

Vectors = NDArray[np.float32]


class Embedder(Protocol):
    """What the rest of RAGnosis needs from an embedding model."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        """One normalized row per text."""
        ...

    def embed_query(self, text: str) -> Vectors:
        """A single normalized vector."""
        ...


def prefixes_for(model_name: str) -> tuple[str, str]:
    """Return the (query, passage) prefixes the model was trained with.

    E5 models learned to tell questions from passages by these prefixes and give
    noticeably worse matches without them. Other models expect none.
    """
    if "e5" in model_name.lower().split("/")[-1].split("-"):
        return "query: ", "passage: "
    return "", ""


class SentenceTransformerEmbedder:
    """Runs a sentence-transformers model locally on the CPU."""

    def __init__(self, model_name: str, *, batch_size: int = 32) -> None:
        """Load the model, downloading it on first use (to the Hugging Face cache)."""
        # Imported here because loading PyTorch takes seconds; commands and tests
        # that never embed anything should not pay for it.
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s", model_name)
        self._model = SentenceTransformer(model_name, device="cpu")
        self._model_name = model_name
        self._batch_size = batch_size
        self._query_prefix, self._passage_prefix = prefixes_for(model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return int(self._model.get_embedding_dimension() or 0)

    def embed_documents(self, texts: Sequence[str]) -> Vectors:
        return self._encode([self._passage_prefix + text for text in texts])

    def embed_query(self, text: str) -> Vectors:
        vector: Vectors = self._encode([self._query_prefix + text])[0]
        return vector

    def _encode(self, texts: list[str]) -> Vectors:
        if not texts:
            # Keep the (rows, dimension) shape so empty results stack with others.
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

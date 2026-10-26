"""Retrieval quality on the real corpus: does each question find the right article?

These run against the local index built by ``python -m ragnosis index`` with the
real model, so they are opt-in (``uv run pytest -m integration``) and skipped
when no index exists. Rerun them after changing chunking, the embedding model
or the title list to see whether retrieval got better or worse.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from ragnosis.embeddings import SentenceTransformerEmbedder
from ragnosis.vector_store import VectorStore

pytestmark = pytest.mark.integration

INDEX_DIR = Path(__file__).resolve().parents[1] / "data" / "index"
TOP_K = 5

# Question, and articles any of which counts as a correct find.
CASES = [
    ("How did Blackbeard die?", {"Blackbeard"}),
    ("Hur dog Blackbeard?", {"Blackbeard"}),
    pytest.param(
        "Hur dog Svartskägg?",
        {"Blackbeard"},
        marks=pytest.mark.xfail(
            strict=True,
            reason="the model does not map translated names (Svartskägg) to the English ones",
        ),
    ),
    ("Vilken flagga hade pirater?", {"Jolly Roger"}),
    ("Which female pirates sailed with Calico Jack?", {"Anne Bonny", "Mary Read", "Calico Jack"}),
    ("Who led the Carthaginian army across the Alps?", {"Hannibal", "Punic Wars"}),
    ("Vilken lag gav Pompejus makt att bekämpa pirater?", {"Lex Gabinia", "Pompey"}),
    ("What kind of warship did the Athenians use at Salamis?", {"Trireme", "Battle of Salamis"}),
    ("När föll det västromerska riket?", {"Fall of the Western Roman Empire"}),
    ("Vem var Klaus Störtebeker?", {"Klaus Störtebeker", "Victual Brothers"}),
    ("Where did the Barbary corsairs operate from?", {"Barbary corsairs"}),
    ("What flag did pirates fly?", {"Jolly Roger"}),
]


@pytest.fixture(scope="module")
def search() -> Callable[[str], list[str]]:
    if not VectorStore.exists(INDEX_DIR):
        pytest.skip("No local index; run `python -m ragnosis index` first")
    store = VectorStore.load(INDEX_DIR)
    embedder = SentenceTransformerEmbedder(store.embedding_model)

    def top_titles(question: str) -> list[str]:
        results = store.search(embedder.embed_query(question), k=TOP_K)
        return [result.chunk.article_title for result in results]

    return top_titles


@pytest.mark.parametrize(("question", "expected"), CASES)
def test_question_finds_a_relevant_article(
    search: Callable[[str], list[str]], question: str, expected: set[str]
) -> None:
    titles = search(question)
    assert expected & set(titles), f"Top {TOP_K} for {question!r}: {titles}"

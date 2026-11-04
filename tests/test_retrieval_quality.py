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
from ragnosis.vector_store import SearchResult, VectorStore

pytestmark = pytest.mark.integration

INDEX_DIR = Path(__file__).resolve().parents[1] / "data" / "index"
TOP_K = 5

# Question, and articles any of which counts as a correct find.
CASES = [
    ("How did Blackbeard die?", {"Blackbeard"}),
    ("Hur dog Blackbeard?", {"Blackbeard"}),
    # Found by the other names from Wikidata stored with the passages (0011).
    ("Hur dog Svartskägg?", {"Blackbeard"}),
    ("Who was Calico Jack?", {"John Rackham"}),
    ("Who was Black Bart?", {"Bartholomew Roberts"}),
    ("Vem var Kapten Kidd?", {"William Kidd"}),
    ("Vilken flagga hade pirater?", {"Jolly Roger"}),
    ("Which female pirates sailed with Calico Jack?", {"Anne Bonny", "Mary Read", "John Rackham"}),
    ("Who led the Carthaginian army across the Alps?", {"Hannibal", "Punic Wars"}),
    (
        "Vilken lag gav Pompejus makt att bekämpa pirater?",
        {"Lex Gabinia de piratis persequendis", "Pompey"},
    ),
    ("What kind of warship did the Athenians use at Salamis?", {"Trireme", "Battle of Salamis"}),
    ("När föll det västromerska riket?", {"Fall of the Western Roman Empire"}),
    ("Vem var Klaus Störtebeker?", {"Klaus Störtebeker", "Victual Brothers"}),
    ("Where did the Barbary corsairs operate from?", {"Barbary corsairs"}),
    ("What flag did pirates fly?", {"Jolly Roger"}),
]


# Questions only a table answers, which the text extracts leave out (0012),
# and the section the table is in.
TABLE_CASES = [
    ("How many ships did Athens send to the Battle of Salamis?", "Greek fleet > Table"),
    ("Hur många skepp skickade Korinth till slaget vid Salamis?", "Greek fleet > Table"),
]


@pytest.fixture(scope="module")
def search() -> Callable[[str], list[SearchResult]]:
    if not VectorStore.exists(INDEX_DIR):
        pytest.skip("No local index; run `python -m ragnosis index` first")
    store = VectorStore.load(INDEX_DIR)
    embedder = SentenceTransformerEmbedder(store.embedding_model)

    def top(question: str) -> list[SearchResult]:
        return store.search(embedder.embed_query(question), k=TOP_K)

    return top


@pytest.mark.parametrize(("question", "expected"), CASES)
def test_question_finds_a_relevant_article(
    search: Callable[[str], list[SearchResult]], question: str, expected: set[str]
) -> None:
    titles = [result.chunk.article_title for result in search(question)]
    assert expected & set(titles), f"Top {TOP_K} for {question!r}: {titles}"


@pytest.mark.parametrize(("question", "section"), TABLE_CASES)
def test_question_finds_the_table(
    search: Callable[[str], list[SearchResult]], question: str, section: str
) -> None:
    sections = [result.chunk.section for result in search(question)]
    assert any(s.endswith(section) for s in sections), f"Top {TOP_K}: {sections}"

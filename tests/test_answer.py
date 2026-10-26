"""Tests for answering questions from retrieved passages, with fake models."""

from __future__ import annotations

import numpy as np
import pytest
from factories import FakeChatModel, FakeEmbedder, make_chunk
from hypothesis import given
from hypothesis import strategies as st

from ragnosis.answer import (
    NO_PASSAGES_TEXT,
    SAME_LANGUAGE_RULE,
    SWEDISH_RULE,
    SYSTEM_PROMPT,
    Answer,
    ask,
    build_messages,
    cited_numbers,
    is_swedish,
    system_prompt,
)
from ragnosis.llm import LLMError
from ragnosis.vector_store import SearchResult, VectorStore

PASSAGES = {
    1: ("Blackbeard", "Death", "Teach was killed at Ocracoke Inlet in 1718."),
    2: ("Hannibal", "Introduction", "Hannibal crossed the Alps with elephants."),
    3: ("Anne Bonny", "Introduction", "Anne Bonny was an Irish pirate."),
}


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store(embedder: FakeEmbedder) -> VectorStore:
    store = VectorStore(embedder.model_name, embedder.dimension)
    for page_id, (title, section, text) in PASSAGES.items():
        chunk = make_chunk(
            chunk_id=f"{page_id}-0",
            page_id=page_id,
            article_title=title,
            section=section,
            text=text,
            url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        )
        store.replace_article(page_id, 1, [chunk], np.array([embedder.vector(chunk.text)]))
    return store


def result(title: str, text: str = "Some text.", section: str = "Introduction") -> SearchResult:
    return SearchResult(make_chunk(article_title=title, section=section, text=text), 0.5)


# --- cited_numbers ---


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Killed in 1718 [1].", [1]),
        ("Killed [2] in 1718 [1].", [2, 1]),
        ("Both say so [1][3].", [1, 3]),
        ("Both say so [1, 3] and [2 ,1].", [1, 3, 2]),
        ("Repeated [1] and again [1].", [1]),
        ("No citations here.", []),
        ("Out of range [0] [4] [99].", []),
        ("Not citations: [a] [] [1.5] (1).", []),
    ],
)
def test_cited_numbers(text: str, expected: list[int]) -> None:
    assert cited_numbers(text, count=3) == expected


@given(st.text(), st.integers(0, 10))
def test_cited_numbers_are_unique_and_in_range(text: str, count: int) -> None:
    numbers = cited_numbers(text, count)
    assert len(numbers) == len(set(numbers))
    assert all(1 <= n <= count for n in numbers)


# --- build_messages ---


def test_prompt_has_instructions_then_numbered_sources_and_question() -> None:
    messages = build_messages(
        "How did Blackbeard die?",
        [result("Blackbeard", "Killed in 1718.", "Death"), result("Piracy", "Robbery at sea.")],
    )

    assert [m.role for m in messages] == ["system", "user"]
    assert messages[0].content == f"{SYSTEM_PROMPT} {SAME_LANGUAGE_RULE}"
    assert messages[1].content == (
        "Sources:\n\n"
        "[1] Blackbeard > Death\nKilled in 1718.\n\n"
        "[2] Piracy > Introduction\nRobbery at sea.\n\n"
        "Question: How did Blackbeard die?"
    )


@pytest.mark.parametrize("rule", ["only the numbered sources", "[1]", "cannot answer"])
def test_system_prompt_states_the_grounding_rules(rule: str) -> None:
    assert rule in SYSTEM_PROMPT


# --- Answer language ---


@pytest.mark.parametrize(
    "question",
    [
        "Vem var Anne Bonny?",
        "Vem ledde den karthagiska armén över Alperna?",
        "Vilken flagga hade pirater?",
        "Hur dog Blackbeard?",
        "VAD hände vid Salamis?",
        "Berätta om Hannibal",
        "Varför föll Rom?",
        "Störtebeker",
    ],
)
def test_recognises_swedish_questions(question: str) -> None:
    assert is_swedish(question)


@pytest.mark.parametrize(
    "question",
    [
        "Who was Anne Bonny?",
        "How did Blackbeard die?",
        "What flag did pirates fly?",
        "Hvem var Anne Bonny?",  # Norwegian and Danish
        "Hvordan døde Svartskjegg?",  # Norwegian
        "Hvad var Jolly Roger?",  # Danish
        "Hannibal",
        "",
        "Was ist die Hauptstadt?",  # German
    ],
)
def test_does_not_mistake_other_languages_for_swedish(question: str) -> None:
    assert not is_swedish(question)


def test_swedish_questions_are_told_to_answer_in_swedish() -> None:
    prompt = system_prompt("Vem var Anne Bonny?")
    assert prompt == f"{SYSTEM_PROMPT} {SWEDISH_RULE}"
    assert "Swedish" in prompt
    assert "Norwegian" in prompt


def test_other_questions_keep_the_same_language_rule() -> None:
    assert system_prompt("Who was Anne Bonny?") == f"{SYSTEM_PROMPT} {SAME_LANGUAGE_RULE}"


def test_prompt_for_a_swedish_question_carries_the_swedish_rule() -> None:
    messages = build_messages("Vem var Anne Bonny?", [result("Anne Bonny")])
    assert messages[0].content.endswith(SWEDISH_RULE)


# --- ask ---


def test_answers_from_the_best_matching_passages(
    store: VectorStore, embedder: FakeEmbedder
) -> None:
    llm = FakeChatModel("Teach was killed ", "at Ocracoke [1].")

    answer = ask("Where was Teach killed?", store=store, embedder=embedder, llm=llm, k=2)

    assert answer.text == "Teach was killed at Ocracoke [1]."
    assert answer.question == "Where was Teach killed?"
    assert len(answer.passages) == 2
    assert answer.passages[0].chunk.article_title == "Blackbeard"
    (prompt,) = llm.prompts
    assert "[1] Blackbeard > Death" in prompt[1].content
    assert prompt[1].content.endswith("Question: Where was Teach killed?")


def test_streams_pieces_to_the_callback_in_order(
    store: VectorStore, embedder: FakeEmbedder
) -> None:
    received: list[str] = []
    llm = FakeChatModel("Teach ", "died ", "[1].")

    ask("Teach?", store=store, embedder=embedder, llm=llm, on_text=received.append)

    assert received == ["Teach ", "died ", "[1]."]


def test_cited_lists_only_the_passages_the_answer_cites(
    store: VectorStore, embedder: FakeEmbedder
) -> None:
    llm = FakeChatModel("Irish pirate [2], unlike Teach [1][2] or [7].")

    answer = ask("Ocracoke Teach Irish", store=store, embedder=embedder, llm=llm, k=3)

    assert [(n, r.chunk.article_title) for n, r in answer.cited] == [
        (2, answer.passages[1].chunk.article_title),
        (1, answer.passages[0].chunk.article_title),
    ]


def test_answer_without_citations_cites_nothing(store: VectorStore, embedder: FakeEmbedder) -> None:
    llm = FakeChatModel("I cannot answer from the sources.")
    answer = ask("Capital of Japan?", store=store, embedder=embedder, llm=llm)
    assert answer.cited == []


def test_question_is_trimmed(store: VectorStore, embedder: FakeEmbedder) -> None:
    llm = FakeChatModel()
    answer = ask("  Teach?  \n", store=store, embedder=embedder, llm=llm)
    assert answer.question == "Teach?"
    assert llm.prompts[0][1].content.endswith("Question: Teach?")


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_blank_question_raises_without_calling_models(
    store: VectorStore, embedder: FakeEmbedder, question: str
) -> None:
    llm = FakeChatModel()
    with pytest.raises(ValueError, match="blank"):
        ask(question, store=store, embedder=embedder, llm=llm)
    assert llm.prompts == []


def test_empty_index_answers_without_calling_the_model(embedder: FakeEmbedder) -> None:
    empty = VectorStore(embedder.model_name, embedder.dimension)
    llm = FakeChatModel()
    received: list[str] = []

    answer = ask("Teach?", store=empty, embedder=embedder, llm=llm, on_text=received.append)

    assert answer == Answer("Teach?", NO_PASSAGES_TEXT, [])
    assert received == [NO_PASSAGES_TEXT]
    assert llm.prompts == []


def test_works_without_a_callback_on_an_empty_index(embedder: FakeEmbedder) -> None:
    empty = VectorStore(embedder.model_name, embedder.dimension)
    answer = ask("Teach?", store=empty, embedder=embedder, llm=FakeChatModel())
    assert answer.text == NO_PASSAGES_TEXT


def test_model_errors_reach_the_caller(store: VectorStore, embedder: FakeEmbedder) -> None:
    class FailingModel(FakeChatModel):
        def stream_chat(self, messages, *, temperature=0.2):  # type: ignore[no-untyped-def]
            yield "Partial "
            raise LLMError("model crashed")

    with pytest.raises(LLMError, match="model crashed"):
        ask("Teach?", store=store, embedder=embedder, llm=FailingModel())

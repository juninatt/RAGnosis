"""Tests for answering questions from retrieved passages, with fake models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from factories import FakeChatModel, FakeEmbedder, ScriptedChatModel, make_chunk
from hypothesis import given
from hypothesis import strategies as st

from ragnosis.answer import (
    ANSWER_LENGTHS,
    ANSWER_TEMPERATURE,
    DEFAULT_K,
    LENGTH_RULES,
    MAX_QUERY_CHARS,
    NO_PASSAGES_TEXT,
    REWRITE_EXAMPLES,
    REWRITE_PROMPT,
    SAME_LANGUAGE_RULE,
    SWEDISH_RULE,
    SYSTEM_PROMPT,
    Answer,
    ask,
    build_messages,
    cited_numbers,
    is_swedish,
    rewrite_query,
    should_rewrite,
    system_prompt,
)
from ragnosis.llm import LLMError, Message
from ragnosis.settings import RewriteMode
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
    assert messages[0].content == f"{SYSTEM_PROMPT} {LENGTH_RULES['short']} {SAME_LANGUAGE_RULE}"
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
    assert prompt == f"{SYSTEM_PROMPT} {LENGTH_RULES['short']} {SWEDISH_RULE}"
    assert "Swedish" in prompt
    assert "Norwegian" in prompt


def test_other_questions_keep_the_same_language_rule() -> None:
    assert (
        system_prompt("Who was Anne Bonny?")
        == f"{SYSTEM_PROMPT} {LENGTH_RULES['short']} {SAME_LANGUAGE_RULE}"
    )


# --- Answer length ---


def test_short_is_the_default_length() -> None:
    assert system_prompt("Who was Anne Bonny?") == system_prompt("Who was Anne Bonny?", "short")
    assert LENGTH_RULES["short"] == "Be concise: at most 5 sentences."


def test_detailed_answers_ask_for_a_paragraph_or_two() -> None:
    prompt = system_prompt("Who was Anne Bonny?", "detailed")
    assert prompt == f"{SYSTEM_PROMPT} {LENGTH_RULES['detailed']} {SAME_LANGUAGE_RULE}"
    assert "6 to 10 sentences" in prompt
    assert "at most 5 sentences" not in prompt


def test_every_length_keeps_the_grounding_and_language_rules() -> None:
    for length in ANSWER_LENGTHS:
        prompt = system_prompt("Vem var Anne Bonny?", length)
        assert prompt.startswith(SYSTEM_PROMPT)
        assert prompt.endswith(SWEDISH_RULE)


def test_ask_passes_the_length_to_the_prompt(store: VectorStore, embedder: FakeEmbedder) -> None:
    llm = ScriptedChatModel("Teach was killed at Ocracoke [1].")
    ask("Where was Teach killed?", store=store, embedder=embedder, llm=llm, length="detailed")
    assert LENGTH_RULES["detailed"] in llm.prompts[0][0].content


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


def test_reads_four_passages_by_default(embedder: FakeEmbedder) -> None:
    store = VectorStore(embedder.model_name, embedder.dimension)
    chunks = [make_chunk(chunk_id=f"9-{i}", page_id=9, position=i) for i in range(6)]
    store.replace_article(9, 1, chunks, np.array([embedder.vector(c.text) for c in chunks]))
    llm = FakeChatModel()

    answer = ask("Teach?", store=store, embedder=embedder, llm=llm)

    assert DEFAULT_K == 4
    assert len(answer.passages) == 4
    prompt = llm.prompts[0][1].content
    assert "\n[4] " in prompt
    assert "\n[5] " not in prompt


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


# --- Rewriting the question for search ---


@pytest.mark.parametrize(
    ("question", "mode", "expected"),
    [
        ("Hur dog Svartskägg?", "swedish", True),
        ("How did Blackbeard die?", "swedish", False),
        ("How did Blackbeard die?", "always", True),
        ("Hur dog Svartskägg?", "never", False),
    ],
)
def test_should_rewrite(question: str, mode: RewriteMode, expected: bool) -> None:
    assert should_rewrite(question, mode) is expected


@pytest.mark.parametrize(
    ("reply", "query"),
    [
        ("How did Blackbeard die", "How did Blackbeard die"),
        ('"How did Blackbeard die"', "How did Blackbeard die"),
        ("“Blackbeard death”", "Blackbeard death"),
        ("`Blackbeard death`", "Blackbeard death"),
        ("Search query: Blackbeard death", "Blackbeard death"),
        ("English search query: Blackbeard death", "Blackbeard death"),
        ("Query: Blackbeard death", "Blackbeard death"),
        ("\n\nBlackbeard death\nThis query uses the English name.", "Blackbeard death"),
        ("  Blackbeard death  ", "Blackbeard death"),
    ],
)
def test_rewrite_query_cleans_up_the_reply(reply: str, query: str) -> None:
    assert rewrite_query("Hur dog Svartskägg?", ScriptedChatModel(reply)) == query


@pytest.mark.parametrize("reply", ["", "   \n  ", '""', "Query:"])
def test_empty_rewrite_falls_back_to_the_question(reply: str) -> None:
    assert rewrite_query("Hur dog Svartskägg?", ScriptedChatModel(reply)) == "Hur dog Svartskägg?"


def test_overlong_rewrite_is_cut() -> None:
    query = rewrite_query("Q?", ScriptedChatModel("word " * 100))
    assert len(query) == MAX_QUERY_CHARS


def test_rewrite_shows_the_model_examples_before_the_question() -> None:
    llm = ScriptedChatModel("Blackbeard death")
    rewrite_query("Hur dog Svartskägg?", llm)

    (prompt,) = llm.prompts
    assert prompt[0] == Message("system", REWRITE_PROMPT)
    assert [m.role for m in prompt[1:-1]] == ["user", "assistant"] * len(REWRITE_EXAMPLES)
    assert prompt[1:3] == [
        Message("user", REWRITE_EXAMPLES[0][0]),
        Message("assistant", REWRITE_EXAMPLES[0][1]),
    ]
    assert prompt[-1] == Message("user", "Hur dog Svartskägg?")
    assert llm.temperatures == [0.0]


def test_rewrite_prompt_does_not_give_away_evaluation_answers() -> None:
    # Examples taken from the known questions would make the scorecard measure
    # what the prompt told the model rather than what it can do.
    from ragnosis.evaluation import DEFAULT_CASES_FILE, load_cases

    repo = Path(__file__).resolve().parents[1]
    prompt_text = " ".join([REWRITE_PROMPT, *(q + " " + k for q, k in REWRITE_EXAMPLES)])
    for case in load_cases(repo / DEFAULT_CASES_FILE):
        assert case.question not in prompt_text
    assert "Svartskägg" not in prompt_text


def test_ask_searches_with_the_rewritten_query_but_answers_the_question(
    store: VectorStore, embedder: FakeEmbedder
) -> None:
    # No word of the Swedish question matches; the English query finds Ocracoke.
    llm = ScriptedChatModel("Teach killed Ocracoke Inlet", "Han dödades vid Ocracoke [1].")

    answer = ask("Hur dog Svartskägg?", store=store, embedder=embedder, llm=llm, rewrite="swedish")

    assert answer.search_query == "Teach killed Ocracoke Inlet"
    assert answer.passages[0].chunk.article_title == "Blackbeard"
    assert answer.question == "Hur dog Svartskägg?"
    answer_prompt = llm.prompts[1]
    assert answer_prompt[1].content.endswith("Question: Hur dog Svartskägg?")
    assert answer_prompt[0].content.endswith(SWEDISH_RULE)


def test_answers_are_written_with_temperature_zero(
    store: VectorStore, embedder: FakeEmbedder
) -> None:
    llm = ScriptedChatModel("Han dödades [1].")
    ask("Where was Teach killed?", store=store, embedder=embedder, llm=llm)
    assert llm.temperatures == [ANSWER_TEMPERATURE] == [0.0]


def test_ask_does_not_rewrite_unless_asked(store: VectorStore, embedder: FakeEmbedder) -> None:
    llm = ScriptedChatModel("Han dödades [1].")
    answer = ask("Hur dog Svartskägg?", store=store, embedder=embedder, llm=llm)
    assert answer.search_query is None
    assert len(llm.prompts) == 1


def test_swedish_mode_leaves_english_questions_alone(
    store: VectorStore, embedder: FakeEmbedder
) -> None:
    llm = ScriptedChatModel("He was killed [1].")
    answer = ask(
        "Where was Teach killed?", store=store, embedder=embedder, llm=llm, rewrite="swedish"
    )
    assert answer.search_query is None
    assert len(llm.prompts) == 1


def test_rewritten_query_is_kept_when_nothing_is_found(embedder: FakeEmbedder) -> None:
    empty = VectorStore(embedder.model_name, embedder.dimension)
    llm = ScriptedChatModel("Blackbeard death")
    answer = ask("Hur dog Svartskägg?", store=empty, embedder=embedder, llm=llm, rewrite="always")
    assert answer == Answer("Hur dog Svartskägg?", NO_PASSAGES_TEXT, [], "Blackbeard death")


def test_model_errors_reach_the_caller(store: VectorStore, embedder: FakeEmbedder) -> None:
    class FailingModel(FakeChatModel):
        def stream_chat(self, messages, *, temperature=0.2):  # type: ignore[no-untyped-def]
            yield "Partial "
            raise LLMError("model crashed")

    with pytest.raises(LLMError, match="model crashed"):
        ask("Teach?", store=store, embedder=embedder, llm=FailingModel())

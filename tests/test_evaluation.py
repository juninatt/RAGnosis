"""Tests for the answer checks and the cases file, with hand-written answers."""

from __future__ import annotations

from pathlib import Path

import pytest
from factories import make_chunk

from ragnosis.answer import Answer
from ragnosis.evaluation import (
    DEFAULT_CASES_FILE,
    Case,
    CaseError,
    answer_language,
    check,
    load_cases,
)
from ragnosis.vector_store import SearchResult

REPO_ROOT = Path(__file__).resolve().parents[1]

PASSAGES = [
    SearchResult(make_chunk(article_title="Blackbeard", section="Death"), 0.9),
    SearchResult(make_chunk(article_title="Piracy"), 0.8),
]

BLACKBEARD_SV = Case(
    id="blackbeard-sv",
    question="Hur dog Blackbeard?",
    language="sv",
    sources=("Blackbeard",),
    mentions=("Ocracoke", "1718"),
)
DECLINES_EN = Case(id="japan", question="Capital of Japan?", language="en", declines=True)


def answer(text: str, passages: list[SearchResult] | None = None) -> Answer:
    return Answer("q", text, PASSAGES if passages is None else passages)


# --- answer_language ---


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("Han dödades vid Ocracoke och hans huvud fördes till Virginia [1].", "sv"),
        ("Hon var en pirat som också seglade med Rackham, enligt källorna.", "sv"),
        # A real answer the first scorecard misread as "unknown" (one marker only).
        (
            "Hannibal ledde den karthagiska armén över Alperna [2]. Han samlade en armé "
            "i Iberien och ledde den längs kusten. Armén korsade Alperna för att nå Italien.",
            "sv",
        ),
        # A real answer the second scorecard misread: one marker, but ä and ö.
        (
            "Svartskägg var en engelsk pirat som opererade i Västindien och på "
            "Britanniens nordamerikanska kolonier [2]. Han fick sitt namn av sitt skägg.",
            "sv",
        ),
        ("He was killed at Ocracoke and his head was taken to Virginia [1].", "en"),
        # Swedish letters in names do not make English text Swedish.
        ("Störtebeker was a pirate of the Baltic, and his crew were the Vitalienbrüder.", "en"),
        ("Hun kaldes Svartskägg og var ikke kendt før 1717.", "no/da"),
        ("Hun var en pirat som seilte med Rackham og ble kjent i 1720.", "no/da"),
        ("Hun er kendt som pirat og havde ikke noget skib.", "no/da"),
        ("Blackbeard [1].", "unknown"),
        ("", "unknown"),
    ],
)
def test_answer_language(text: str, language: str) -> None:
    assert answer_language(text) == language


# --- check ---


def test_good_swedish_answer_passes() -> None:
    verdict = check(BLACKBEARD_SV, answer("Han dödades vid Ocracoke 1718 och är begravd där [1]."))
    assert verdict.passed
    assert verdict.problems == []


def test_wrong_source_is_reported() -> None:
    verdict = check(BLACKBEARD_SV, answer("Han dödades vid Ocracoke och är känd [2]."))
    assert verdict.problems == ["cites ['Piracy'], expected one of ['Blackbeard']"]


def test_no_citation_is_reported() -> None:
    verdict = check(BLACKBEARD_SV, answer("Han dödades vid Ocracoke och är känd."))
    assert verdict.problems == ["cites nothing, expected one of ['Blackbeard']"]


def test_missing_fact_is_reported() -> None:
    verdict = check(BLACKBEARD_SV, answer("Han dödades och är inte glömd [1]."))
    assert verdict.problems == ["mentions none of ['Ocracoke', '1718']"]


def test_mentions_are_matched_without_case() -> None:
    assert check(BLACKBEARD_SV, answer("Han dödades vid OCRACOKE och är känd [1].")).passed


def test_wrong_language_is_reported() -> None:
    verdict = check(BLACKBEARD_SV, answer("He was killed at Ocracoke in 1718 and his ship [1]."))
    assert verdict.problems == ["is in en, expected sv"]


def test_norwegian_answer_to_swedish_question_is_reported() -> None:
    text = "Han ble drept ved Ocracoke i 1718, og hun var ikke der [1]."
    assert check(BLACKBEARD_SV, answer(text)).problems == ["is in no/da, expected sv"]


def test_every_problem_is_reported() -> None:
    verdict = check(BLACKBEARD_SV, answer("He was famous and the sea was his home."))
    assert len(verdict.problems) == 3
    assert not verdict.passed


@pytest.mark.parametrize(
    "text",
    [
        "The sources do not contain the capital of Japan, so I cannot answer.",
        "I cannot answer this from the sources; they are about piracy.",
    ],
)
def test_declining_answer_passes(text: str) -> None:
    assert check(DECLINES_EN, answer(text)).passed


def test_swedish_declining_answer_passes() -> None:
    case = Case(id="vm", question="Vem vann VM?", language="sv", declines=True)
    text = "Källorna innehåller inte information om fotboll, så jag kan inte svara på det."
    assert check(case, answer(text)).passed


def test_answer_that_should_decline_but_cites_is_reported() -> None:
    verdict = check(DECLINES_EN, answer("The capital is Tokyo, as the sources say [1]."))
    assert verdict.problems == [
        "should decline but cites ['Blackbeard']",
        "should decline but does not say it cannot answer",
    ]


def test_uncited_made_up_answer_to_decline_case_is_reported() -> None:
    verdict = check(DECLINES_EN, answer("The capital of Japan is Tokyo, which is a big city."))
    assert verdict.problems == ["should decline but does not say it cannot answer"]


# --- load_cases ---


def write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "cases.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_cases_with_defaults(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        '[[case]]\nid = "a"\nquestion = "Q?"\nlanguage = "en"\nmentions = ["x"]\n\n'
        '[[case]]\nid = "b"\nquestion = "R?"\nlanguage = "sv"\ndeclines = true\n',
    )
    assert load_cases(path) == [
        Case(id="a", question="Q?", language="en", mentions=("x",)),
        Case(id="b", question="R?", language="sv", declines=True),
    ]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not = [toml", "not valid TOML"),
        ("title = 'no cases'\n", "no \\[\\[case\\]\\] tables"),
        ('[[case]]\nquestion = "Q?"\nlanguage = "en"\nmentions = ["x"]\n', "non-empty 'id'"),
        ('[[case]]\nid = "a"\nquestion = " "\nlanguage = "en"\nmentions = ["x"]\n', "'question'"),
        ('[[case]]\nid = "a"\nquestion = "Q?"\nlanguage = "de"\nmentions = ["x"]\n', "language"),
        ('[[case]]\nid = "a"\nquestion = "Q?"\nlanguage = "en"\nmentions = "x"\n', "list of"),
        ('[[case]]\nid = "a"\nquestion = "Q?"\nlanguage = "en"\nsources = [""]\n', "list of"),
        ('[[case]]\nid = "a"\nquestion = "Q?"\nlanguage = "en"\ndeclines = "yes"\n', "true or"),
        ('[[case]]\nid = "a"\nquestion = "Q?"\nlanguage = "en"\n', "expects nothing"),
        (
            '[[case]]\nid = "a"\nquestion = "Q?"\nlanguage = "en"\ndeclines = true\n'
            'mentions = ["x"]\n',
            "cannot also expect",
        ),
        (
            '[[case]]\nid = "a"\nquestion = "Q?"\nlanguage = "en"\nmentions = ["x"]\n'
            'answer = "cheat"\n',
            "unknown fields \\['answer'\\]",
        ),
        (
            '[[case]]\nid = "a"\nquestion = "Q?"\nlanguage = "en"\nmentions = ["x"]\n'
            '[[case]]\nid = "a"\nquestion = "R?"\nlanguage = "en"\nmentions = ["y"]\n',
            "Duplicate case ids: \\['a'\\]",
        ),
    ],
    ids=[
        "not-toml",
        "no-cases",
        "no-id",
        "blank-question",
        "bad-language",
        "mentions-not-list",
        "empty-source",
        "declines-not-bool",
        "expects-nothing",
        "declines-and-expects",
        "unknown-field",
        "duplicate-id",
    ],
)
def test_malformed_cases_are_rejected(tmp_path: Path, content: str, message: str) -> None:
    with pytest.raises(CaseError, match=message):
        load_cases(write(tmp_path, content))


# --- Guards on the committed cases ---


def test_committed_cases_load() -> None:
    cases = load_cases(REPO_ROOT / DEFAULT_CASES_FILE)
    assert len(cases) >= 10


def test_committed_cases_cover_both_languages_and_declining() -> None:
    cases = load_cases(REPO_ROOT / DEFAULT_CASES_FILE)
    assert {case.language for case in cases} == {"sv", "en"}
    assert any(case.declines for case in cases)


def test_committed_sources_name_articles_in_the_topics_file() -> None:
    # A typo in a source title would make a case impossible to pass.
    from ragnosis.settings import DEFAULT_TOPICS_FILE
    from ragnosis.topics import load_titles

    titles = set(load_titles(REPO_ROOT / DEFAULT_TOPICS_FILE)) | {"Barbary corsairs"}
    for case in load_cases(REPO_ROOT / DEFAULT_CASES_FILE):
        assert set(case.sources) <= titles, case.id

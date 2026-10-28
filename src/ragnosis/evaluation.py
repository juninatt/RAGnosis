"""Check answers against known questions (docs/decisions/0008).

Each case says what a good answer looks like: which articles it cites, which
facts it mentions, which language it is in, or that it declines. The checks are
plain rules, so a verdict is the same every time and says why it failed.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ragnosis.answer import Answer

Language = Literal["sv", "en"]
LANGUAGES: tuple[Language, ...] = ("sv", "en")
DEFAULT_CASES_FILE = Path("evaluation/answers.toml")

# Words that are frequent in one language and rare in the others. Swedish,
# Norwegian and Danish share many words, so only telling ones are listed.
_MARKERS: dict[str, frozenset[str]] = {
    "sv": frozenset(
        [
            "och",
            "är",
            "inte",
            "hon",
            "också",
            "när",
            "där",
            "från",
            "dödades",
            "föddes",
            "innan",
            "vilket",
            "vilka",
            "enligt",
            "även",
            # Norwegian and Danish write these å/at, for, over, hadde/havde,
            # siden, mellom/mellem, gjennom/gennem and uten/uden.
            "att",
            "för",
            "över",
            "hade",
            "sedan",
            "mellan",
            "genom",
            "utan",
        ]
    ),
    "no/da": frozenset(
        ["og", "ikke", "hun", "også", "ble", "etter", "hvor", "kjent", "kendt", "havde", "fra"]
    ),
    "en": frozenset(
        [
            "the",
            "and",
            "was",
            "is",
            "of",
            "were",
            "his",
            "her",
            "she",
            "he",
            "with",
            "from",
            "which",
        ]
    ),
}
_WORD = re.compile(r"[a-zåäöæø]+")

# How a declining answer tends to start, in the languages the cases use.
_DECLINE_PHRASES = (
    "cannot answer",
    "can't answer",
    "do not contain",
    "does not contain",
    "don't contain",
    "no information",
    "not mentioned",
    "kan inte svara",
    "innehåller inte",
    "innehåller ingen",
    "saknar information",
    "finns ingen information",
    "nämns inte",
)


class CaseError(ValueError):
    """The cases file is malformed."""


@dataclass(frozen=True)
class Case:
    """One known question and what a good answer to it looks like."""

    id: str
    question: str
    language: Language
    sources: tuple[str, ...] = ()
    mentions: tuple[str, ...] = ()
    declines: bool = False


@dataclass(frozen=True)
class Verdict:
    """How one answer measured up; empty ``problems`` means it passed."""

    case: Case
    answer: Answer
    problems: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.problems


def answer_language(text: str) -> str:
    """ "sv", "en" or "no/da", or "unknown" when the text gives too little to go on.

    Telling words decide, helped by the letters ä and ö: Swedish text nearly
    always has them, Norwegian and Danish write æ and ø instead. Names such as
    Störtebeker put them in English text too, so the letters only count when
    Swedish words are at least as frequent as the other languages' words.
    """
    lowered = text.lower()
    counts = {language: 0 for language in _MARKERS}
    for word in _WORD.findall(lowered):
        for language, markers in _MARKERS.items():
            if word in markers:
                counts[language] += 1
    swedish_letters = lowered.count("ä") + lowered.count("ö")
    if swedish_letters >= 2 and counts["sv"] >= max(counts["no/da"], counts["en"]):
        return "sv"
    best = max(counts, key=lambda language: counts[language])
    return best if counts[best] >= 2 else "unknown"


def check(case: Case, answer: Answer) -> Verdict:
    """Compare ``answer`` with what ``case`` expects."""
    problems: list[str] = []
    text = answer.text.lower()
    cited_titles = {result.chunk.article_title for _, result in answer.cited}

    if case.declines:
        if cited_titles:
            problems.append(f"should decline but cites {sorted(cited_titles)}")
        if not any(phrase in text for phrase in _DECLINE_PHRASES):
            problems.append("should decline but does not say it cannot answer")
    else:
        if case.sources and not cited_titles & set(case.sources):
            found = sorted(cited_titles) or "nothing"
            problems.append(f"cites {found}, expected one of {list(case.sources)}")
        if case.mentions and not any(m.lower() in text for m in case.mentions):
            problems.append(f"mentions none of {list(case.mentions)}")

    language = answer_language(answer.text)
    if language != case.language:
        problems.append(f"is in {language}, expected {case.language}")
    return Verdict(case, answer, problems)


def load_cases(path: Path) -> list[Case]:
    """Read and validate the cases in a TOML file of ``[[case]]`` tables."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise CaseError(f"{path} is not valid TOML: {exc}") from exc
    tables = data.get("case")
    if not isinstance(tables, list) or not tables:
        raise CaseError(f"{path} has no [[case]] tables")

    cases = [_parse_case(table, number) for number, table in enumerate(tables, start=1)]
    ids = [case.id for case in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise CaseError(f"Duplicate case ids: {duplicates}")
    return cases


_FIELDS = {"id", "question", "language", "sources", "mentions", "declines"}


def _parse_case(table: dict[str, Any], number: int) -> Case:
    where = f"case {number}"
    unknown = set(table) - _FIELDS
    if unknown:
        raise CaseError(f"{where} has unknown fields {sorted(unknown)}")
    for name in ("id", "question"):
        if not isinstance(table.get(name), str) or not table[name].strip():
            raise CaseError(f"{where} needs a non-empty {name!r}")
    if table.get("language") not in LANGUAGES:
        raise CaseError(f"{where} needs 'language' set to one of {list(LANGUAGES)}")
    for name in ("sources", "mentions"):
        value = table.get(name, [])
        if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
            raise CaseError(f"{where}: {name!r} must be a list of non-empty strings")
    declines = table.get("declines", False)
    if not isinstance(declines, bool):
        raise CaseError(f"{where}: 'declines' must be true or false")
    if not declines and not (table.get("sources") or table.get("mentions")):
        raise CaseError(f"{where} expects nothing; give sources, mentions or declines")
    if declines and (table.get("sources") or table.get("mentions")):
        raise CaseError(f"{where} declines, so it cannot also expect sources or mentions")
    return Case(
        id=table["id"],
        question=table["question"],
        language=table["language"],
        sources=tuple(table.get("sources", [])),
        mentions=tuple(table.get("mentions", [])),
        declines=declines,
    )

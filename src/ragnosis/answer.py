"""Answer a question from retrieved passages, citing them.

The passages that best match the question are numbered in the prompt, and the
model is told to use only those and to cite them as [1], [2] (see
docs/decisions/0006). The citations are parsed back out, so the sources listed
with an answer are the ones the model actually used.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from ragnosis.embeddings import Embedder
from ragnosis.llm import Message
from ragnosis.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)

DEFAULT_K = 5

SYSTEM_PROMPT = (
    "You answer questions using only the numbered sources given. "
    "Cite every claim with the source number in brackets, like [1] or [2][3]. "
    "If the sources do not contain the answer, say that you cannot answer from the sources. "
    "Answer in the same language as the question. Be concise: at most 5 sentences."
)
NO_PASSAGES_TEXT = "There are no indexed passages to answer from; run ingest and index first."

# "[1]", "[2][3]" and "[1, 2]" all occur in model output.
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


class ChatModel(Protocol):
    """What answering needs from a language model; the real one is OllamaClient."""

    def stream_chat(
        self, messages: Sequence[Message], *, temperature: float = ...
    ) -> Iterator[str]: ...


@dataclass(frozen=True)
class Answer:
    """The model's answer and the passages it was given, numbered from 1."""

    question: str
    text: str
    passages: list[SearchResult]

    @property
    def cited(self) -> list[tuple[int, SearchResult]]:
        """The passages the answer cites, with their numbers, in order of first citation."""
        return [(n, self.passages[n - 1]) for n in cited_numbers(self.text, len(self.passages))]


def cited_numbers(text: str, count: int) -> list[int]:
    """Source numbers cited in ``text``, first occurrence first.

    Numbers outside 1..``count`` are ignored: a model sometimes cites a source
    it was never given, and listing a wrong source is worse than listing none.
    """
    numbers: list[int] = []
    for match in _CITATION.finditer(text):
        for part in match.group(1).split(","):
            number = int(part)
            if 1 <= number <= count and number not in numbers:
                numbers.append(number)
    return numbers


def build_messages(question: str, passages: Sequence[SearchResult]) -> list[Message]:
    """The prompt: instructions, then the numbered passages and the question."""
    sources = "\n\n".join(
        f"[{number}] {result.chunk.article_title} > {result.chunk.section}\n{result.chunk.text}"
        for number, result in enumerate(passages, start=1)
    )
    return [
        Message("system", SYSTEM_PROMPT),
        Message("user", f"Sources:\n\n{sources}\n\nQuestion: {question}"),
    ]


def ask(
    question: str,
    *,
    store: VectorStore,
    embedder: Embedder,
    llm: ChatModel,
    k: int = DEFAULT_K,
    on_text: Callable[[str], None] | None = None,
) -> Answer:
    """Retrieve the ``k`` best passages and have ``llm`` answer from them.

    ``on_text`` receives the answer piece by piece while it is written, so a
    caller can show it immediately instead of after the whole minute.

    Raises:
        ValueError: the question is blank.
        LLMError: the model could not answer (see ragnosis.llm).
    """
    question = question.strip()
    if not question:
        raise ValueError("The question must not be blank")

    passages = store.search(embedder.embed_query(question), k)
    logger.info(
        "Answering from: %s", ", ".join(r.chunk.article_title for r in passages) or "nothing"
    )
    if not passages:
        if on_text:
            on_text(NO_PASSAGES_TEXT)
        return Answer(question, NO_PASSAGES_TEXT, [])

    pieces: list[str] = []
    for piece in llm.stream_chat(build_messages(question, passages)):
        pieces.append(piece)
        if on_text:
            on_text(piece)
    return Answer(question, "".join(pieces), passages)

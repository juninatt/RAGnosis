"""Answer a question from retrieved passages, citing them.

The passages that best match the question are numbered in the prompt, and the
model is told to use only those and to cite them as [1], [2] (see
docs/decisions/0006). The citations are parsed back out, so the sources listed
with an answer are the ones the model actually used.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from ragnosis.embeddings import Embedder
from ragnosis.llm import Message
from ragnosis.settings import RewriteMode
from ragnosis.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)

# Reading the passages is most of the wait before the first word on a CPU
# (about 50 tokens a second). Four found a relevant passage for as many known
# questions as five did, with a fifth less to read; three missed one.
DEFAULT_K = 4
# The answer should restate the sources, not vary from run to run; with
# temperature 0 the model always picks its most likely next word (0008).
ANSWER_TEMPERATURE = 0.0

SYSTEM_PROMPT = (
    "You answer questions using only the numbered sources given. "
    "Cite every claim with the source number in brackets, like [1] or [2][3]. "
    "If the sources do not contain the answer, say that you cannot answer from the sources."
)
# The reader chooses how long an answer is. A detailed one takes 10 to 15
# seconds longer in total, but starts as soon and is written faster than read.
AnswerLength = Literal["short", "detailed"]
ANSWER_LENGTHS: tuple[AnswerLength, ...] = ("short", "detailed")
LENGTH_RULES: dict[AnswerLength, str] = {
    "short": "Be concise: at most 5 sentences.",
    "detailed": (
        "Answer in one or two short paragraphs, about 6 to 10 sentences, "
        "with the most important facts first."
    ),
}
# Asked to answer "in the question's language", the model answered Swedish
# questions in Norwegian or English; naming the language fixes that (0006).
SAME_LANGUAGE_RULE = "Answer in the same language as the question."
SWEDISH_RULE = (
    "The question is in Swedish: answer in Swedish (svenska), not in Norwegian, Danish or English."
)
NO_PASSAGES_TEXT = "There are no indexed passages to answer from; run ingest and index first."
# When even the best passage scores below the minimum, none is about the
# question: the model would decline anyway, but often cites the passages while
# doing so (0010). Answering without it gives no sources and is faster.
NOT_COVERED_TEXT = "I cannot answer from the sources: none of them is relevant to the question."
NOT_COVERED_TEXT_SV = "Jag kan inte svara utifrån källorna: ingen av dem är relevant för frågan."

# Rewriting the question into an English search query fixes searches the
# embedding model gets wrong, such as Swedish names ("Svartskägg" for
# Blackbeard). It costs an extra model call, so it can be limited (0008).
# Keywords rather than a question: "How did Blackbeard die?" ranked the passage
# about his death 8th, "Blackbeard killed in battle" ranked it 1st. The small
# model follows examples better than descriptions. None of the examples may come
# from evaluation/answers.toml, or the scorecard would measure what it was told.
REWRITE_PROMPT = (
    "Turn the user's question into a search query for an English encyclopedia: "
    "4 to 8 English keywords. Translate names to the standard English names of people, "
    "places and events, for example 'Karthago' becomes 'Carthage' and 'Djingis khan' "
    "becomes 'Genghis Khan'. Add words for what the question asks about. "
    "Reply with the keywords only."
)
REWRITE_EXAMPLES = (
    ("Hur dog Julius Caesar?", "Julius Caesar assassination death killed Ides of March"),
    ("Vad var Hansan?", "Hanseatic League trade towns Baltic"),
    ("Who built the Parthenon?", "Parthenon construction architects Pericles Athens"),
    ("Varför gick Titanic under?", "Titanic sinking cause iceberg 1912"),
)
MAX_QUERY_CHARS = 200
# Reciprocal rank fusion's usual constant: the first places differ little, so
# one list cannot outweigh the other with its first passage alone.
RANK_OFFSET = 60
_QUERY_LABEL = re.compile(r"^(english )?(search )?query\s*:\s*", re.IGNORECASE)

# "[1]", "[2][3]" and "[1, 2]" all occur in model output.
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# Words that start or carry Swedish questions and occur neither in English nor
# in Norwegian or Danish (hvem, hvad, hvordan). "var" is left out: it is
# Norwegian and Danish too.
_SWEDISH_WORDS = frozenset(
    {
        "vem", "vad", "vilken", "vilket", "vilka", "när", "hur", "varför",
        "är", "och", "inte", "berätta", "beskriv", "förklara", "jämför", "finns",
    }
)  # fmt: skip
_WORD = re.compile(r"[a-zåäöæø]+")


def is_swedish(question: str) -> bool:
    """Whether ``question`` is recognisably Swedish.

    ä and ö are Swedish letters (Norwegian and Danish write æ and ø), and a few
    common Swedish words are enough for short questions such as "Vem var
    Hannibal?". Anything else gets the general same-language rule.
    """
    text = question.lower()
    return "ä" in text or "ö" in text or any(w in _SWEDISH_WORDS for w in _WORD.findall(text))


def system_prompt(question: str, length: AnswerLength = "short") -> str:
    """The instructions, with the rules for how long and in which language to answer."""
    language_rule = SWEDISH_RULE if is_swedish(question) else SAME_LANGUAGE_RULE
    return f"{SYSTEM_PROMPT} {LENGTH_RULES[length]} {language_rule}"


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
    # The rewritten query the passages were found with; None when the question
    # itself was searched for.
    search_query: str | None = None

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


def build_messages(
    question: str, passages: Sequence[SearchResult], length: AnswerLength = "short"
) -> list[Message]:
    """The prompt: instructions, then the numbered passages and the question.

    Each passage is headed as it was embedded, with the article's other names,
    so the model knows that a passage about Blackbeard answers a question
    about Svartskägg (0011).
    """
    sources = "\n\n".join(
        f"[{number}] {result.chunk.embedding_text}"
        for number, result in enumerate(passages, start=1)
    )
    return [
        Message("system", system_prompt(question, length)),
        Message("user", f"Sources:\n\n{sources}\n\nQuestion: {question}"),
    ]


def should_rewrite(question: str, mode: RewriteMode) -> bool:
    """Whether ``mode`` asks for ``question`` to be rewritten before searching."""
    if mode == "always":
        return True
    if mode == "swedish":
        return is_swedish(question)
    return False


def rewrite_query(question: str, llm: ChatModel) -> str:
    """An English search query for ``question``, from the model.

    Small models sometimes add quotes, a "Query:" label or a second line; those
    are removed. If nothing usable is left, the question itself is returned.
    """
    messages = [Message("system", REWRITE_PROMPT)]
    for example, keywords in REWRITE_EXAMPLES:
        messages += [Message("user", example), Message("assistant", keywords)]
    messages.append(Message("user", question))
    reply = "".join(llm.stream_chat(messages, temperature=0.0))
    first_line = next((line for line in reply.splitlines() if line.strip()), "")
    query = _QUERY_LABEL.sub("", first_line.strip()).strip(" \"'`“”").strip()
    return query[:MAX_QUERY_CHARS] or question


def with_titles(question: str, titles_by_name: Mapping[str, Sequence[str]]) -> str:
    """``question`` with the article title after each other name it uses.

    "Hur dog Svartskägg?" becomes "Hur dog Svartskägg (Blackbeard)?", so the
    rewrite does not have to know the name, and a wrong translation such as
    "Black Death" cannot hide the article (0011). Names match as whole words
    in any case, longest name first. A name Wikidata writes with a capital
    letter also matches with a genitive s ("Svartskäggs"); other names do
    not, as an s there is a plural: "corsairs" is not the privateer "corsair".
    A title the question already contains is not added again.
    """
    if not titles_by_name:
        return question
    by_lower = {name.lower(): titles for name, titles in titles_by_name.items()}
    proper = {name.lower() for name in titles_by_name if name[:1].isupper()}
    names = sorted(by_lower, key=len, reverse=True)
    pattern = re.compile(
        r"(?<!\w)(" + "|".join(map(re.escape, names)) + r")s?(?!\w)", re.IGNORECASE
    )
    lowered = question.lower()

    def add_titles(match: re.Match[str]) -> str:
        name = match.group(1).lower()
        if match.group(0) != match.group(1) and name not in proper:
            return match.group(0)
        titles = [t for t in by_lower[name] if t.lower() not in lowered]
        return f"{match.group(0)} ({', '.join(titles)})" if titles else match.group(0)

    return pattern.sub(add_titles, question)


def search(
    question: str,
    search_query: str | None,
    *,
    store: VectorStore,
    embedder: Embedder,
    k: int = DEFAULT_K,
) -> list[SearchResult]:
    """The ``k`` best passages for the question and for the rewritten query together.

    The rewritten query finds passages by their English words; the question
    itself finds them by the other names stored with them, such as
    "Svartskägg", which the rewrite may translate wrongly (0011).

    The two lists are merged by place, not by score: a wrong rewrite such as
    "Black Death" can score higher on its own passages than the question does
    on the right one, and would push it out. Each passage counts
    1 / (RANK_OFFSET + place) per list it is in; equal sums keep the
    question's passages first. A passage keeps its better score, which the
    minimum score and the page compare.
    """
    queries = [question] if search_query is None else [question, search_query]
    best: dict[str, SearchResult] = {}
    fused: dict[str, float] = {}
    for query in queries:
        for place, result in enumerate(store.search(embedder.embed_query(query), k), start=1):
            chunk_id = result.chunk.chunk_id
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1 / (RANK_OFFSET + place)
            known = best.get(chunk_id)
            if known is None or result.score > known.score:
                best[chunk_id] = result
    order = sorted(best, key=lambda chunk_id: -fused[chunk_id])
    return [best[chunk_id] for chunk_id in order[:k]]


def ask(
    question: str,
    *,
    store: VectorStore,
    embedder: Embedder,
    llm: ChatModel,
    k: int = DEFAULT_K,
    rewrite: RewriteMode = "never",
    length: AnswerLength = "short",
    min_score: float = 0.0,
    on_text: Callable[[str], None] | None = None,
) -> Answer:
    """Retrieve the ``k`` best passages and have ``llm`` answer from them.

    Other names from the index are first followed by their article titles.
    With ``rewrite``, the passages are also searched for with an English query
    the model writes first; the answer is still written for the original question.
    When no passage scores at least ``min_score``, the question is declined
    without the model; the passages are kept, so their scores can be shown.
    ``length`` asks for a short answer or a detailed one of a paragraph or two.
    ``on_text`` receives the answer piece by piece while it is written, so a
    caller can show it immediately instead of after the whole minute.

    Raises:
        ValueError: the question is blank.
        LLMError: the model could not answer (see ragnosis.llm).
    """
    question = question.strip()
    if not question:
        raise ValueError("The question must not be blank")

    named = with_titles(question, store.titles_by_name())
    if named != question:
        logger.info("Known names: %r", named)
    search_query = rewrite_query(named, llm) if should_rewrite(question, rewrite) else None
    if search_query is not None:
        logger.info("Searching for %r", search_query)
    passages = search(named, search_query, store=store, embedder=embedder, k=k)
    logger.info(
        "Answering from: %s", ", ".join(r.chunk.article_title for r in passages) or "nothing"
    )
    if not passages:
        if on_text:
            on_text(NO_PASSAGES_TEXT)
        return Answer(question, NO_PASSAGES_TEXT, [], search_query)

    best = max(result.score for result in passages)
    if best < min_score:
        logger.info("Declining: the best passage scores %.3f, below %.3f", best, min_score)
        text = NOT_COVERED_TEXT_SV if is_swedish(question) else NOT_COVERED_TEXT
        if on_text:
            on_text(text)
        return Answer(question, text, passages, search_query)

    pieces: list[str] = []
    for piece in llm.stream_chat(
        build_messages(question, passages, length), temperature=ANSWER_TEMPERATURE
    ):
        pieces.append(piece)
        if on_text:
            on_text(piece)
    return Answer(question, "".join(pieces), passages, search_query)

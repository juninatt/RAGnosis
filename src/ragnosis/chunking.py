"""Split articles into passages ("chunks") small enough to embed and retrieve.

A question is usually answered by a paragraph, not a whole article. Embedding
whole articles would blur many topics into one vector, and the language model
can only read a limited amount of retrieved text. How the split is done (see
docs/decisions/0004):

- follow the article's own sections, so a chunk never mixes two topics,
- pack whole paragraphs, and only break a paragraph (at sentences, or at words
  as a last resort) when it is too long on its own,
- let neighbouring chunks overlap a little, so a fact on a boundary survives,
- drop sections that only list sources and links.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragnosis.wikipedia import Article

DEFAULT_MAX_WORDS = 200
DEFAULT_OVERLAP_WORDS = 40
INTRODUCTION = "Introduction"
SECTION_SEPARATOR = " > "

# Sections that list sources or links. Their text is noise for retrieval, and
# so is everything nested under them (e.g. "References > Citations").
SKIPPED_SECTIONS = frozenset(
    {
        "bibliography",
        "citations",
        "external links",
        "footnotes",
        "further reading",
        "notes",
        "notes and references",
        "references",
        "see also",
        "sources",
        "works cited",
    }
)

_HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")
# A sentence ends at . ! or ? followed by whitespace. Abbreviations ("c. 1680")
# cause extra breaks, which is harmless: pieces are packed back together anyway.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Section:
    """A run of article text under one heading path, e.g. "Legacy > In fiction"."""

    path: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """A passage of an article, with what is needed to cite it."""

    chunk_id: str
    page_id: int
    article_title: str
    section: str
    position: int
    text: str
    url: str

    @property
    def embedding_text(self) -> str:
        """The passage prefixed with where it comes from.

        "He was killed in 1718." says nothing about who "he" is; the article
        title and section give the embedding that missing context.
        """
        return f"{self.article_title}{SECTION_SEPARATOR}{self.section}\n{self.text}"


def split_sections(text: str) -> list[Section]:
    """Split TextExtracts text at ``== Heading ==`` lines.

    Text before the first heading becomes the "Introduction" section. Sections
    without text, and skipped sections with everything below them, are left out.
    """
    sections: list[Section] = []
    headings: list[tuple[int, str]] = []
    path = INTRODUCTION
    skipped = False
    lines: list[str] = []

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body and not skipped:
            sections.append(Section(path, body))
        lines.clear()

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match is None:
            lines.append(line)
            continue
        flush()
        level, title = len(match.group(1)), match.group(2)
        while headings and headings[-1][0] >= level:
            headings.pop()
        headings.append((level, title))
        path = SECTION_SEPARATOR.join(name for _, name in headings)
        skipped = any(name.lower() in SKIPPED_SECTIONS for _, name in headings)
    flush()
    return sections


def chunk_article(
    article: Article,
    *,
    max_words: int = DEFAULT_MAX_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[Chunk]:
    """Split ``article`` into chunks of at most ``max_words`` words.

    Words are counted by whitespace. That only approximates the embedding
    model's tokens, so ``max_words`` leaves a wide margin below its limit.
    Chunk ids are ``<page_id>-<position>``, stable as long as the text is.
    """
    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be at least 0 and less than max_words")

    chunks: list[Chunk] = []
    for section in split_sections(article.text):
        pieces = [
            piece
            for paragraph in _paragraphs(section.text)
            for piece in _split_long(paragraph, max_words)
        ]
        for text in _pack(pieces, max_words, overlap_words):
            position = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{article.page_id}-{position}",
                    page_id=article.page_id,
                    article_title=article.title,
                    section=section.path,
                    position=position,
                    text=text,
                    url=article.url,
                )
            )
    return chunks


def _paragraphs(text: str) -> list[str]:
    """Non-empty lines, with runs of whitespace collapsed to single spaces."""
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


def _split_long(paragraph: str, max_words: int) -> list[str]:
    """Return the paragraph whole if it fits, else its sentences, else word windows."""
    if len(paragraph.split()) <= max_words:
        return [paragraph]
    pieces: list[str] = []
    for sentence in _SENTENCE_BREAK.split(paragraph):
        words = sentence.split()
        if len(words) <= max_words:
            pieces.append(sentence)
        else:
            pieces.extend(
                " ".join(words[start : start + max_words])
                for start in range(0, len(words), max_words)
            )
    return pieces


def _pack(pieces: list[str], max_words: int, overlap_words: int) -> list[str]:
    """Greedily join pieces into chunks, starting each new chunk with the tail of the last.

    The overlap shrinks when needed so that no chunk exceeds ``max_words``.
    Every piece is at most ``max_words`` long, which ``_split_long`` guarantees.
    """
    chunks: list[list[str]] = []
    current: list[str] = []
    for piece in pieces:
        words = piece.split()
        if current and len(current) + len(words) > max_words:
            chunks.append(current)
            keep = min(overlap_words, max_words - len(words))
            current = current[-keep:] if keep > 0 else []
        current = current + words
    if current:
        chunks.append(current)
    return [" ".join(words) for words in chunks]

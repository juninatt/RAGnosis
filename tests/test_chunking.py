"""Tests for splitting articles into sections and chunks.

Example-based tests pin down specific behaviour; the property-based tests at the
end check rules that must hold for any text and any size settings.
"""

from __future__ import annotations

import pytest
from factories import make_article
from hypothesis import given
from hypothesis import strategies as st

from ragnosis.chunking import (
    DEFAULT_MAX_WORDS,
    INTRODUCTION,
    Chunk,
    Section,
    _pack,
    chunk_article,
    split_sections,
)


def words(count: int, prefix: str = "w") -> str:
    """``count`` distinct words, so positions in the output can be traced back."""
    return " ".join(f"{prefix}{i}" for i in range(count))


# --- split_sections ---


def test_text_without_headings_is_one_introduction_section() -> None:
    assert split_sections("First paragraph.\nSecond paragraph.") == [
        Section(INTRODUCTION, "First paragraph.\nSecond paragraph.")
    ]


def test_headings_start_new_sections() -> None:
    text = "Lead.\n\n\n== Early life ==\nBorn in Bristol.\n\n== Death ==\nKilled in 1718."
    assert split_sections(text) == [
        Section(INTRODUCTION, "Lead."),
        Section("Early life", "Born in Bristol."),
        Section("Death", "Killed in 1718."),
    ]


def test_nested_headings_form_a_path_and_unwind() -> None:
    text = (
        "== Legacy ==\nIntro to legacy.\n"
        "=== In fiction ===\nNovels.\n"
        "==== Film ====\nMovies.\n"
        "=== In music ===\nSongs.\n"
        "== Death ==\nKilled."
    )
    assert [s.path for s in split_sections(text)] == [
        "Legacy",
        "Legacy > In fiction",
        "Legacy > In fiction > Film",
        "Legacy > In music",
        "Death",
    ]


@pytest.mark.parametrize("heading", ["References", "See also", "EXTERNAL LINKS", "Notes"])
def test_source_and_link_sections_are_skipped(heading: str) -> None:
    text = f"Lead.\n== {heading} ==\nSmith 2001, p. 4."
    assert [s.path for s in split_sections(text)] == [INTRODUCTION]


def test_subsections_of_skipped_sections_are_skipped() -> None:
    text = "Lead.\n== References ==\n=== Citations ===\nSmith 2001.\n=== Sources ===\nJones."
    assert [s.path for s in split_sections(text)] == [INTRODUCTION]


def test_section_after_a_skipped_one_is_kept() -> None:
    text = "== Notes ==\nSee Smith.\n== Legacy ==\nRemembered."
    assert split_sections(text) == [Section("Legacy", "Remembered.")]


def test_skipping_matches_whole_heading_only() -> None:
    text = "== Notes on the crew ==\nThe crew was large."
    assert [s.path for s in split_sections(text)] == ["Notes on the crew"]


def test_sections_without_text_are_dropped() -> None:
    text = "== Biography ==\n\n\n=== Early life ===\nBorn.\n== Empty ==\n\n"
    assert [s.path for s in split_sections(text)] == ["Biography > Early life"]


@pytest.mark.parametrize("line", ["==Death==", "==  Death  ==", "== Death == "])
def test_heading_spacing_variants(line: str) -> None:
    assert split_sections(f"{line}\nKilled.") == [Section("Death", "Killed.")]


@pytest.mark.parametrize("line", ["a == b == c", "= Single =", "== No closing"])
def test_lines_that_are_not_headings_stay_text(line: str) -> None:
    assert split_sections(line) == [Section(INTRODUCTION, line)]


def test_empty_text_has_no_sections() -> None:
    assert split_sections("") == []


# --- chunk_article ---


def test_short_article_becomes_one_chunk_with_metadata() -> None:
    article = make_article(page_id=42, title="Anne Bonny", text="Anne Bonny was a pirate.")
    assert chunk_article(article) == [
        Chunk(
            chunk_id="42-0",
            page_id=42,
            article_title="Anne Bonny",
            section=INTRODUCTION,
            position=0,
            text="Anne Bonny was a pirate.",
            url=article.url,
        )
    ]


def test_positions_and_ids_run_across_sections() -> None:
    article = make_article(page_id=7, text="Lead.\n== A ==\nOne.\n== B ==\nTwo.")
    chunks = chunk_article(article)
    assert [(c.chunk_id, c.position, c.section) for c in chunks] == [
        ("7-0", 0, INTRODUCTION),
        ("7-1", 1, "A"),
        ("7-2", 2, "B"),
    ]


def test_small_paragraphs_in_a_section_are_packed_together() -> None:
    article = make_article(text="First.\nSecond.\n\nThird.")
    assert [c.text for c in chunk_article(article)] == ["First. Second. Third."]


def test_chunks_never_cross_section_boundaries() -> None:
    article = make_article(text=f"{words(5, 'a')}\n== Next ==\n{words(5, 'b')}")
    chunks = chunk_article(article, max_words=50, overlap_words=10)
    assert [c.text for c in chunks] == [words(5, "a"), words(5, "b")]


def test_paragraphs_are_kept_whole_when_they_fit() -> None:
    first, second = words(6, "a"), words(6, "b")
    article = make_article(text=f"{first}\n{second}")
    chunks = chunk_article(article, max_words=10, overlap_words=0)
    assert [c.text for c in chunks] == [first, second]


def test_overlap_repeats_the_tail_of_the_previous_chunk() -> None:
    article = make_article(text=f"{words(8, 'a')}\n{words(8, 'b')}")
    chunks = chunk_article(article, max_words=10, overlap_words=2)
    assert [c.text for c in chunks] == [words(8, "a"), "a6 a7 " + words(8, "b")]


def test_overlap_shrinks_so_the_limit_holds() -> None:
    article = make_article(text=f"{words(8, 'a')}\n{words(9, 'b')}")
    chunks = chunk_article(article, max_words=10, overlap_words=5)
    assert [c.text for c in chunks] == [words(8, "a"), "a7 " + words(9, "b")]


def test_long_paragraph_is_split_at_sentence_ends() -> None:
    paragraph = "One two three four. Five six seven eight! Nine ten eleven twelve?"
    chunks = chunk_article(make_article(text=paragraph), max_words=8, overlap_words=0)
    assert [c.text for c in chunks] == [
        "One two three four. Five six seven eight!",
        "Nine ten eleven twelve?",
    ]


def test_overlong_sentence_is_split_into_word_windows() -> None:
    chunks = chunk_article(make_article(text=words(25)), max_words=10, overlap_words=0)
    assert [len(c.text.split()) for c in chunks] == [10, 10, 5]


def test_whitespace_inside_paragraphs_is_collapsed() -> None:
    article = make_article(text="Teach  was\tfeared.   He  died.")
    assert chunk_article(article)[0].text == "Teach was feared. He died."


def test_article_with_only_skipped_sections_has_no_chunks() -> None:
    article = make_article(text="== References ==\nSmith.\n== External links ==\nsite")
    assert chunk_article(article) == []


def test_embedding_text_names_article_and_section() -> None:
    article = make_article(title="Blackbeard", text="== Death ==\nHe was killed in 1718.")
    (chunk,) = chunk_article(article)
    assert chunk.embedding_text == "Blackbeard > Death\nHe was killed in 1718."


@pytest.mark.parametrize(
    ("max_words", "overlap_words"),
    [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)],
)
def test_invalid_sizes_raise(max_words: int, overlap_words: int) -> None:
    with pytest.raises(ValueError, match="must be"):
        chunk_article(make_article(), max_words=max_words, overlap_words=overlap_words)


def test_packing_nothing_gives_no_chunks() -> None:
    # Unreachable through chunk_article (empty sections are dropped earlier), but
    # the packer must not invent an empty chunk if that ever changes.
    assert _pack([], max_words=10, overlap_words=2) == []


def test_defaults_keep_typical_paragraphs_whole() -> None:
    paragraph = words(150)
    (chunk,) = chunk_article(make_article(text=paragraph))
    assert chunk.text == paragraph
    assert DEFAULT_MAX_WORDS >= 150


# --- Properties that must hold for any input ---

# Sections of paragraphs of sentences, built from distinct words ("w0", "w1", ...)
# so the original word order can be reconstructed from the chunks.
sentence_lengths = st.lists(st.integers(1, 40), min_size=1, max_size=6)
paragraph_shapes = st.lists(sentence_lengths, min_size=1, max_size=5)
section_shapes = st.lists(paragraph_shapes, min_size=1, max_size=4)
sizes = st.integers(1, 60).flatmap(lambda m: st.tuples(st.just(m), st.integers(0, m - 1)))


def build_text(shape: list[list[list[int]]]) -> tuple[str, list[list[str]]]:
    """Article text for ``shape`` plus the expected words of each section."""
    counter = 0
    lines: list[str] = []
    expected: list[list[str]] = []
    for index, paragraphs in enumerate(shape):
        lines.append(f"== Section {index} ==")
        section_words: list[str] = []
        for sentences in paragraphs:
            parts = []
            for length in sentences:
                sentence = [f"w{counter + i}" for i in range(length)]
                counter += length
                section_words.extend(sentence)
                parts.append(" ".join(sentence) + ".")
            lines.append(" ".join(parts))
        expected.append(section_words)
    return "\n".join(lines), expected


def strip_overlap(chunks: list[list[str]], overlap_words: int) -> list[str]:
    """Rejoin chunk words, removing the repeated tail at the start of each chunk."""
    result = list(chunks[0])
    for previous, current in zip(chunks, chunks[1:], strict=False):
        repeated = next(
            k
            for k in range(min(overlap_words, len(previous), len(current)), -1, -1)
            if k == 0 or previous[-k:] == current[:k]
        )
        result.extend(current[repeated:])
    return result


def plain(text: str) -> list[str]:
    return text.replace(".", "").split()


@given(section_shapes, sizes)
def test_no_chunk_exceeds_max_words(shape: list[list[list[int]]], size: tuple[int, int]) -> None:
    max_words, overlap_words = size
    text, _ = build_text(shape)
    chunks = chunk_article(
        make_article(text=text), max_words=max_words, overlap_words=overlap_words
    )
    assert all(1 <= len(c.text.split()) <= max_words for c in chunks)


@given(section_shapes, sizes)
def test_every_word_is_kept_in_order(shape: list[list[list[int]]], size: tuple[int, int]) -> None:
    max_words, overlap_words = size
    text, expected = build_text(shape)
    chunks = chunk_article(
        make_article(text=text), max_words=max_words, overlap_words=overlap_words
    )

    for index, section_words in enumerate(expected):
        in_section = [plain(c.text) for c in chunks if c.section == f"Section {index}"]
        assert strip_overlap(in_section, overlap_words) == section_words


@given(section_shapes, sizes)
def test_overlap_never_exceeds_setting(shape: list[list[list[int]]], size: tuple[int, int]) -> None:
    max_words, overlap_words = size
    text, _ = build_text(shape)
    chunks = chunk_article(
        make_article(text=text), max_words=max_words, overlap_words=overlap_words
    )
    for previous, current in zip(chunks, chunks[1:], strict=False):
        shared = set(plain(previous.text)) & set(plain(current.text))
        assert len(shared) <= overlap_words


@given(section_shapes, sizes)
def test_chunking_is_deterministic(shape: list[list[list[int]]], size: tuple[int, int]) -> None:
    max_words, overlap_words = size
    article = make_article(text=build_text(shape)[0])
    first = chunk_article(article, max_words=max_words, overlap_words=overlap_words)
    assert first == chunk_article(article, max_words=max_words, overlap_words=overlap_words)
    assert [c.position for c in first] == list(range(len(first)))

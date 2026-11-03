"""Turn the tables, infoboxes and image captions of an article's HTML into text.

Wikipedia's text extracts leave these out, although they hold facts the text
does not repeat, such as how many ships each city sent to Salamis (decision
0012). Each becomes a part with the heading path it stood under, so it can be
split into passages like the rest of the article:

- the infobox, one line per fact: "Born: c. 1680, Bristol",
- each table of Wikipedia's standard style, one line per row:
  "City: Athens; Number of ships: 180.",
- the image captions of a section, one line per caption.

Navigation boxes, footnote markers and edit links are left out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag

INTRODUCTION = "Introduction"
SECTION_SEPARATOR = " > "
FACTS = "Facts"
TABLE = "Table"
IMAGES = "Images"
# Shorter captions, such as "Blackbeard, 1736", add little to search for.
MIN_CAPTION_WORDS = 5

_HEADING_TAGS = ("h2", "h3", "h4", "h5", "h6")
# Footnote markers ("[61]"), edit links, content hidden from readers and the
# second, decimal form of coordinates.
_NOISE = (
    "sup.reference, span.mw-editsection, style, .noprint, .mw-empty-elt, "
    ".geo-nondefault, .geo-multi-punct"
)
_SPACE_BEFORE = re.compile(r"\s+([,.;:!?)\]])")
_SPACE_AFTER = re.compile(r"([(\[])\s+")
_SEMICOLONS = re.compile(r"(\s*;)+\s*")


@dataclass(frozen=True)
class Part:
    """Text from the article's HTML, with the heading path it stood under."""

    path: str
    text: str


def extract_parts(html: str) -> list[Part]:
    """The infobox, tables and image captions of ``html``, in the order they appear.

    Captions of one section are joined into one part. Tables and captions
    inside another table, such as an infobox's image, belong to that table.
    """
    soup = BeautifulSoup(html, "html.parser")
    for noise in soup.select(_NOISE):
        noise.decompose()
    # Line breaks and list items separate values, as in "Athens; Sparta".
    for tag in soup.find_all(["br", "li"]):
        tag.insert_after("; ")

    parts: list[Part] = []
    headings: list[tuple[int, str]] = []
    captions: list[str] = []
    caption_path = INTRODUCTION

    def flush_captions() -> None:
        if captions:
            parts.append(Part(_under(caption_path, IMAGES), "\n".join(captions)))
            captions.clear()

    for element in soup.find_all(_is_wanted):
        if element.find_parent("table") is not None:
            continue
        if element.name in _HEADING_TAGS:
            level = int(element.name[1])
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, _text(element)))
            continue
        path = SECTION_SEPARATOR.join(name for _, name in headings) or INTRODUCTION
        if element.name == "figcaption":
            caption = _text(element)
            if len(caption.split()) >= MIN_CAPTION_WORDS:
                if path != caption_path:
                    flush_captions()
                    caption_path = path
                captions.append(caption)
            continue
        flush_captions()
        if "infobox" in element.get_attribute_list("class"):
            lines = _infobox_lines(element)
            label = FACTS
        else:
            lines = _table_lines(element)
            label = TABLE
        if lines:
            parts.append(Part(_under(path, label), "\n".join(lines)))
    flush_captions()
    return parts


def _is_wanted(tag: Tag) -> bool:
    if tag.name in _HEADING_TAGS or tag.name == "figcaption":
        return True
    classes = tag.get_attribute_list("class")
    return tag.name == "table" and ("infobox" in classes or "wikitable" in classes)


def _under(path: str, label: str) -> str:
    return label if path == INTRODUCTION else f"{path}{SECTION_SEPARATOR}{label}"


def _text(tag: Tag) -> str:
    """The visible text, on one line, without the spaces markup leaves before commas."""
    text = " ".join(tag.get_text(" ").replace("\ufeff", " ").split())
    text = _SPACE_AFTER.sub(r"\1", _SPACE_BEFORE.sub(r"\1", text))
    return _SEMICOLONS.sub("; ", text).replace(":;", ":").strip(" ;")


def _infobox_lines(table: Tag) -> list[str]:
    """One line per fact. A heading row, such as "Belligerents", names the rows below it."""
    lines: list[str] = []
    heading = ""
    for row in _rows(table):
        labels = row.find_all("th", recursive=False)
        cells = row.find_all("td", recursive=False)
        nested = [table for cell in cells for table in cell.find_all("table", limit=1)]
        if nested and not labels:
            # Battle infoboxes put date, place and result in a table of their own.
            lines += [line for inner in nested for line in _infobox_lines(inner)]
            continue
        values = [_text(cell) for cell in cells]
        values = [value for value in values if value]
        if labels and values:
            label = _text(labels[0]).replace("; ", " ")
            lines.append(f"{label}: {'; '.join(values)}")
        elif labels:
            heading = _text(labels[0])
        elif values:
            lines.append(f"{heading}: {' | '.join(values)}" if heading else " | ".join(values))
    return [line for line in lines if line.strip(" :|")]


def _table_lines(table: Tag) -> list[str]:
    """The caption, then one line per row, each value after its column's heading."""
    grid = _grid(table)
    header: list[str] = []
    if grid and all(is_heading for is_heading, _ in grid[0]):
        header = [text for _, text in grid.pop(0)]
    caption = table.find("caption")
    lines = [_text(caption)] if caption is not None and _text(caption) else []
    for row in grid:
        cells = [text for _, text in row]
        if len(header) == len(cells):
            pairs = [
                f"{name}: {value}" if name else value
                for name, value in zip(header, cells, strict=True)
            ]
            line = "; ".join(pair for pair, value in zip(pairs, cells, strict=True) if value)
        else:
            line = "; ".join(cell for cell in cells if cell)
        if line:
            lines.append(line + ".")
    return lines


def _grid(table: Tag) -> list[list[tuple[bool, str]]]:
    """The table's cells row by row, with merged cells repeated where they span.

    Each cell is (whether it is a heading, its text).
    """
    grid: list[list[tuple[bool, str]]] = []
    pending: dict[int, tuple[int, tuple[bool, str]]] = {}  # column -> (rows left, cell)
    for row in _rows(table):
        tags = row.find_all(["th", "td"], recursive=False)
        cells: list[tuple[bool, str]] = []
        while tags or len(cells) in pending:
            column = len(cells)
            if column in pending:
                left, cell = pending.pop(column)
                if left > 1:
                    pending[column] = (left - 1, cell)
                cells.append(cell)
                continue
            tag = tags.pop(0)
            cell = (tag.name == "th", _text(tag))
            rows = _span(tag, "rowspan")
            for offset in range(_span(tag, "colspan")):
                if rows > 1:
                    pending[column + offset] = (rows - 1, cell)
                cells.append(cell)
        if cells:
            grid.append(cells)
    return grid


def _rows(table: Tag) -> list[Tag]:
    """The table's own rows, not those of tables nested in it."""
    return [row for row in table.find_all("tr") if row.find_parent("table") is table]


_NUMBER = re.compile(r"\d+")


def _span(cell: Tag, attribute: str) -> int:
    """How many columns or rows a cell spans; 1 when unset or unreadable."""
    match = _NUMBER.match(str(cell.get(attribute) or "1"))
    return max(1, min(int(match.group()), 50)) if match else 1

# 0012: Tables, infoboxes and image captions

**Status:** accepted

## Context

Wikipedia's text extracts (0003) give an article's running text only. Tables,
the infobox and image captions are left out, although they hold facts the
text does not repeat. The article on the Battle of Salamis lists the ships
each Greek city sent in a table; its text keeps only the table's note, "Plain
numbers represent triremes", so a question about Athens' ships could not be
answered.

## Decision

- **`ingest` also fetches each article's HTML,** for the same revision as its
  text, so the two always match. Articles stored before get it on the next
  `ingest`, as with the other names (0011).
- **Three kinds of content are taken from it, as text** (BeautifulSoup reads
  the HTML):
  - the infobox, one line per fact, under `Facts`:
    `Born: Edward Teach; c. 1680; Possibly Bristol, England`,
  - each table of Wikipedia's standard style (`wikitable`), one line per row
    with each value after its column's heading, under the section it stood in:
    `Opposing forces > Greek fleet > Table` with
    `City: Athens; Number of ships: 180.`,
  - image captions of at least 5 words, joined per section under `Images`.
    Shorter ones, such as "Blackbeard, 1736", add little.

  Navigation boxes, footnote markers and edit links are left out.
- **The text extract stays the source of the running text.** The new content
  is added as sections of its own after the text, so the passages from the
  text, and their ids, do not change.
- **`index` embeds an article again when its passages change,** not only when
  its revision or other names do. Comparing the passages catches any change to
  what is embedded. The index format stays 2: `ingest` and then `index` are
  enough after upgrading.

## Consequences

- Facts that are only in a table or an infobox can be found and cited. Both
  new questions that only a table answers passed (see `docs/evaluation.md`).
- The index grew by 22 %, from 3,354 to 4,095 passages, most of them short
  image captions. The text grew by 6 %.
- `ingest` makes one more request per article; for the 64 articles it took
  under a minute.
- A table whose first row is not all headings, or whose layout is unusual,
  becomes rows of plain values without their headings.
- One more dependency, BeautifulSoup, in pure Python.

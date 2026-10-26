# 0004: Chunking articles

**Status:** accepted

## Context

Articles are long (median about 6,500 words in the first corpus), while a
question is usually answered by a single paragraph. Retrieval works on passages
("chunks"), so each article has to be split. Splits that are too big blur
several topics into one embedding; splits that are too small lose context. The
planned embedding model reads at most 512 tokens.

## Decision

- **Follow the article's sections.** A chunk never spans two sections, so it
  stays on one topic, and the section path ("Legacy > In fiction") is kept as
  metadata for citations.
- **Pack whole paragraphs** up to 200 words. A paragraph is only broken when it
  is too long on its own: at sentence ends first, at word boundaries last.
- **Overlap of up to 40 words** between neighbouring chunks of a section, so a
  fact on a boundary is found from either side.
- **Embed with a header.** The text that gets embedded starts with
  "Article > Section", because passages often refer to "he" or "the war"
  without naming them.
- **Skip source and link sections** (References, See also, External links and
  similar) along with their subsections. They are lists, not prose.
- **Count words, not tokens.** No tokenizer dependency is needed, and
  200 words plus the header stays well below 512 tokens.

## Consequences

- On the first corpus: 3,355 chunks, median 157 words, very few tiny ones.
- Chunk ids (`<page_id>-<position>`) shift when an article's text changes, so an
  article is re-indexed as a whole rather than chunk by chunk.
- Sizes are parameters and can be tuned later against retrieval tests.

# 0003: Ingesting Wikipedia articles

**Status:** accepted

## Context

The corpus needs a defined set of articles, fetched as clean text, stored so
later steps can be rerun without downloading again. Wikipedia categories were
considered for choosing articles, but they nest deeply and mix in loosely related
pages, which would make the corpus large, noisy and different from run to run.

## Decision

- **A curated title list** (`topics/*.txt`) defines the corpus. It is committed,
  so every change to the corpus shows up in a diff.
- **Plain text from the TextExtracts API.** It returns article text without
  markup, so no parser is needed, and keeps section headings for later splitting.
- **Disambiguation and missing pages count as failures** and are reported, so a
  bad title is fixed in the list rather than silently indexed.
- **One JSON file per article** under `data/articles/`, keyed by the requested
  title. A rerun skips titles already stored; `--refresh` fetches them again.
- **A polite client:** a User-Agent naming the project, one request at a time,
  and exponential backoff on rate limits and server errors.

## Consequences

- Growing the corpus means editing the list. Category-based discovery can be
  added later as a tool that proposes titles for the list.
- Stored articles do not update by themselves; `--refresh` brings in newer
  revisions, and each file records its `revision_id` and `fetched_at`.
- Tables, infoboxes and image captions are not in the extracts, so facts that
  only appear there are missing. Wikidata is the planned source for those.

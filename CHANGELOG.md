# Changelog

Notable changes per release. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). From 1.0.0, what stays compatible
is set by [decision 0009](docs/decisions/0009-what-1-0-promises.md); before
it, commands, settings and the index format changed between minor versions.

## [1.3.0] - 2026-11-05

Facts that are only in a table, an infobox or an image caption can be found.

### Added

- `ingest` fetches each article's tables, infobox and image captions from the
  same revision as its text, and fills them in for articles already stored.
  They become passages of their own, such as `Opposing forces > Greek fleet >
  Table`. Decision 0012 explains how.
- Two known questions that only a table answers.

### Changed

- `index` embeds an article again whenever its passages change, not only
  when its revision or other names do. After upgrading, run `ingest` and then
  `index`; the index format is unchanged, so no rebuild is needed.
- BeautifulSoup is a new dependency, for reading the HTML.

### Known issues

- A table whose first row is not all headings loses its column headings.
- Names Wikidata does not have, and inflected forms such as "Svartskäggen",
  are not recognised.
- Swedish answers have small language errors, such as "løytnant".
- An answer takes 30 to 60 seconds on a laptop CPU.

## [1.2.0] - 2026-11-03

Questions that use another name, such as "Svartskägg" for Blackbeard, find
the article.

### Added

- `ingest` fetches each article's other names from Wikidata, such as
  translations, nicknames and full names, and fills them in for articles
  already stored. Decision 0011 explains how they are used.
- `RAGNOSIS_NAME_LANGUAGES` (default `en,sv`): the languages of those names.
- Known names in a question are followed by their article's title before the
  search, and the model sees the names with each source.
- `evaluate` fails an answer that says it cannot answer and then answers with
  sources, unless the case is marked `partial`: the sources answer only part
  of it, and refusing with context is right.
- Three known questions that use other names, and two the sources only
  partly answer.

### Changed

- The index stores the names with every passage (index format 2). An index
  built by an older version is refused until `index --rebuild` has run; run
  `ingest` first to fetch the names.
- Search uses both the question and the rewritten query, merged by place, so
  a wrong rewrite cannot push out the passages the question finds.

### Fixed

- "Hur dog Svartskägg?" is answered from the article on Blackbeard instead of
  passages about the Black Death and Rome.
- The scorecard no longer reads Swedish answers that quote English titles as
  English.
- A known question cited a title that Wikipedia redirects; cases now use the
  titles the redirects lead to, and a test rejects redirected ones.

### Known issues

- Names Wikidata does not have, and inflected forms such as "Svartskäggen",
  are not recognised.
- Wikipedia's text extracts leave out tables, so facts that are only in a
  table cannot be found.
- Swedish answers have small language errors, such as "osmanlig" for
  "osmansk".
- An answer takes 30 to 60 seconds on a laptop CPU.

## [1.1.0] - 2026-11-01

Questions outside the topics are declined without unrelated sources.

### Added

- `RAGNOSIS_MIN_SCORE` (default 0.83): when no passage scores this high, the
  question is declined without the model and without sources, in about 6
  seconds. Decision 0010 explains how the default was measured.
- `evaluate` shows the search score of the best passage for every question.
- Three known questions close to the edge of the topics.

### Changed

- The required level in `evaluation/answers.toml` is a share, `pass_rate`
  (85 %), instead of a number, so it holds as questions are added. The old
  `required` is no longer read.

### Fixed

- Answers to questions outside the topics no longer cite unrelated sources.
- Two known questions now accept the article on Calico Jack, which Wikipedia
  names John Rackham.

### Known issues

- Names translated into Swedish, such as "Svartskägg" for Blackbeard, are not
  found.
- A question just below the minimum score is declined although a passage
  covers it; the margin measured is small (0.811 against 0.845).
- Swedish answers have small language errors, such as "osmanlig" for
  "osmansk".
- An answer takes 30 to 60 seconds on a laptop CPU.

## [1.0.0] - 2026-10-28

First stable release: 0.6.0 passed the test period without new problems.

### Changed

- The commands, exit codes, settings, web API and data on disk are now
  stable within 1.x, as decision 0009 sets out.
- The description no longer mentions Wikidata, which is not used yet.

### Known issues

- Names translated into Swedish, such as "Svartskägg" for Blackbeard, are not
  found.
- An answer that declines may go on to cite unrelated sources.
- An answer takes 30 to 60 seconds on a laptop CPU.

## [0.6.0] - 2026-10-28

Release candidate for 1.0: what 1.0 promises, and a guide for new users.

### Added

- Decision 0009 records what stays compatible through 1.x: the commands and
  exit codes, the `RAGNOSIS_*` settings, the web API and the data on disk.
- `evaluate` passes when a full run reaches the level set in the cases file,
  14 of 16 known questions, and exits with `1` when it does not.
- An installation guide in the README, tried from a clean clone, with a
  troubleshooting table.
- A table of every setting with its default; a test keeps it in step with
  `.env` and the code.
- `docs/release-checks.md`: the checks before a release and a log of the
  manual ones.

### Known issues

- Names translated into Swedish, such as "Svartskägg" for Blackbeard, are not
  found.
- An answer that declines may go on to cite unrelated sources. Hiding them was
  tried and removed, as it also hid correct sources.

## [0.5.0] - 2026-10-28

Better answers: questions are rewritten for search, and answers are measured.

### Added

- `evaluate` command: answers 16 known questions in Swedish and English and
  checks each answer's sources, facts and language, and that it declines when
  it should. Results are kept in `docs/evaluation.md`.
- Short or detailed answers: a choice in the web UI, remembered in the
  browser, and `--detailed` for `ask` and `evaluate`.

### Changed

- Questions are rewritten into English search keywords before searching
  (`RAGNOSIS_QUERY_REWRITE`, default `always`): 15 of 16 known questions now
  pass, against 12 of 16 without rewriting. It adds about 6 seconds a question.
- Answers are reproducible: the same question and sources give the same
  answer (temperature 0 and a fixed seed).

### Known issues

- Names translated into Swedish, such as "Svartskägg" for Blackbeard, are not
  found; storing Swedish article names in the index is planned.
- A detailed answer that declines may go on to cite unrelated sources.

## [0.4.0] - 2026-10-26

Web UI: ask in the browser, with answers that start sooner.

### Added

- `serve` command: a local web UI where answers stream in and cited sources
  link to Wikipedia; localhost only, one question at a time.
- The web UI shows how long each answer took, in total and to the first words.

### Changed

- The language model stays loaded for 30 minutes after an answer
  (`RAGNOSIS_LLM_KEEP_ALIVE`) and is loaded when `serve` starts, so answers no
  longer wait for it.
- The model reads four passages instead of five, which shortens the wait
  before the first words without finding fewer relevant articles.

### Fixed

- Swedish questions are answered in Swedish; some were answered in Norwegian
  or English.

## [0.3.0] - 2026-10-26

Answers: questions are answered from the passages, with cited sources.

### Added

- `ask` command: answers from the best passages with a local model through
  Ollama (`qwen3.5:4b` by default), streamed as written, with cited sources.
- README note on running the test and type-check tools as modules when
  Windows Smart App Control blocks their launchers.

## [0.2.0] - 2026-10-26

Retrieval: questions find the passages that answer them.

### Added

- Splitting of articles into section-aware passages of up to 200 words.
- Local embeddings with `intfloat/multilingual-e5-small`; Swedish questions
  find English passages.
- A numpy vector store with exact cosine search.
- `index` command, which embeds only changed articles, and `search` command.
- Opt-in retrieval quality tests with known questions.

### Changed

- Build backend is `uv_build`, so the package builds with Windows Smart App
  Control on.
- The Wikipedia User-Agent carries the package version.

## 0.1.0 - 2026-10-23 (not tagged)

First working pipeline step: articles from Wikipedia.

### Added

- `ingest` command, which downloads a curated list of Wikipedia articles
  (pirates and antiquity in Europe) as plain text.
- Settings in env files, central logging, and CI on Linux and Windows.

[1.1.0]: https://github.com/juninatt/RAGnosis/releases/tag/v1.1.0
[1.0.0]: https://github.com/juninatt/RAGnosis/releases/tag/v1.0.0
[0.6.0]: https://github.com/juninatt/RAGnosis/releases/tag/v0.6.0
[0.5.0]: https://github.com/juninatt/RAGnosis/releases/tag/v0.5.0
[0.4.0]: https://github.com/juninatt/RAGnosis/releases/tag/v0.4.0
[0.3.0]: https://github.com/juninatt/RAGnosis/releases/tag/v0.3.0
[0.2.0]: https://github.com/juninatt/RAGnosis/releases/tag/v0.2.0

# Changelog

Notable changes per release. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/): until 1.0.0, commands, settings and
the index format may still change between minor versions.

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

[0.6.0]: https://github.com/juninatt/RAGnosis/releases/tag/v0.6.0
[0.5.0]: https://github.com/juninatt/RAGnosis/releases/tag/v0.5.0
[0.4.0]: https://github.com/juninatt/RAGnosis/releases/tag/v0.4.0
[0.3.0]: https://github.com/juninatt/RAGnosis/releases/tag/v0.3.0
[0.2.0]: https://github.com/juninatt/RAGnosis/releases/tag/v0.2.0

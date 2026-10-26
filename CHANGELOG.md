# Changelog

Notable changes per release. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/): until 1.0.0, commands, settings and
the index format may still change between minor versions.

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

[0.3.0]: https://github.com/juninatt/RAGnosis/releases/tag/v0.3.0
[0.2.0]: https://github.com/juninatt/RAGnosis/releases/tag/v0.2.0

# 0001: Free, local stack

**Status:** accepted

## Context

RAGnosis should cost nothing to run and work offline once the data is fetched.
The development machine has 16 GB RAM and integrated Intel Iris graphics, so
there is no CUDA GPU and models run on the CPU.

## Decision

| Concern | Choice | Why |
|---|---|---|
| Source data | Wikipedia API, later Wikidata SPARQL | Free and openly licensed. Wikipedia gives prose to retrieve from; Wikidata gives exact facts. |
| Embeddings | `sentence-transformers`, multilingual model | Runs locally and handles more than English. |
| Vector store | ChromaDB | Embedded and persists to a folder, so there is no server to run. |
| Language model | Ollama with a small (about 3–4B) model | Free and local; models of that size answer at usable speed on a CPU with 16 GB RAM. |
| Tooling | `uv`, `pytest`, `ruff`, `mypy` | One tool for Python versions and locked dependencies, plus fast checks that CI enforces. |

## Consequences

- Answers are slower and less fluent than from large hosted models. The model
  name is configuration, so switching to a larger or hosted model later is cheap.
- Retrieval quality matters more than model size, so it gets its own tests.
- Wikipedia text is CC BY-SA. Downloaded data stays out of the repository.

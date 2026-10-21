# RAGnosis

A free, local RAG (retrieval-augmented generation) assistant grounded in
Wikipedia and Wikidata.

Ask a question and RAGnosis finds the relevant Wikipedia passages, then has a
local language model answer from those passages, citing them. Grounding answers
in retrieved text makes them checkable and reduces made-up facts. Running
everything locally keeps it free and private.

The first dataset covers pirates and antiquity, with a focus on Europe.

> **Status:** early development. Only project scaffolding exists so far.

## How it works

1. **Ingest:** fetch articles and split them into passages.
2. **Index:** embed each passage and store it in a local vector database.
3. **Ask:** embed the question, retrieve the closest passages and let the
   language model answer from them.

The reasons behind the stack are in [docs/decisions](docs/decisions/).

## Getting started

Requires [uv](https://docs.astral.sh/uv/), which also installs the right Python version.

```bash
uv sync
uv run pytest
```

## Development

| Task | Command |
|---|---|
| Run unit tests | `uv run pytest` |
| Run with coverage | `uv run pytest --cov` |
| Run integration tests (real services) | `uv run pytest -m integration` |
| Lint | `uv run ruff check` |
| Format | `uv run ruff format` |
| Type check | `uv run mypy` |

CI runs all of these on every push and pull request.

### Conventions

- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/)
  with a scope, e.g. `feat(ingest): fetch articles from the Wikipedia API`.
- **Tests** come with every new feature. Unit tests mock network and models so
  they stay fast and deterministic, and coverage must stay at 90% or above.
- **Logging, not print:** modules use `logging.getLogger(__name__)`, and ruff
  rejects `print`. Set the verbosity with `RAGNOSIS_LOG_LEVEL` (default `INFO`).
- **Configuration:** non-sensitive defaults live in the committed `.env`.
  Secrets and personal overrides go in `.env.local`, which git ignores. Real
  environment variables override both. A test fails if `.env` gains a key that
  looks like a secret.
- **Decisions** that shape the project are recorded in `docs/decisions/`.

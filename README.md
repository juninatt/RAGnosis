# RAGnosis

A free, local RAG (retrieval-augmented generation) assistant grounded in
Wikipedia and Wikidata.

Ask a question and RAGnosis finds the relevant Wikipedia passages, then has a
local language model answer from those passages, citing them. Grounding answers
in retrieved text makes them checkable and reduces made-up facts. Running
everything locally keeps it free and private.

The first dataset covers pirates and antiquity, with a focus on Europe.

> **Status:** early development. Articles can be fetched; indexing and
> answering come next.

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

With Windows Smart App Control on, uv's own Python builds may be blocked
because they are unsigned. Installing Python 3.12 from python.org and setting
`python-preference = "system"` in your user `uv.toml` avoids that.

## Usage

### Fetch the articles

```bash
uv run python -m ragnosis ingest
```

`uv run ragnosis ingest` does the same, but Windows Smart App Control may block
the unsigned `ragnosis.exe` launcher; `python -m` works everywhere.

Downloads every title in the topics file (`RAGNOSIS_TOPICS_FILE`) from
Wikipedia and stores each article as JSON in `data/articles/`. Reruns only
fetch titles that are missing, so an interrupted run can simply be restarted.

| Option | Effect |
|---|---|
| `--topics FILE` | Use another title list. |
| `--refresh` | Fetch again even if already stored, to pick up newer revisions. |

Titles that do not exist or are disambiguation pages are reported at the end;
fix them in the topics file and rerun. Exit codes: `0` all fetched, `1` some
titles failed, `2` bad configuration or usage.

### Settings

All settings are listed with their defaults in [`.env`](.env).

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

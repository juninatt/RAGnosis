# RAGnosis

A free, local RAG (retrieval-augmented generation) assistant grounded in
Wikipedia and Wikidata.

Ask a question and RAGnosis finds the relevant Wikipedia passages, then has a
local language model answer from those passages, citing them. Grounding answers
in retrieved text makes them checkable and reduces made-up facts. Running
everything locally keeps it free and private.

The first dataset covers pirates and antiquity, with a focus on Europe.

> **Status:** early development. Questions are answered from the indexed
> articles, on the command line or in a local web UI.

## How it works

1. **Ingest:** fetch articles from Wikipedia and store them locally.
2. **Index:** split articles into passages, embed each passage and store the
   vectors in a local index.
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

### Build the search index

```bash
uv run python -m ragnosis index
```

Splits the downloaded articles into passages and embeds them with the model in
`RAGNOSIS_EMBEDDING_MODEL` (downloaded on first use, about 0.5 GB). Reruns only
embed articles that changed. `--rebuild` starts over, which is required after
changing the model. Exit code `1` means some titles have not been downloaded.
On Windows, the PyTorch library behind the model needs a current
[Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist).

### Search

```bash
uv run python -m ragnosis search "Vem var Hannibal?"
```

Prints the passages closest to the question, best first, with their source.
Questions can be in any language the model knows. `-k N` sets how many.

### Ask

```bash
uv run python -m ragnosis ask "Who led the Carthaginian army across the Alps?"
```

Retrieves the best passages and has a local language model answer from them,
citing them as `[1]`, `[2]`; the cited sources are listed after the answer. The
answer appears as it is written; on a laptop CPU a full answer takes about a
minute. `-k N` sets how many passages the model reads (default 4).

Needs [Ollama](https://ollama.com) running and the model in
`RAGNOSIS_LLM_MODEL` downloaded once:

```bash
ollama pull qwen3.5:4b
```

Exit code `2` means something to set up: no index, Ollama not running or the
model not downloaded; the message says which.

### Web UI

```bash
uv run python -m ragnosis serve
```

Then open <http://localhost:8000>. Type a question; the answer appears as it is
written and the cited sources link to Wikipedia. The models load once when the
server starts (about half a minute), and one question is answered at a time.
`--port N` picks another port. The server only listens on this computer and
the page loads nothing from the internet. Stop it with Ctrl+C.

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

On Windows with Smart App Control on, the `pytest.exe` and `mypy.exe`
launchers that uv creates can be blocked because they are unsigned. Running
the tools as modules avoids that: `uv run python -m pytest` and
`uv run python -m mypy` accept the same options.

### Automation on GitHub

- **Dependabot** opens a pull request every Monday when newer versions of the
  Python packages or the GitHub Actions are out; CI tests each one.
- **Weekly checks** run CI plus the integration tests against the real
  Wikipedia API and embedding model every Monday, to catch outside changes.
  Start them by hand under *Actions → Weekly checks → Run workflow*.

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

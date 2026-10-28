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

From nothing to a first answer takes about half an hour, most of it waiting for
downloads and the index. Everything runs on your own computer; a laptop with
16 GB of RAM and no graphics card is enough.

### 1. Install the tools

| Tool | What for | How (Windows) |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | Python, packages and running the commands | `winget install --id=astral-sh.uv -e` |
| [Ollama](https://ollama.com) | Runs the language model | `winget install --id=Ollama.Ollama -e` |
| [Git](https://git-scm.com) | Getting the code | `winget install --id=Git.Git -e` |
| [Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist) | Needed by PyTorch on Windows | `winget install --id=Microsoft.VCRedist.2015+.x64 -e` |

Open a new terminal afterwards, so it finds the new tools. On macOS and Linux,
install uv, Ollama and Git as their websites describe; the Visual C++ runtime
is not needed there.

### 2. Get the code and the models

```bash
git clone https://github.com/juninatt/RAGnosis.git
cd RAGnosis
uv sync
ollama pull qwen3.5:4b
```

`uv sync` installs Python 3.12 and the packages (about 1 GB, mostly PyTorch).
`ollama pull` downloads the language model (3.3 GB).

On Windows, clone into a short folder such as `C:\code` or `Documents`. Some
package files have long names, and Windows by default refuses paths longer than
260 characters; a deep folder makes `index` fail (see below).

### 3. Build your copy of the knowledge base

```bash
uv run python -m ragnosis ingest
uv run python -m ragnosis index
```

`ingest` downloads the articles in the topics file from Wikipedia (about a
minute). `index` splits them into passages and embeds them; the first time it
also downloads the embedding model (0.5 GB). On a laptop CPU it takes 5 to 15
minutes, and later runs only redo articles that changed.

### 4. Ask

```bash
uv run python -m ragnosis serve
```

Open <http://localhost:8000> and ask a question, in Swedish or English. Or ask
on the command line: `uv run python -m ragnosis ask "Who was Anne Bonny?"`.

### If something goes wrong

| Symptom | Cause and fix |
|---|---|
| `An Application Control policy has blocked this file` | Windows Smart App Control blocks unsigned programs, including uv's own Python. Install Python 3.12 from python.org (`winget install --id=Python.Python.3.12 -e`), add `python-preference = "system"` to `%APPDATA%\uv\uv.toml`, delete the `.venv` folder and run `uv sync` again. Use `uv run python -m ...` rather than `uv run ragnosis ...`. |
| `DLL initialization routine failed` when indexing | The Visual C++ Redistributable is missing or old; install it as in step 1. |
| `FileNotFoundError` for a file under `.venv\Lib\site-packages\transformers` when indexing | The folder path is too long for Windows (over 260 characters with the package's file names). Clone into a shorter folder, delete `.venv` and run `uv sync` again, or turn on Windows long paths. |
| `Ollama is not reachable` | Start Ollama (it normally starts with Windows), then try again. |
| `Model 'qwen3.5:4b' is not downloaded` | Run `ollama pull qwen3.5:4b`. |
| `No index in data/index` | Run `ingest` and then `index`, as in step 3. |
| Answers take well over a minute | Close heavy programs and use the power mode "Best performance"; a laptop slows down a lot on battery saving or when hot. |

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
`--detailed` asks for a paragraph or two instead of at most five sentences;
the first words come as soon, the whole answer 10 to 15 seconds later.

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
the page loads nothing from the internet. Stop it with Ctrl+C. Under the
question box you choose a short or a detailed answer; the page remembers the
choice in this browser.

### Evaluate the answers

```bash
uv run python -m ragnosis evaluate
```

Answers the known questions in [`evaluation/answers.toml`](evaluation/answers.toml)
with the real models and checks each answer: does it cite the right article,
mention the expected facts, use the question's language, or decline when the
sources do not cover the question? Prints PASS or FAIL per question with the
reasons, and a summary. `--case ID` runs only some questions, and `--detailed`
scores detailed answers. A full run passes (exit code 0) when it reaches the
`required` level in the cases file, 14 of 16; a run of chosen cases passes
when all of them do. A full run takes 10 to 15 minutes. Compare runs before
and after a change: answers are reproducible, so a difference comes from the
change. Results so far are in [`docs/evaluation.md`](docs/evaluation.md).

### Settings

Settings are environment variables. Their defaults are in [`.env`](.env); put
your own values in `.env.local`, which git ignores and which overrides `.env`.
Real environment variables override both.

| Setting | Default | What it does |
|---|---|---|
| `RAGNOSIS_LOG_LEVEL` | `INFO` | How much the commands log: `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`. |
| `RAGNOSIS_DATA_DIR` | `data` | Where downloaded articles and the index are kept. |
| `RAGNOSIS_TOPICS_FILE` | `topics/pirates-and-antiquity-europe.txt` | The list of Wikipedia titles that makes up the corpus. |
| `RAGNOSIS_WIKI_LANGUAGE` | `en` | The Wikipedia language edition the titles are from. |
| `RAGNOSIS_EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | The model that turns text into vectors. Changing it needs `index --rebuild`. |
| `RAGNOSIS_OLLAMA_URL` | `http://localhost:11434` | Where Ollama runs. |
| `RAGNOSIS_LLM_MODEL` | `qwen3.5:4b` | The model that writes answers; download it with `ollama pull`. |
| `RAGNOSIS_LLM_KEEP_ALIVE` | `30m` | How long Ollama keeps the model loaded after an answer, e.g. `30m`, `1h`, or `-1` for always. |
| `RAGNOSIS_QUERY_REWRITE` | `always` | When questions are rewritten into English search keywords: `always`, `swedish` or `never`. |

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
- **Releases:** pushing a tag such as `v0.5.0` publishes a GitHub release with
  that version's `CHANGELOG.md` section, after checking that the tag matches
  the version in `pyproject.toml`. *Actions → Release → Run workflow* does the
  same for an existing tag.
- **Commit subjects** are checked on every push and pull request
  (`tools/check_commits.py`): `type(scope): description`, at most 72 characters.

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
- **Releases** follow the checklist in [`docs/release-checks.md`](docs/release-checks.md),
  which also logs the manual checks.

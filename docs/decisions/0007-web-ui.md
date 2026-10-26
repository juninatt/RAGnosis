# 0007: A small web UI

**Status:** accepted

## Context

Asking through the command line works, but a page with a question box, an
answer that appears as it is written and clickable sources is easier to use.
Streamlit was the first idea. On the development machine Windows Smart App
Control blocked one of its dependencies (a compiled `pandas` module), which
would make the UI fail unpredictably.

## Decision

- **Starlette and uvicorn** serve the UI. Both are pure Python, so there is no
  compiled code for Smart App Control to block, and Starlette's test client
  makes the HTTP layer easy to test.
- **One HTML page with a little plain JavaScript**, shipped inside the package.
  No build step, no external scripts, fonts or styles: the page works offline
  and sends nothing anywhere else.
- **A small HTTP API:** `POST /api/ask` streams the answer as JSON lines (text
  pieces, then the cited sources, or an error), and `GET /api/health` reports
  that the server is up.
- **Localhost only.** The server binds to `127.0.0.1`, only accepts the host
  names `localhost` and `127.0.0.1` (against DNS rebinding) and only JSON
  requests (so other web pages cannot post questions without a CORS check).
- **One question at a time.** A CPU answers one question at a time anyway; a
  second request gets HTTP 429 rather than slowing both down.
- **Models load once**, when the server starts, instead of on each question.
- **Model output is inserted as text**, never as HTML, so an answer cannot run
  code in the page.

## Consequences

- `python -m ragnosis serve` starts the UI; the CLI and the UI share `ask()`.
- The JavaScript has no automated tests, since that needs a browser in CI. It
  is kept small and checked by hand; end-to-end browser tests are a candidate
  before 1.0.
- Serving several users at once would need a queue or a bigger machine; that
  is not a goal for a local assistant.

# 0009: What 1.0 promises

**Status:** accepted

## Context

Until 1.0, commands, settings and the index format changed freely between
minor versions. 1.0 is where that stops: someone who sets RAGnosis up should be
able to upgrade within 1.x without changing how they use it, and should get
answers of a known quality.

## Decision

From 1.0, changing any of these in an incompatible way needs version 2.0:

- **Commands and their options:** `ingest`, `index`, `search`, `ask`, `serve`
  and `evaluate`, and their exit codes (0 success, 1 some work failed, 2 bad
  usage or configuration).
- **Settings:** the `RAGNOSIS_*` variables listed in the README, and what they
  mean. Defaults may change in a minor version, with a note in the changelog.
- **The web API** the page uses: `POST /api/ask` with `question`, `k` and
  `length`, answered with JSON lines of `text`, `sources` and `error` events.
- **Data on disk:** downloaded articles and the index format. A change that
  needs `index --rebuild` is allowed in a minor version only when the program
  says so clearly, since the index can always be rebuilt from the articles.

Not promised: the Python modules inside `ragnosis`, the page's look, and the
prompts, which are tuned against the scorecard.

**Quality:** with the default settings, a release must pass at least 14 of the
16 known questions in `evaluation/answers.toml`. The required level lives in
that file, so `evaluate` can tell whether a build meets it. Two failures are
allowed for known limits of the small local model, which the changelog lists.

## Consequences

- New settings and options can be added in 1.x; renaming or removing one waits
  for 2.0.
- Prompt and model changes are judged by the scorecard rather than by promise.
- A test guards that every setting is documented, so the promise covers all of
  them.

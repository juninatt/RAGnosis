# 0008: Measuring answers, and rewriting questions for search

**Status:** accepted

## Context

Retrieval has quality tests (0005), but the answers themselves do not: whether
a change makes answers better or worse was judged by reading a few by hand.
Known failures are in the search step: "Hur dog Svartskägg?" finds nothing
about Blackbeard, because the embedding model does not know that Svartskägg is
Blackbeard's Swedish name, and "Hur dog Blackbeard?" finds general Blackbeard
passages rather than his death.

## Decision

- **Known questions with checkable expectations** live in
  `evaluation/answers.toml`: which articles a good answer cites, facts it
  mentions, the language it is written in, or that it declines to answer.
- **Rule-based checks**, not a model as judge. A small local model judging its
  own answers would be unreliable, and rules give the same verdict every time
  and say exactly why an answer failed.
- **`python -m ragnosis evaluate`** runs the questions through the real chain
  and prints a scorecard with timings. Model output varies a little between
  runs, so the scorecard is a measurement to compare, not a unit test that
  must pass.
- **The question is rewritten for search.** Before retrieval the model turns
  the question into an English search query with the standard English names;
  the answer is still written from the original question, so its language is
  right. A setting chooses when to rewrite, since it costs an extra model call.

## Consequences

- Changes to prompts, models, chunking or search can be compared by running
  the scorecard before and after; results are kept in `docs/evaluation.md`.
- Rewriting adds a few seconds before the first words.
- The expectations are only as good as the cases; new findings from testing
  become new cases.

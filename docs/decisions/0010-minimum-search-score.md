# 0010: A minimum search score for answering

**Status:** accepted

## Context

Search always returns the best `k` passages, even when none is about the
question. Asked who won the 2022 World Cup, the model is given four passages
about Rome and Greece. It usually declines, but often cites those passages
while doing so, so unrelated sources are listed under a refusal. Telling it not
to in the prompt did not work, and hiding the sources of answers that start by
declining removed correct sources too (see `docs/evaluation.md`).

## Decision

- **No passage counts as relevant below a minimum score** of the best passage
  (`RAGNOSIS_MIN_SCORE`, default 0.83). The question is then declined with a
  fixed text in its language, without the model and without sources.
- **The default is measured, not guessed.** With `intfloat/multilingual-e5-small`
  and the current topics, every known question that should be answered scored
  at least 0.845 and every one that should be declined at most 0.811.
- **`evaluate` shows the best score of every case,** so the limit can be
  measured again.

## Consequences

- Questions outside the topics get a clear answer without unrelated sources,
  in about 6 seconds instead of 30.
- The margin is small (0.811 against 0.845): E5 gives most text a score
  between 0.75 and 0.9. A relevant question just below the limit is declined.
- The limit depends on the embedding model and the passages. Measure again
  with `evaluate` when either changes, such as when topics are added.
- Answers that decline although passages scored above the limit, or that
  answer after declining, are not fixed by this.

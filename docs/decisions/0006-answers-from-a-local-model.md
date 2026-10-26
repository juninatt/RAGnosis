# 0006: Answers from a local model

**Status:** accepted

## Context

Retrieval finds the passages that answer a question; a language model has to
turn them into an answer. It must run free and locally (0001) on a CPU with
16 GB RAM, and it must not invent facts the passages do not contain.

## Decision

- **Ollama** runs the model and is reached over its local HTTP API. The
  executable is signed, so it runs with Smart App Control on.
- **Model `qwen3.5:4b`** (3.3 GB, Apache 2.0) by default. It is recent, handles
  several languages and fits in memory next to the embedding model. Its
  "thinking" mode is turned off: on a CPU it multiplies answer time, and the
  answer is in the passages anyway.
- **Grounded prompt.** The top passages are numbered in the prompt, and the
  model is told to use only them, to cite them as `[1]`, `[2]`, to say when
  they do not hold the answer, and to answer in the question's language.
- **Citations are parsed** from the answer, so only sources the model actually
  used are listed with the answer.
- **Answers are streamed** token by token, because a full answer takes 40 to
  55 seconds on the development machine.

## Consequences

- A first test was grounded and cited, and declined a question the sources did
  not cover.
- Swedish answers are understandable but stiff; a larger model writes better
  and runs slower. The model is a setting.
- Short Swedish words can be misread as English: "Hur dog Blackbeard?" was read
  as being about a dog, and retrieval matched general Blackbeard passages
  rather than his death. Rewriting the question into an English search query
  before retrieval is planned for 0.5.0, together with translated names such as
  "Svartskägg".

# Evaluation results

Scorecards from `python -m ragnosis evaluate` over the known questions in
[`evaluation/answers.toml`](../evaluation/answers.toml) (see
[decision 0008](decisions/0008-measuring-answers-and-rewriting-questions.md)).
Rerun them after changing prompts, models, chunking or search, and add the
results here, newest first.

Times depend on the machine and what else runs on it; compare them only within
one table.

## 2026-10-28: hiding the sources of declining answers (tried, not kept)

To stop detailed refusals from listing unrelated sources, an answer whose first
sentence declined ("I cannot answer…") was given no sources at all.

| Answer length | 0.5.0 | With the change |
|---|---|---|
| short | 15 / 16 | 14 / 16 |
| detailed | 14 / 16 | 14 / 16 |

world-cup-sv passed in detailed mode, as intended, but female-pirates-en failed
in both: the model says it cannot answer (it does not know that John Rackham is
Calico Jack) and then names Mary Read and Anne Bonny correctly, with sources.
Hiding those sources makes the answer worse, so the change was removed. A fix
needs to judge whether the cited sources are relevant, for example by their
search scores. The times in these runs are not usable: the computer slept
during some cases.

## 2026-10-28: short and detailed answers (0.5.0)

Same setup as below, rewriting `always`. `evaluate --detailed` asks for one or
two paragraphs of 6 to 10 sentences instead of at most 5 sentences.

| Answer length | Passed | Median per answer | Total |
|---|---|---|---|
| short (default) | 15 / 16 | 44 s | 13.8 min |
| detailed | 14 / 16 | 41 s | 11.5 min |

- Facts, citations and language held up in detailed answers.
- **world-cup-sv fails in detailed mode only:** the answer correctly says it
  cannot answer, then retells what the sources are about and cites them, so
  the page would list four unrelated sources under a refusal. A prompt rule
  against citing or describing the sources did not stop it; a fix in code is
  planned.
- The machine's speed varied between runs more than the answer length did, so
  the medians do not show the cost. One detailed answer in the web UI took 56 s
  against about 35 to 45 s for short ones; the first words come as soon.

## 2026-10-27/28: rewriting questions for search (0.5.0)

**Setup:** `qwen3.5:4b` through Ollama, `intfloat/multilingual-e5-small`, 4
passages per answer, temperature 0 with a fixed seed, laptop CPU (Intel Core
i5-1145G7, 16 GB RAM). 16 cases: 9 in Swedish, 7 in English, 2 of them outside
the sources.

| `RAGNOSIS_QUERY_REWRITE` | Passed | Median per answer | Total |
|---|---|---|---|
| `never` | 12 / 16 | 32 s | 9.2 min |
| `swedish` | 13 / 16 | 34 s | 10.3 min |
| **`always`** (chosen default) | **15 / 16** | 44 s | 13.8 min |

| Case | never | swedish | always | Notes |
|---|---|---|---|---|
| blackbeard-death-en | ✗ | ✗ | ✓ | "How did Blackbeard die?" ranks the passage about his death 8th; the keywords find it |
| blackbeard-death-sv | ✗ | ✓ | ✓ | Without rewriting, the answer was about his beard |
| svartskagg-death-sv | ✗ | ✗ | ✗ | The model does not know that Svartskägg is Blackbeard and searches for the Black Death; it declines rather than inventing an answer |
| salamis-en | ✗ | ✗ | ✓ | Triremes were only found with keywords |
| the other 12 | ✓ | ✓ | ✓ | |

**Findings**

- Search matches keywords better than questions: "How did Blackbeard die?"
  ranked the passage about his death 8th, "Blackbeard killed in battle" 1st.
  Rewriting into keywords helps English questions as much as Swedish ones.
- Rewriting costs about 6 seconds per question.
- Translated names need more than rewriting with this model. Planned: store
  each article's Swedish name from Wikipedia or Wikidata in the index.
- The rewrite prompt's examples must not come from the known questions. An
  early prompt used "Svartskägg becomes Blackbeard" as its example, which would
  have passed that case for the wrong reason; a test now guards against it.

### Before the measurements were reliable

The first two runs had to be redone, and are kept here so the numbers above
can be read with that in mind:

- **First run, without rewriting: 12 of 16.** Two failures were in the checks,
  not the answers: the language check called a correct Swedish answer
  "unknown", and the flag case expected the name "Jolly Roger" although the
  answer described the flag correctly. After fixing both, the same answers
  scored 14 of 16.
- **Second run, rewriting Swedish questions: 11 of 16.** Two English cases,
  untouched by rewriting, changed verdict between runs: answers varied by
  chance. Answers have since been written with temperature 0 and a fixed seed,
  and identical inputs now give identical answers, as the `never` and
  `swedish` runs above show for the English cases.

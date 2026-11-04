# Evaluation results

Scorecards from `python -m ragnosis evaluate` over the known questions in
[`evaluation/answers.toml`](../evaluation/answers.toml) (see
[decision 0008](decisions/0008-measuring-answers-and-rewriting-questions.md)).
Rerun them after changing prompts, models, chunking or search, and add the
results here, newest first.

Times depend on the machine and what else runs on it; compare them only within
one table.

## 2026-11-04: tables, infoboxes and image captions (1.3.0)

Same setup as below. Two cases were added that only a table answers: the ships
Athens and Corinth sent to Salamis. The level of 85 % became 23 of 26
([decision 0012](decisions/0012-tables-infoboxes-and-captions.md)).

| | Passed | Median per answer | Total |
|---|---|---|---|
| With tables, infoboxes and captions | **26 / 26** | 43 s | 18.4 min |

- athens-ships-en and corinth-ships-sv passed with 180 and 40 ships, numbers
  that are only in the table. Their best scores were 0.912 and 0.871.
- salamis-en passed again, now answering from four passages of the article
  on the battle: the answer mentions triremes and no longer says it cannot
  answer.
- The margin of the minimum score held: the questions to be answered scored
  at least 0.857, those to be declined at most 0.812.
- The times cannot be compared with 1.2.0. The computer was busy: the tests
  and `index` ran about twice as slowly as usual too. A second run was
  stopped before it finished.

## 2026-11-02/03: other names from Wikidata (1.2.0)

Same setup as below, rewriting `always`, short answers, minimum score 0.83.
Three cases were added that ask by another name than the title, and the level
of 85 % became 21 of 24
([decision 0011](decisions/0011-other-names-from-wikidata.md)).

| | Passed | Median per answer | Total |
|---|---|---|---|
| Names in the index, both searches merged by score | 21 / 24 | 35 s | 12.7 min |
| **With the four fixes below** | **23 / 24** | 37 s | 13.3 min |

| Case | Should | Best score | By score | Fixed |
|---|---|---|---|---|
| svartskagg-death-sv | answer | 0.861 | ✗ | ✓ |
| black-bart-en | answer | 0.894 | ✗ | ✓ |
| jolly-roger-sv | answer | 0.917 | ✗ | ✓ |
| salamis-en | answer | 0.905 | ✓ | ✗ |
| calico-jack-en | answer | 0.923 | ✓ | ✓ |
| kapten-kidd-sv | answer | 0.874 | ✓ | ✓ |
| 14 other cases | answer | 0.857–0.924 | ✓ | ✓ |
| 4 cases outside the topics | decline | 0.769–0.812 | ✓ | ✓ |

- With the names only in the index, two rewrites still went wrong:
  "Svartskägg" became the Black Death and "Black Bart" a robber gang leader, and
  their passages scored higher than the right ones the question found.
  Without a source naming Svartskägg, the model also called him "Cleitus the
  Black".
- jolly-roger-sv was a correct Swedish answer that the language check read as
  English: "Death's Head" and "Pirates of the Caribbean" gave two English
  markers against one Swedish.
- The fixes, one commit each: known names get their article's title before
  the rewrite, the two searches are merged by place, the model sees the names
  with each source, and the language check knows "finns" and "ett".
- salamis-en: merged by place, the passage from the article on triremes no
  longer made the four. The answer says the sources do not name the type of
  ship, then that the Greek fleet had triremes, overlooking a sentence that
  says the Athenian fleet had them. A table of ships per city is missing from
  the passage, as Wikipedia's text extracts leave tables out.
- The margin of the minimum score grew: 0.857 against 0.812, from 0.845
  against 0.811.
- A third run, after a genitive s was limited to names with a capital letter,
  gave the same verdicts; its times are not usable, as the computer slept.

## 2026-11-01: answers that refuse and then answer (after 1.1.0)

In 1.0, female-pirates-en said it could not answer and then named Anne Bonny
and Mary Read correctly, with sources. A check now fails such answers, and two
questions the sources only partly answer were added.

| | Passed |
|---|---|
| With the check | 18 / 21 |
| With a prompt rule for partial answers (tried, not kept) | 18 / 21 |

- The 1.0 failure no longer happens: female-pirates-en answers directly.
- largest-crew-en and pompey-ships-en say they cannot answer and then give
  context with sources, such as Blackbeard's crew of 300 men. Reading them,
  that is right: the sources do not say which pirate had the largest crew or
  how many ships Pompey used. Such cases are marked `partial` and may refuse
  with context; both pass.
- A prompt rule asking the model to answer the covered part and name what is
  missing changed neither answer, and made svartskagg-death-sv cite the Roman
  passages it declined, so it was removed.
- Times are not given: other work on the computer slowed the first run about
  threefold.

## 2026-11-01: a minimum search score (1.1.0)

Same setup as below, rewriting `always`, short answers. Three cases were added
close to the edge of the topics, and the level became 85 % (17 of 19). The
best search score of each case decided the default minimum score of 0.83
([decision 0010](decisions/0010-minimum-search-score.md)).

| | Passed | Median per answer | Total |
|---|---|---|---|
| Without a minimum | 15 / 19 | 34 s | 10.9 min |
| **Minimum 0.83** (default) | **18 / 19** | 33 s | 9.1 min |

| Case | Should | Best score | Without | 0.83 |
|---|---|---|---|---|
| 11 other cases | answer | 0.864–0.910 | ✓ | ✓ |
| female-pirates-en | answer | 0.879 | ✗ | ✓ |
| svartskagg-death-sv | answer | 0.858 | ✗ | ✗ |
| zheng-yi-sao-sv | answer | 0.845 | ✓ | ✓ |
| china-emperor-en | decline | 0.811 | ✗ | ✓ |
| japan-capital-en | decline | 0.797 | ✓ | ✓ |
| world-cup-sv | decline | 0.789 | ✓ | ✓ |
| sadie-farrell-sv | decline | 0.766 | ✗ | ✓ |

- Answers that should be declined scored at most 0.811, those that should be
  answered at least 0.845. The margin is small.
- china-emperor-en and sadie-farrell-sv declined correctly without the
  minimum but cited the passages anyway. Below the minimum the model is not
  run, so nothing is cited, and the answer takes about 6 s instead of 30.
- female-pirates-en was right in both runs; the case did not accept the
  article on Calico Jack, which Wikipedia names John Rackham.
- svartskagg-death-sv still fails: the question is rewritten to the Black
  Death, and passages about Rome score 0.858. Swedish names in the index are
  planned for that.

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

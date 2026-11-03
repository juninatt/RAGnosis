# 0011: Other names from Wikidata

**Status:** accepted

## Context

Questions often use another name than the article's title: a Swedish name
("Svartskägg" for Blackbeard), a nickname ("Black Bart" for Bartholomew
Roberts) or a common form ("Calico Jack" for John Rackham). The embedding model
does not know that these are the same person, and the small language model
translates them wrongly when it rewrites the question: "Hur dog Svartskägg?"
was searched for as the Black Death and answered from passages about Rome.

## Decision

- **`ingest` fetches each article's other names from Wikidata**: its labels
  and then its aliases in the languages of `RAGNOSIS_NAME_LANGUAGES` (default
  `en,sv`), without duplicates or the title, at least 4 characters and at most
  10 names. One request covers 50 articles. The names are stored with the
  article and only help to find passages; answers still come from Wikipedia
  text.
- **The names are embedded with every passage** of the article, after the
  title: `Blackbeard (Svartskägg, Edward Teach) > Death`. This is index format
  2; an older index must be rebuilt.
- **Known names in the question get their article's title** before anything
  else happens: "Hur dog Svartskägg?" becomes "Hur dog Svartskägg
  (Blackbeard)?". Names match as whole words in any case, longest first. A
  genitive s ("Svartskäggs") matches only after names Wikidata writes with a
  capital letter, as after other words it is a plural: "corsairs" is not the
  alias "corsair" of Privateer.
- **Both the question and the rewritten query are searched for,** and the two
  lists are merged by place (reciprocal rank fusion with the usual constant
  60), not by score. Merged by score, a wrong rewrite that scores high on its
  own passages pushed out the right ones. Each passage keeps its better
  score, which the minimum score (0010) is compared with.
- **The model sees the names with each source,** in the same heading as is
  embedded, so it knows that a passage about Blackbeard answers a question
  about Svartskägg.

## Consequences

- The known questions went from 21 to 23 of 24 (see `docs/evaluation.md`);
  every question asked by another name now finds its article.
- The minimum score has a wider margin: the questions to be answered scored
  at least 0.857, those to be declined at most 0.812.
- Answering is not measurably slower. `ingest` makes one more request per 50
  articles.
- Merging by place can drop a passage that only one search ranks high:
  salamis-en lost the passage on triremes and its answer became too cautious.
- Names Wikidata does not have, and other inflections than the genitive s
  ("Svartskäggen"), are still not recognised. An alias that is a common word
  can add a wrong title, which only adds passages to choose from.
- After upgrading, run `ingest` to fetch the names of stored articles and
  `index --rebuild`.

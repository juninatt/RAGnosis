# Release checks

What is checked before a release, and a log of the checks that cannot be
automated. CI and the scorecard cover the code and the answers; the manual
checks cover what a new user meets: installing from scratch and using the page.

## Checklist

Before tagging a release:

**Automated**

- [ ] CI is green on Linux and Windows for the commit to be tagged.
- [ ] `uv run python -m ragnosis evaluate` reaches the required level (14 of 16)
      with the default settings; add the scorecard to
      [evaluation.md](evaluation.md) if prompts, models or search changed.

**Manual** (log the result below)

- [ ] **Clean install:** clone into an empty, short folder and follow
      *Getting started* in the README word for word, through a first answer
      with `ask` and in the web UI.
- [ ] **Web UI:** ask one question in Swedish and one in English with short
      and with detailed answers; check that the answer streams in, sources
      link to Wikipedia, the time line shows, and the length choice is
      remembered after a reload. Look at it in dark mode and at phone width.
- [ ] `CHANGELOG.md` has a section for the version, and `pyproject.toml`
      carries the version.

## Log

Newest first. Note what was checked, on which machine, the result, and what
was fixed because of it.

| Date | Version | Check | Result | Notes |
|---|---|---|---|---|
| 2026-10-28 | 0.6.0 | Clean install, second try | ✓ | Short folder `Documents\juninatt\_clean-test`, steps as in the README: clone 1 s, `uv sync` 7 s, `ingest` 28 s (64 articles), `index` 4.5 min (3,357 passages), `ask` 64 s with sources, `serve` up in 24 s with the page served. Same machine and caveats as below; the guide's estimate for `index` became "5 to 15 minutes". |
| 2026-10-28 | 0.6.0 | Clean install | ✗ | Clone in a deep folder: `index` failed with `FileNotFoundError` under `transformers`, as the path reached 269 characters and Windows long paths were off. The README now says to clone into a short folder and lists the error under troubleshooting. Windows 11, i5-1145G7, 16 GB; uv, Ollama, Git and the model were already installed, so download times were not measured. |

# 0002: Configuration in env files

**Status:** accepted

## Context

Settings such as log level, model name and data paths should be easy to change
without editing code, and the same on every machine. Secrets, such as a future
hosted-model API key, must never reach GitHub.

## Decision

- Non-sensitive defaults live in `.env`, which is committed.
- Secrets and personal overrides live in `.env.local`, which git ignores.
- Precedence, highest first: real environment variables, `.env.local`, `.env`.
- All keys start with `RAGNOSIS_` so they cannot collide with other tools.

## Consequences

- A fresh clone works without any setup, because `.env` carries the defaults.
- Committing `.env` means a secret added there would be published. Tests reject
  keys that look like secrets (`KEY`, `TOKEN`, `SECRET`, `PASSWORD` and similar)
  and keys without the project prefix.
- CI and one-off runs override settings through real environment variables.

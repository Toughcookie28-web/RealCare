# Documentation instructions

## Purpose
The docs folder is the persistent source of truth for architecture, decisions, and meaningful implementation history.

## Rules
- Keep docs concise, concrete, and current.
- Do not treat chat history as the source of truth when docs exist.
- When architecture changes materially, update `architecture/current-state.md`.
- When a major design decision is made, add or update an ADR in `decisions/`.
- When a meaningful implementation change is completed, append a concise entry to `changes/implementation-log.md`.

## Editing guidance
- Prefer updating existing docs over creating redundant new docs.
- Keep ADRs focused on context, decision, consequences, and alternatives.
- Keep implementation log entries focused on:
  - what changed
  - why it changed
  - tradeoff introduced
  - what must remain true afterward

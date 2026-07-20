# Domain documentation

This repository uses a single domain context.

Before changing behavior, read the root `AGENTS.md`, the owning model family's
README and tests, and relevant ADRs under `docs/adr/`. If a root `CONTEXT.md` is
added later, treat it as the repository glossary and use its terms in issues,
tests, interfaces, and documentation.

No multi-context workspace signal currently exists. MantisV2 remains the active
model-family boundary; do not infer separate contexts for reserved `mantis/` or
`mantis-plus/` directories.

If a proposed change contradicts an ADR, surface the conflict explicitly rather
than silently overriding it.

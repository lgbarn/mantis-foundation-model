# Issue tracker: GitHub

Implementation slices and their authoritative execution contracts live in
GitHub Issues at `lgbarn/mantis-foundation-model`. Accepted parent specifications
remain versioned repository documents. When an issue and its named parent spec
conflict, the accepted parent spec wins until both are deliberately reconciled.
Use the `gh` CLI from this clone so repository selection follows the configured
remote.

## Operations

- Create: `gh issue create --title "..." --body-file <path>`
- Read: `gh issue view <number> --comments`
- List: `gh issue list --state open --json number,title,body,labels,comments`
- Comment: `gh issue comment <number> --body "..."`
- Label: `gh issue edit <number> --add-label "..."`
- Close: `gh issue close <number> --comment "..."`

Issue bodies are authoritative contracts. Comments provide context but do not
replace the acceptance criteria in the body.

## Pull requests as a triage surface

External pull requests are not a request surface. Do not mix them into the issue
triage queue. Pull requests created from accepted issues follow the repository's
normal review and merge workflow.

## Dependencies

Prefer GitHub native blocked-by relationships. If the repository or API does not
support them, put a normalized `Blocked by` section in the issue body. An issue
is unblocked only when every implementation blocker is closed. Decision-only
edges must be labeled explicitly.

## AFK readiness

Apply `ready-for-agent` only when the issue body contains a complete, observable
contract: current state, desired behavior, independently verifiable acceptance
criteria, owned surfaces, resolved design decisions, out-of-scope boundaries,
and all dependencies. Missing authority, ambiguous behavior, manual-only checks,
or absent test oracles require `needs-info` instead.

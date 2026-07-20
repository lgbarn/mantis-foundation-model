# Triage labels

| Canonical role | GitHub label | Meaning |
| --- | --- | --- |
| `needs-triage` | `needs-triage` | Maintainer evaluation required |
| `needs-info` | `needs-info` | Waiting for missing context or authority |
| `ready-for-agent` | `ready-for-agent` | Fully specified and ready for AFK execution |
| `ready-for-human` | `ready-for-human` | Ready but requires human implementation or judgment |
| `wontfix` | `wontfix` | Will not be actioned |

The `tdd` modifier is orthogonal to triage state. Apply it to deep-module slices
whose branching behavior, money paths, parsers, or state machines must be built
test-first. Thin configuration or wiring slices do not need it.

An issue has at most one canonical triage-state label at a time.

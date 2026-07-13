# weave-agent-signals specs

Detailed specs behind [`DESIGN.md`](DESIGN.md). Each is self-contained; `DESIGN.md` stays the high-level map (including M2+ designs in sections L2–L4).

| # | Spec | Milestone | Status |
|---|---|---|---|
| — | [Design overview](DESIGN.md): architecture, layers, milestones | all | draft |
| 01 | [Data flow & span reader](01-data-flow.md): query model, turn/session hydration | M1 | draft |
| 02 | [Outcome extractor](02-outcome-extractor.md): test/build/lint/git parsing from Bash spans | M1 | draft |
| 03 | [Implicit feedback](03-implicit-feedback.md): correction density, abandonment | M1 | draft |
| 04 | [Efficiency scorers](04-efficiency-scorers.md): error loops, repeated reads, waste ratio | M1 | draft |
| 05 | [Score write-back](05-score-writeback.md): feedback API, ref types, dedup, batch | M1 | draft |
| 06 | [CLI](06-cli.md): score/backfill/inspect, scheduling | M1 | draft |
| 07 | [REST API](07-api.md): FastAPI server wrapping CLI operations | M1 | draft |
| 08 | [Frontend](08-frontend.md): React SPA for viewing and triggering operations | M1 | draft |
| 09 | [Evaluation runs](09-evaluation-runs.md): unified run pipeline replacing separate score/judge/reflect pages | M2 | draft |

Convention: keep specs lean. They give context for agents reading the repository — *what* and *why*, not implementation details (code examples, exact signatures, JSON shapes) readable from the source itself. `DESIGN.md` is the high-level map.

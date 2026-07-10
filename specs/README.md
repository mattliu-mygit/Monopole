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

Convention: specs describe *intended* behavior for their milestone. Items marked **OPEN** need verification against live APIs or real session data. M2+ designs live in `DESIGN.md` sections L2–L4 and get promoted to numbered specs when their milestone starts.

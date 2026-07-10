# weave-agent-signals specs

Detailed specs behind [`DESIGN.md`](DESIGN.md). Each is self-contained; `DESIGN.md` stays the high-level map.

| # | Spec | Milestone | Status |
|---|---|---|---|
| — | [Design overview](DESIGN.md): architecture, layers, milestones | all | draft |
| 01 | [Data flow & span reader](01-data-flow.md): query model, turn/session hydration | M1 | draft |
| 02 | [Outcome extractor](02-outcome-extractor.md): test/build/lint parsing from Bash spans | M1 | draft |
| 03 | [Implicit feedback](03-implicit-feedback.md): steering, denials, follow-ups, abandonment | M1 | draft |
| 04 | [Efficiency scorers](04-efficiency-scorers.md): error loops, repeated reads, waste flags | M1 | draft |
| 05 | [Score write-back](05-score-writeback.md): feedback API, ref types, dedup, batch | M1 | draft |
| 06 | [CLI](06-cli.md): score/backfill/inspect/setup, scheduling, configuration | M1 | draft |
| — | [Future phases](FUTURE.md): gates, judges, digest, patterns, RSI, coach agent (stretch) | M2+ | parking lot |

Convention: specs describe *intended* behavior. Items marked **OPEN** need verification against live APIs or real session data. Future-phase sections in `FUTURE.md` hold settled research decisions and get promoted to numbered specs when their milestone starts.

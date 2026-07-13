# Spec 02: Outcome extractor

Parses verifiable environment outcomes from `execute_tool` Bash spans. When a test suite runs and reports results, that's ground truth no LLM judge can beat — the strongest deterministic signal in the pipeline.

Ties to spec 01 (reads `ToolSpan`), spec 05 (write-back).

## What we parse (M1)

Four evidence types from Bash tool calls: test runs, build/compile, lint/format, git operations. All Tier 1 — regex detection over the command string plus framework-specific output parsing, with **exit code as universal fallback** when output parsing fails. `confidence=1.0` always (deterministic).

Covers this user's sessions well because they're pytest-dominant; other frameworks (jest, cargo, go test, rspec, etc.) are supported but secondary. M2 adds a micro-LLM fallback for the unparsed residue — see DESIGN.md L1 Class 3.

**Classification order matters**: lint is checked before build, because `pyright` (a type checker) would otherwise be misclassified as a compiler alongside `tsc`/`cargo build`.

## Command parsing

Bash commands are classified with a regex-based classifier, not a real shell parser. Rules worth knowing:

- Pipes (`|`): classify by the first (producer) command — `pytest | tee log` is a test run.
- `cd` and env-var prefixes are stripped before classifying the rest.
- Subshells (`$(...)`, backticks) are skipped — too ambiguous to classify safely.
- Chained commands (`&&`, `||`, `;`) are **not split** in M1 — only the segment before a pipe is classified. Known gap. Accepted trade-off: false negatives (missed commands) are fine, false positives (wrong classification) are not.

Backfill reports classification coverage (% of Bash spans classified) so coverage is measured, not assumed.

## Turn-level scoring and the attribution caveat

Each outcome type produces its own score (`outcome.test`, `outcome.build`, `outcome.lint`, `outcome.git`) — a turn can carry several.

**Important semantics**: a turn-level outcome score marks *where a test ran*, not *whose work made it pass* — tests passing in turn N reflect cumulative work from turns 1..N, not just turn N's diff. Turn-level scores are navigation anchors; the meaningful evaluation units are session-level **arcs** computed from those anchors: `session_ended_green` (last test outcome passed), `recovered` (fail → later pass), `ended_red` (fail never resolved).

## Edge cases worth knowing

- Bash output is capped at 32KB by the adapter; a truncated summary line falls back to exit code.
- Missing exit codes are inferred from output text patterns (`"FAILED"`, `"Error:"`).
- Nested test runs (`make test` invoking `pytest`) are classified as the outer command, not double-counted.
- Interactive commands (`git rebase -i`) are skipped — not observable from output alone.
- Dry runs (`pytest --collect-only`, `black --check --diff`) classify as lint/check, not test/format.

## Non-goals

- Non-Bash tool outputs (Read/Edit/Write have no verifiable outcome in this sense).
- Semantic analysis of code changes — that's L2's job.
- Test quality (coverage, assertion depth) — not observable from output.

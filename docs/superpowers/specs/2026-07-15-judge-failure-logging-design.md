# Judge Failure Logging Design

## Goal

Make a judging failure diagnosable from server logs without weakening coverage
rules or exposing judge inputs and outputs.

## Observable behavior

When an applicable rubric receives no successful reviewer score, the warning
identifies the rubric and summarizes every reviewer attempt. Each attempt shows
its position, trigger, requested and resolved model identity, outcome, output
mode, schema version, transport request count, safe error category, and raw
output digest when available.

The summary makes these cases visibly different:

- every reviewer returned a valid insufficient-evidence verdict;
- a transport failed before returning a verdict;
- returned content failed JSON or verdict validation; and
- a mixture of abstained and failed attempts produced no score.

The existing fail-closed behavior is unchanged: an applicable rubric with no
successful reviewer still fails stage coverage, and the complete bounded
attempt audit remains available in run state.

## Safety boundary

Application logs never include prompts, evidence text or IDs, tool content,
raw model output, verdict rationales, provider response bodies, credentials, or
arbitrary exception text. The diagnostic summary is built from an explicit
allowlist of already-normalized metadata. Text identifiers are whitespace
normalized and bounded before logging.

The output digest remains a one-way SHA-256 identifier that can correlate a log
entry with the durable attempt audit without reproducing model content.

## Architecture

The judge runner remains the authoritative boundary because it has the final
ordered reviewer attempts and is used by both managed runs and the standalone
judge command. One focused formatter converts those attempts into a bounded,
deterministically ordered diagnostic string. The failed-review exception owns
that safe summary, so the runner warning and the enclosing run-stage warning
describe the same failure without duplicating formatting logic.

No run schema, frontend contract, rubric prompt, verdict schema, review policy,
applicability rule, or model invocation setting changes.

## Testing

A regression test first reproduces the current lossy warning. It then requires
the log to expose per-attempt status and safe transport/schema metadata for a
mixed abstention-and-validation failure. Sentinel rationale, evidence, error,
and raw-output text must remain absent. Existing judging and run-stage tests
continue to prove score aggregation, escalation, durable audit, and incomplete
coverage behavior.

## Cleanup criteria

The implementation adds one formatter and its focused regression coverage.
There are no compatibility adapters or duplicate logging paths. Canonical
evaluation documentation is updated only if the final behavior adds a durable
product invariant not already expressed by the inference trust boundary. This
temporary design and its implementation plan are removed after the final code
and canonical documentation agree.

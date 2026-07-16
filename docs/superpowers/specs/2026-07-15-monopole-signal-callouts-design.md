# Monopole Signal callouts design

## Goal

Call out conversations with low standalone Agent Signal feedback in Monopole's
existing session surfaces so they can be selected for full evaluation. Signal
feedback is high-recall triage evidence, not a Monopole judgment and not proof
that a session failed.

## Boundary

Monopole already reads turn spans from Weave and groups them by
`conversation_id`. The integration extends that same Sessions API pass: after
the visible session cohort is known, it batch-reads feedback attached to those
exact turn refs and adds low Signal evidence to the corresponding session
summaries.

There is no second hydration endpoint, CLI subprocess, JSON snapshot, service,
scheduler, database, or import from `signal_monitoring/`. The standalone system
remains separately installable and usable. Both systems read the same durable
Weave turn feedback, and conversation identity remains the existing
`conversation_id`.

## Eligible feedback

Only completed standalone Signal feedback is eligible:

- feedback type is `wandb.agent_monitor`;
- `runnable_ref` identifies a versioned scorer named
  `agent-signal-<slug>-<version>-scorer`;
- payload output contains one exact numeric anchor from `0.0`, `0.25`, `0.5`,
  `0.75`, or `1.0` plus a non-empty reason; and
- rating is at or below the catalog recommendation threshold of `0.5`.

This structural identity keeps Monopole independent from the standalone Python
package while avoiding accidental display of unrelated Agent Monitor results.
The Signal slug and version are derived from the scorer identity. Monopole does
not duplicate scorer prompts or resolve monitor objects.

Repeated feedback for the same Signal and turn resolves to the newest row by
score time and feedback ID, matching standalone hydration. Malformed eligible
feedback fails the Sessions request rather than silently appearing healthy.
Unknown feedback and ratings above the threshold are ignored.

## Session contract

Each returned session summary gains one always-present `signal_evidence` list.
Each item contains:

- Signal slug and version;
- numeric rating and grounded reason;
- triggering turn ID; and
- triggering turn start time.

Evidence is ordered by turn time, Signal, and turn ID. The frontend derives the
lowest rating and distinct Signal names from this list, so the API does not add
redundant aggregate fields. An empty list means the complete feedback read found
no eligible low Signal for the returned session.

Feedback is fetched only for sessions returned under the existing `limit`.
`total`, truncation, session ordering, date filtering, and selection behavior do
not change.

## Product behavior

The main Sessions page and evaluation-run session selector both show the same
compact callout when `signal_evidence` is non-empty:

```text
Needs review · 0.25 · user frustration, low quality response
```

The callout uses an amber treatment to mean "recommended for review," not a red
failure state. Signal slugs are rendered as short human-readable labels. The
lowest rating and all distinct Signal names remain visible without opening the
session.

Sessions remain selectable and are not auto-selected, hidden, re-ranked, or
automatically submitted for evaluation. Sessions without low Signal evidence
render exactly as before.

The first slice does not add new Session Detail behavior. That route already
hydrates turn feedback; richer per-turn Signal presentation can be added later
if the list-level triage proves useful.

## Failure behavior

Session discovery and Signal annotation form one complete response. An
incomplete or saturated feedback batch, malformed eligible feedback, or invalid
Signal identity fails the request and uses the existing retry UI. Monopole never
returns an unmarked session list that could be mistaken for a complete Signal
read.

Unrelated feedback types, unrelated Agent Monitor scorers, healthy Signal
ratings, and incomplete scorer attempts do not create callouts.

## Verification

Backend checks cover:

- strict recognition of versioned standalone scorer refs;
- `value` and `rating` payload aliases with exact anchor validation;
- low-threshold filtering;
- deterministic duplicate resolution;
- grouping evidence onto the existing conversation identity;
- stable evidence ordering;
- one batched exact-ref feedback read for the visible sessions; and
- fail-closed behavior for malformed eligible feedback.

Frontend checks cover:

- the same compact callout on Sessions and Run Selection;
- human-readable distinct Signal names and lowest rating;
- no callout for an empty evidence list; and
- unchanged navigation and selection behavior.

Tests use synthetic turns and feedback only. A live check confirms that a known
low-scoring turn appears as a callout on its existing Monopole conversation and
remains selectable for an evaluation run.

## Deferred work

- re-ranking or filtering the Sessions list by Signal severity;
- automatic session selection or evaluation-run creation;
- acknowledgement and review state;
- rich per-turn Signal UI on Session Detail; and
- session-level or trajectory Signals.

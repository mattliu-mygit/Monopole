# Evaluation runs

Evaluation runs make scoring, judging, reflection, review, and promotion
reproducible. They persist every important input boundary so the product can
show what was evaluated, what is proposed, and what will change.

## Pinned execution

The pipeline is linear:

```text
created -> scoring -> judging -> reflecting -> complete
```

`failed` and `cancelled` are terminal exits. Selection, configuration, and
automatic chaining are mutable only while a run is created. Starting requires
an explicit nonempty session selection.

Start atomically pins:

- the ordered turn cohort and its session identities;
- the resolved model and rubric catalogs, including each model's context
  capacity and token-counter identity;
- the proposal writer, one through three ordered run judges, and proposal evaluator;
- rubric versions, thresholds, judging context policy, candidate-attempt budget,
  and other effective configuration; and
- one pipeline version.

Later stages re-query only the pinned traces and restore their stored order.
Missing or identity-changed evidence fails the run. Newly recorded turns cannot
enter an active cohort. Before inference, judging separately pins complete raw
session coverage, a planned-or-skipped disposition for every selected reviewer
per session, every applicable reviewer's bounded window plan, and the exact
digest/window/merge protocol. Authentication recomputes applicability and the
window plan before external work rather than trusting a self-consistent stored
disposition. Reflection applies the same evidence-eligibility boundary as
analysis, then pins the exact eligible feedback records it consumes and an
exact baseline bundle before generating proposals. Audit-only judgment rows are
neither pinned as reflection evidence nor sent to proposal models.

Each stage has an explicit success marker distinct from the presence of a
partial result. Manual continuation requires that marker. Automatic handoff
persists success and the next status together.

After restart, compatible digest, window, merge, and reflection work may be
finalized or resumed from persisted evidence without repeating paid work.
Judging artifacts are reusable only after their content, phase, model, schema,
protocol, and exact plan identity are revalidated. Incomplete outcomes may
resume from valid earlier artifacts; incompatible or tampered artifacts fail
visibly instead of being adapted. Cancellation consults durable state,
terminates active model processes, and cannot interleave with protected
external write batches or finalized review evidence.

## Model roles and activity

Runs pin three independent roles:

1. the proposal writer generates candidate Markdown bundles;
2. run judges evaluate complete sessions through sliding raw windows and a final
   merge; and
3. the proposal evaluator predicts whole-bundle scores for Past B and candidate
   C revisions.

The form uses catalog-backed selectors with recommendations preselected and
user overrides allowed. The evaluator recommendation prefers a different model
family from the writer when available, but explicit compatible choices remain
authoritative and produce visible bias warnings rather than silent replacement.

Reflection records semantic activity rather than raw process logs. Its snapshot
contains phase, status, start time, configured attempt budget, monotonic counts
for attempted, valid, rejected, and evaluated proposals, and a bounded event
history. Events may identify role, model, attempt, evaluation, score, changed
paths, and safe failure evidence. Rejected writer output keeps a digest and a
bounded redacted structural excerpt, never proposal bodies, prompts, reasoning,
credentials, or arbitrary subprocess output.

Progress is observational. Failure to publish an activity update cannot change
candidate generation, evaluation, or selection.

Judging progress separately reports planned sessions, turns, raw windows,
applicable reviewer attempts, and unique completed digest, window, and merge
artifacts. Capacity-skipped reviewers remain visible with the sole skip reason
`insufficient_context_capacity`, but perform no inference, do not count as
failed or completed attempts, and have zero planned model work. If every
reviewer for a session is skipped, its rubrics are auditable not-evaluable
outcomes and the run continues without opening a model client for that work.
Reused artifacts retain ordered inference provenance without incrementing
unique completed-work counts.

## Managed instruction targets

`serve` and standalone `reflect` require one closed, versioned JSON target
registry. Exact file entries admit one Markdown file and may point inside or
outside Git. Skill collection entries admit only direct
`<validated-skill-name>/SKILL.md` children beneath one configured root; they do
not grant recursive or wildcard Markdown access. Markdown-root entries capture
and update only their explicit relative `files`. When `allow_create` is true,
they admit new recursive `.md` paths beneath the root.

The service exposes stable registry locators to models, not absolute paths.
Relative configured paths resolve from the registry document, while absolute
and tilde paths support global instruction files. Duplicate or overlapping
targets, path escape, non-Markdown exact files, invalid UTF-8, and symlinks at a
target or existing parent fail before model work. The registry manifest and
digest are pinned with reflection input.

A proposal may create or update admitted files. Before target publication, new
Markdown-root paths are added to the registry with an atomic compare-and-swap
write. Registry drift or registration failure blocks every target write. A
failure before any target is published restores registrations added by that
attempt so the pending review can retry against the same B. A partial terminal
promotion may retain an explicitly registered absent target. Delete, archive,
rename, and broad update discovery are not supported.

## Reflection

The proposal writer and evaluator are separate pinned models. A run allows one
through ten proposal attempts, with three by default. Invalid proposals are
recorded and may be followed by another attempt. Reflection stops early after
two consecutive scored attempts fail to improve the best predicted evaluator
score; rejected or duplicate revisions do not falsely count as scored
non-improvements.

Every proposal is a complete action set against B, not an unstructured patch.
The writer must return valid create or update actions for registry locators,
and a no-op proposal is rejected. Candidate provenance must map
unambiguously to a successful generation attempt.

The proposal evaluator scores exact immutable bundle revisions. Its output is a
predicted artifact-quality score over the pinned evaluation feedback, not a
verification run. Evaluation failure or an invalid score cannot become scored
evidence. The baseline and each candidate retain evaluator identity, score, and
rationale; generation attempts retain writer identity, outcome, usage, changed
paths, and safe rejection evidence.

The recommended revision is recomputed from persisted scores. Ties keep the
earliest revision, so B wins a tie. When eligible feedback or managed targets
are empty, reflection performs no model calls and records why it did not run.
Valid deterministic scores are eligible; model judgments are eligible only as
complete current session judgments with fully identified comparison context.
When every proposal is invalid, the product still shows evaluated B and the
failed-attempt audit. When B beats every valid C, alternatives remain read-only
evidence and no promotion review is created.

## B, C, and D

- **B — Past** is the exact evaluated baseline and owns its score.
- **C — Proposed** is an exact immutable generated bundle and owns its score,
  rationale, B-to-C delta, writer provenance, and evaluator provenance.
- **D — Edited proposal** is optional user-edited content derived from selected
  C. It is not evaluated and never inherits C's score.

A score belongs to a complete bundle revision, not an individual file. The
review keeps relevant Markdown snapshots and diffs beside the whole-bundle
scores so a user can audit interactions across multiple changed files.

D can alter file contents but must preserve C's target membership and
create/update action set. Promoting D requires explicit acknowledgement
that it was not evaluated. There is no mandatory pre-promotion inference run:
the user accepts the deviation and can start a later evaluation run after
promotion.

Review decisions use compare-and-swap revisions. Candidate selection, draft
save/reset, promotion, and dismissal cannot overwrite a concurrent or already
resolved decision. Evidence is immutable once review begins.

## Drift and promotion

The only product-level blocking promotion precheck is whole-scope drift from B.
Before candidate selection, draft mutation, or promotion, the registry compares
the complete live target scope—including skill membership—with evaluated B. Any
change makes the review stale. Detail view then shows B, C, optional D, the
changed locators, and Current when it can be captured; editing and promotion
remain disabled and a new run is required. Dismissal remains available because
it does not mutate managed files.

The local service serializes promotion for one loaded registry. Promotion
validates every action against B before writing, stages every complete UTF-8
file on the destination filesystem, and then rechecks each target immediately
before publication. Updates use atomic replacement; creates publish without
overwriting an independently created destination. No file is exposed with
partial contents.

Files are applied independently in deterministic order. A later drift or write
failure stops the remaining actions but does not roll back earlier complete
files. An all-applied receipt resolves the review as `promoted`; a mixed receipt
resolves it as terminal `partial`; failure before the first write leaves the
review pending. There is no filesystem journal, backup transaction, restart
recovery, or multi-file rollback.

The receipt anchors exact B, selected C, requested C-or-D, the review revision,
and an applied/not-applied outcome for every file. A repeated decision is
idempotent only after its receipt is persisted. If persistence fails after a
write, later reads report ordinary source drift rather than inferring or
recovering an unrecorded transaction. Git is not required.

## HTTP contract

FastAPI response models are the authority for transported run, review, catalog,
inspection, and analysis data. The repository deterministically exports OpenAPI
and generates the frontend transport types into `frontend/src/generated/`.
Generated files are never edited by hand, and the freshness check fails when
backend models and committed TypeScript output differ. Handwritten TypeScript
types remain only where the UI needs richer local view state.

## List and detail views

The run list is a compact persisted summary used for polling. It reads only
scalar status, selection, success, and review-summary fields and does not load
full evidence or inspect the live repository.

Run detail is the authoritative full evidence view. It derives live drift by
comparing the current registry scope with B and displays exact per-file receipt
outcomes.
This separation keeps list polling cheap without weakening promotion safety.

The local run database is disposable pre-release state. The product supports
the current schema only; it does not migrate or adapt legacy rows.

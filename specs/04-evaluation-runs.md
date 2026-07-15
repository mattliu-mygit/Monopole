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
- the resolved model and rubric catalogs;
- the proposal writer, ordered run judges, review depth, and proposal evaluator;
- rubric versions, thresholds, judging context policy, candidate-attempt budget,
  and other effective configuration; and
- one pipeline version.

Later stages re-query only the pinned traces and restore their stored order.
Missing or identity-changed evidence fails the run. Newly recorded turns cannot
enter an active cohort. Before inference, judging separately pins complete raw
session coverage, each reviewer's bounded window plan, and the exact
digest/window/merge protocol. Reflection pins the exact feedback records it
consumes and an exact baseline bundle before generating proposals.

Each stage has an explicit success marker distinct from the presence of a
partial result. Manual continuation requires that marker. Automatic handoff
persists success and the next status together.

After restart, compatible digest, window, merge, reflection, and promotion work
may be finalized or resumed from persisted evidence without repeating paid or
destructive work. Judging artifacts are reusable only after their content,
phase, model, schema, protocol, and exact plan identity are revalidated.
Incomplete outcomes may resume from valid earlier artifacts; incompatible or
tampered artifacts fail visibly instead of being adapted. Cancellation consults
durable state, terminates active model processes, and cannot interleave with
protected external write batches or finalized review evidence.

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
reviewer-attempt bounds, and unique completed digest, window, and merge
artifacts. The maximums are pinned work bounds rather than promises that
selective review will invoke every configured reviewer. Reused artifacts retain
ordered inference provenance without incrementing unique completed-work counts.

## Repository Markdown scope

The current project-file adapter manages lowercase `.md` files recursively
from the repository root, including root-level Markdown. It skips named VCS,
product-state, temporary-planning, virtual-environment, dependency, cache,
coverage, and build directories.

The scope is deliberately bounded:

- at most 500 files;
- at most 256 KiB per file;
- at most 512 KiB across the captured bundle;
- UTF-8 regular files only; and
- paths represented as sorted repository-relative POSIX locators.

Non-excluded symlinks, cross-filesystem entries, unreadable files or
directories, case-folding collisions, non-file collisions at Markdown paths,
and ambiguous or escaping locators fail closed. The exact adapter contract and
scope policy are pinned with reflection input so later review does not reinterpret
what B contained.

A proposal may create, update, or delete several managed Markdown files. The
same policy governs baseline capture, proposal validation, drift detection, and
promotion.

## Reflection

The proposal writer and evaluator are separate pinned models. A run allows one
through ten proposal attempts, with three by default. Invalid proposals are
recorded and may be followed by another attempt. Reflection stops early after
two consecutive scored attempts fail to improve the best predicted evaluator
score; rejected or duplicate revisions do not falsely count as scored
non-improvements.

Every proposal is a complete action set against B, not an unstructured patch.
The writer must return valid create, update, or delete actions for managed
Markdown paths, and a no-op proposal is rejected. Candidate provenance must map
unambiguously to a successful generation attempt.

The proposal evaluator scores exact immutable bundle revisions. Its output is a
predicted artifact-quality score over the pinned evaluation feedback, not a
verification run. Evaluation failure or an invalid score cannot become scored
evidence. The baseline and each candidate retain evaluator identity, score, and
rationale; generation attempts retain writer identity, outcome, usage, changed
paths, and safe rejection evidence.

The recommended revision is recomputed from persisted scores. Ties keep the
earliest revision, so B wins a tie. When feedback or managed targets are empty,
reflection performs no model calls and records why it did not run. When every
proposal is invalid, the product still shows evaluated B and the failed-attempt
audit. When B beats every valid C, alternatives remain read-only evidence and no
promotion review is created.

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
create/update/delete action set. Promoting D requires explicit acknowledgement
that it was not evaluated. There is no mandatory pre-promotion inference run:
the user accepts the deviation and can start a later evaluation run after
promotion.

Review decisions use compare-and-swap revisions. Candidate selection, draft
save/reset, promotion, and dismissal cannot overwrite a concurrent or already
resolved decision. Evidence is immutable once review begins.

## Drift and promotion

The only product-level blocking promotion precheck is whole-scope drift from B.
Before candidate selection, draft mutation, or promotion, the adapter compares
the complete live Markdown scope—including membership—with evaluated B. Any
change makes the review stale. Detail view then shows B, C, optional D, the
changed locators, and Current when it can be captured; editing and promotion
remain disabled and a new run is required. Dismissal remains available because
it does not mutate managed files.

Promotion applies the approved multi-file bundle as one locked, journaled
transaction. Its durability contract is:

- validate the complete decision and live baseline before mutation;
- stage durable new contents and backups;
- recheck each live target immediately before its change;
- durably apply every create, update, or delete;
- persist an immutable receipt; and
- roll back safely on partial failure without overwriting later user edits.

Restart recovery handles the narrow window between committed files and receipt
persistence. Promotion IDs are idempotency keys: an identical repeated success
returns its receipt, while a different later decision is rejected.

The receipt anchors exact B, selected C, promoted C-or-D, every file action,
whether promoted content was evaluated, the D acknowledgement, and the review
revision that authorized it. Git metadata is supporting context, not the source
of truth, because managed instructions may live outside a useful Git workflow.

## List and detail views

The run list is a compact persisted summary used for polling. It reads only
scalar status, selection, success, and review-summary fields and does not load
full evidence or inspect the live repository.

Run detail is the authoritative full evidence view. It can recover a committed
receipt and derives live drift by comparing the current Markdown scope with B.
This separation keeps list polling cheap without weakening promotion safety.

The local run database is disposable pre-release state. The product supports
the current schema only; it does not migrate or adapt legacy rows.

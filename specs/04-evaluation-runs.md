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

New-run setup remains an unpersisted browser draft until the user starts
scoring. Starting creates the run, saves its requested inputs, and advances it;
opening the setup form alone does not add a run. Users may permanently delete
created, complete, failed, or cancelled runs and their owned audit records after
explicit confirmation. Active scoring, judging, or reflecting runs must be
cancelled before deletion so workers cannot write to a removed run.

Start atomically pins:

- the ordered turn cohort and its session identities;
- the evaluated model family, model ID, and effort level for every pinned turn;
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
Judge calls are reusable only when their request identity binds the exact model,
messages, response schema, options, and protocol, and their validated result is
still present. Incomplete outcomes may resume from valid earlier calls;
incompatible or tampered records fail
visibly instead of being adapted. Cancellation consults durable state,
terminates active model processes, and cannot interleave with protected
external write batches or finalized review evidence.

## Model roles and activity

Runs pin four independent roles:

1. the proposal writer generates candidate Markdown bundles;
2. run judges evaluate complete sessions through sliding raw windows and a final
   merge;
3. the proposal evaluator predicts whole-bundle scores for Past A and candidate
   revisions, then authors one blinded verification task package for the
   provisional B; and
4. a separately pinned A/B verification panel compares A and the provisional B
   on that task.

The form uses catalog-backed selectors with recommendations preselected and
user overrides allowed. Normal judging recommends three available reviewers;
A/B verification independently recommends one and permits one through three.
The evaluator recommendation prefers a different model family from the writer
when available, but explicit compatible choices remain authoritative and
produce visible bias warnings rather than silent replacement.

Reflection records semantic activity rather than raw process logs. Its progress
contains phase, status, start time, configured attempt budget, and monotonic
counts for attempted, valid, rejected, and evaluated proposals. Shared bounded
event rows may identify role, model, attempt, evaluation, score, changed
paths, and safe failure evidence. Rejected writer output keeps a digest and a
bounded redacted structural excerpt, never proposal bodies, prompts, reasoning,
credentials, or arbitrary subprocess output.

Progress is observational. Failure to publish an activity update cannot change
judging, candidate generation, evaluation, or selection.

Judging progress separately reports planned sessions, turns, raw windows,
applicable reviewer attempts, and unique completed digest, window, and merge
calls. Capacity-skipped reviewers remain visible with the sole skip reason
`insufficient_context_capacity`, but perform no inference, do not count as
failed or completed attempts, and have zero planned model work. If every
reviewer for a session is skipped, its rubrics are auditable not-evaluable
outcomes and the run continues without opening a model client for that work.
Reused calls retain ordered inference provenance without incrementing
unique completed-work counts.

Scoring, judging, and reflection append to one bounded chronological event table
and retain one current plain-language stage status. Semantic events identify session, rubric, reviewer,
digest, window, merge, and feedback-write work. Local CLI transport events make
retries, recovery, and terminal request failures visible with safe diagnostic
metadata: model, request attempt, elapsed time, categorical reason, exit code,
prompt and output sizes, estimated input and context limits, output mode, and a
content hash. Exact prompt echoes are removed before failure classification and
hashing. A recognized provider error envelope may additionally contribute only
its HTTP status, error code, and a bounded redacted message. Prompts, raw model
output, credentials, and unstructured provider output are never stored in
progress. The product emphasizes the current operation, failures, retries, and
recoveries, including digest, window, and merge boundaries. Low-level transport
attempts and cache-reuse events remain available through a debug-event control
across refreshes.

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

Each registry target may include a concise model-facing description of its
purpose and when instructions in that target should be updated or created. The
pinned proposal scope presents that description together with the target kind,
stable ID, allowed actions, and locator format. It never exposes filesystem
paths. These profiles help the proposal writer choose the right managed target;
the registry remains authoritative and rejects any unadmitted action or locator.

A proposal may create or update admitted files. Before target publication, new
Markdown-root paths are added to the registry with an atomic compare-and-swap
write. Registry drift or registration failure blocks every target write. A
failure before any target is published restores registrations added by that
attempt so the pending review can retry against the same A. A partial terminal
promotion may retain an explicitly registered absent target. Delete, archive,
rename, and broad update discovery are not supported.

## Reflection

The proposal writer and evaluator are separate pinned models. A run allows one
through ten proposal attempts, with three by default. Invalid proposals are
recorded and may be followed by another attempt. Reflection stops early after
two consecutive scored attempts fail to improve the best predicted evaluator
score; rejected or duplicate revisions do not falsely count as scored
non-improvements.

Writer, evaluator, run-judge, and A/B-judge choices all resolve through one
versioned model catalog. A model's provider-qualified ID is the durable
selection and audit identity; its pinned provider and exact provider model name
determine whether the call uses Claude, Codex, Antigravity, W&B Inference, or
OpenAI. Roles may independently select models from different providers.

Every proposal is a complete action set against A, not an unstructured patch.
The writer must return valid create or update actions for registry locators,
and a no-op proposal is rejected. Candidate provenance must map
unambiguously to a successful generation attempt.
All structured writer, evaluator, task-author, and paired-judge requests include
their exact schema and schema-owned canonical examples. A second example is
included only when the contract has a materially different valid mode.

The proposal evaluator scores exact immutable bundle revisions. Its output is a
predicted artifact-quality score over the pinned evaluation feedback, not a
verification run. Evaluation failure or an invalid score cannot become scored
evidence. The baseline and each candidate retain evaluator identity, score, and
rationale; generation attempts retain writer identity, outcome, usage, changed
paths, and safe rejection evidence.

The predicted scores choose one provisional B. A provisional B that strictly
beats A remains available for human review even when paired verification is
incomplete or invalid. Ties keep the earliest revision, so A wins a
predicted-score tie.
When eligible feedback or managed targets are empty, reflection performs no
model calls and records why it did not run.
Valid deterministic scores are eligible; model judgments are eligible only as
complete current session judgments with fully identified comparison context.
When every proposal is invalid, the product still shows evaluated A and the
failed-attempt audit. When A beats every valid B, alternatives remain read-only
evidence and no promotion review is created. When B wins the predicted
comparison, the product creates a pending review for that exact provisional B
regardless of the paired-verification outcome.

## Paired sandbox verification

After a provisional B is selected, the proposal evaluator acts as a
blinded material planner and task author. A first structured call receives the
pinned bounded evaluation digest and a compact deterministic summary of the
seed workspace, with managed instruction contents removed, and selects a setup
mode plus pinned public materials for a fresh analogous task. After those
materials are fetched, a second structured call receives their bounded path
manifest and authors the prompt, measurable goal, task-specific judging
criteria, descriptive starting-state checks, and declarative file, executable,
and Git-metadata prerequisites. Neither call sees an instruction bundle, arm
identity, or evaluated agent identity. The generated task is grounded in the
currently fetched revision; it does not claim to replay the historical task or
source tree exactly.

Public source preparation is deliberately narrow. A material may name an HTTPS
Git repository on GitHub, GitLab, Bitbucket, or Codeberg and a safe relative
destination. Trusted service code resolves the repository's advertised `HEAD`
to a full commit SHA before constructing the pinned material plan; the model
does not supply revisions. Model-authored host shell commands, credentials,
private sources, and moving branches are rejected. A repository that cannot be
resolved anonymously is rejected and returned to the planner for a different
public-only analogous material choice, for at most three complete planning
attempts. The service fetches each pinned revision once into a temporary checkout
and excludes Git metadata before task authoring. In `prepared_workspace` mode,
it overlays the fetched files on the common execution workspace before forking
A and B. In `agent_bootstrap` mode, fetched files are authoring context only;
both agents receive the same pinned material specification and must perform the
requested clone or other repository setup themselves. A prepared task cannot
require Git metadata. Exact required files are checked against the common
initial snapshot, and one disposable sandbox verifies the pinned harness and
required executable names before either arm starts. Fetch, file-preflight, or
capability-preflight failure invalidates the task without running A or B. The
product does not choose a different harness by task category; coding and
research tasks use the same configured runtime boundary. Author-supplied
starting-state checks remain task and judge context; they are not executed as
arbitrary host commands.

The service creates two concurrent local Smol Machines microVMs from one
content-addressed OCI image or digest-authenticated local `.smolmachine`
artifact. Each receives a private copy of the same prepared or bootstrap
workspace snapshot, parent environment, declared config and credential files,
network policy, timeout, and command. The image digest pins the remaining
toolchain and filesystem. A gets
the captured managed Markdown; B gets exactly the provisional bundle changes.
Managed paths beneath the configured workspace remain in `/workspace`;
managed paths beneath the host home are mirrored at the corresponding guest
home path. Materialization proves that these declared Markdown paths are the
only initial differences. There are no live host mounts.

The evaluated turn cohort must identify one model, family, and effort level.
That identity selects the Codex or Claude CLI command; the guest CLI version
must equal the pinned host version before either task counts. The immutable
runtime image supplies the CLI, repository toolchain, and excluded dependency
trees. Explicit runtime files reproduce local CLI configuration or stored
credentials without persisting their contents. Every inherited environment
value and runtime file is named and content-hashed in the execution identity.

Both agent processes receive the same task package and run until they exit or
reach the pinned timeout. The example runtime defaults to 1,800 seconds, while
the deployed runtime may pin another positive timeout for longer experiments.
A nonzero exit, timeout, or unsuccessful task is
behavioral evidence. VM creation, image, workspace setup, harness preflight, or
artifact-extraction failure makes the pair incomplete and is not scored as an
agent failure. Workspace file count, per-file bytes, total bytes, archive bytes,
process-output capture, and persisted text diffs are bounded; exceeding a
sandbox evidence bound fails infrastructure closed rather than exhausting the
service. Dependency and cache trees excluded from the frozen source snapshot
are also excluded from final artifact capture in both arms.

The separately configured ordered panel of one through three A/B judges then
receives blinded arm labels, the task package, bounded transcripts, exit
outcomes, artifact counts, bounded changed-file diffs, and the exact run
rubrics. Managed instruction
paths, shortened path forms, and contents are excluded from artifact counts,
transcript evidence, and diff evidence so they cannot reveal which blinded arm
received A or B. The complete request, including its schema, is token-counted
against each configured judge and evidence is reduced
deterministically before inference when necessary. Each judge first decides
whether the task was achievable, workspace-applicable, and fair; an invalid-task
panel majority fails closed without selecting either arm. The A/B catalog
recommendation is one judge by default for cost, independent of the three-judge
normal evaluation recommendation; users may explicitly choose two or three. A
multi-model A/B panel receives the same family-diversity and evaluated-family
bias warnings as the normal judge panel. A
strict panel majority selects A or B; otherwise the result is a tie. Each panel
member uses the backend pinned by its own selected model descriptor. A complete
B win marks the provisional candidate as sandbox verified. An A win, tie,
invalid task, or incomplete comparison does not erase the predicted result or
make B read-only; it marks B unverified. Promoting an unverified B requires an
explicit human acknowledgement. The task, execution identity, arm evidence,
blinded verdicts, per-rubric A/B scores, signed deltas, transcripts, and
changed-file diffs remain attached to the immutable reflection result and are
lazily inspectable in review.

## A, B, and C

- **A — Past** is the exact evaluated baseline and owns its score.
- **B — Proposed** is an exact immutable generated bundle and owns its predicted
  score, rationale, A-to-B delta, writer provenance, evaluator provenance, and
  paired sandbox evidence when it is the provisional candidate.
- **C — Edited proposal** is optional user-edited content derived from selected
  B. It is not evaluated and never inherits B's score.

A score belongs to a complete bundle revision, not an individual file. The
review keeps relevant Markdown snapshots and diffs beside the whole-bundle
scores so a user can audit interactions across multiple changed files.

For the selected proposal, review presents one affected-file selector above a
two-column workspace. The left column is an editable monospaced B-or-C editor
with line numbers and a total line count. The right column is a live unified
A-to-C diff, which is equivalent to A-to-B before the user edits. Editing
starts from the exact selected B or its saved C draft;
save and reset retain the existing whole-bundle draft contract.

C can alter file contents but must preserve B's target membership and
create/update action set. Promoting C requires explicit acknowledgement that C
differs from evaluated B and did not itself receive predicted evaluation or
paired verification. There is no second pre-promotion run for edited C. The
user accepts the deviation and can start a later evaluation run after
promotion.

An unverified B and an edited C are distinct conditions. Promotion requires a
separate acknowledgement for each condition that applies. The UI describes the
predicted and paired outcomes independently; it never states that A scored best
when B won the predicted comparison but verification retained no winner.

Review decisions use compare-and-swap revisions. Candidate selection, draft
save/reset, promotion, and dismissal cannot overwrite a concurrent or already
resolved decision. Evidence is immutable once review begins.

## Drift and promotion

The only product-level blocking promotion precheck is whole-scope drift from A.
Before candidate selection, draft mutation, or promotion, the registry compares
the complete live target scope—including skill membership—with evaluated A. Any
change makes the review stale. Detail view then shows A, B, optional C, the
changed locators, and Current when it can be captured; editing and promotion
remain disabled and a new run is required. Dismissal remains available because
it does not mutate managed files.

The local service serializes promotion for one loaded registry. Promotion
validates every action against A before writing, stages every complete UTF-8
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

The receipt anchors exact A, selected B, requested B-or-C, the review revision,
the paired-verification status and challenge identity or failure reason, both
human acknowledgement decisions, and an applied/not-applied outcome for every
file. A repeated decision is idempotent only after its receipt is persisted. If
persistence fails after a write, later reads report ordinary source drift
rather than inferring or recovering an unrecorded transaction. Git is not
required.

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
comparing the current registry scope with A and displays exact per-file receipt
outcomes.
This separation keeps list polling cheap without weakening promotion safety.

Runtime reads only schema epoch 10. Startup performs the one supported epoch-9
preservation migration after creating a timestamped SQLite backup; imported
judge audits remain presentation-only and are never reusable. Other schema
epochs fail closed instead of being reset or adapted.

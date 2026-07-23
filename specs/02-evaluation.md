# Evaluation

Monopole combines deterministic evaluation with model judgment. Deterministic
scorers cover evidence that can be interpreted with high precision. Model
judges cover process and session qualities that require contextual assessment.
Both write ratings on the same higher-is-better `[0, 1]` scale.

Ordinary evaluation applies only to complete `agent_session` conversations.
Signal scorer, judge, reflection, other-system, and mixed-role traces are
operational evidence, not members of the agent-quality population. Discovery
enforces that boundary before child hydration, inference, cohort pinning, or
feedback writes. Rehydrating a pinned run also rejects role drift before
external work.

## Deterministic evaluation

Outcome evaluation recognizes observed test, build, lint, installation,
mutating Git, and definite generic-command failures. Explicit numeric exit codes
are authoritative. An explicit null exit code is unknown; framework output or
span status is used only when it can establish an outcome safely. Unknown
evidence produces no invented success or failure.

Process evaluation produces:

- turn and session efficiency, based on observable repeated failing attempts and
  repeated reads without intervening edits;
- session correction-free rate, the complement of the fraction of turns with a
  steering or denial event; and
- session completion, the complement of a conservative abandonment signal.

Raw correction density, abandonment, token counts, duration, tool counts, and
model tier remain explanatory context rather than negative-direction ratings.

Deterministic parsers favor precision over coverage. They inspect only retained
tool evidence and do not claim intent, semantic code quality, test quality, or
causal attribution for all work that led to an observed result.

## Planned model evaluation

Before inference, an evaluation run derives and pins a versioned judging plan
from its immutable cohort. The plan pins the context policy, ordered model
descriptors including context capacity and token-counter identity, rubric
descriptors, complete session trace identities, each reviewer's planned-or-
skipped disposition, reviewer-specific window manifests, and the exact
digest/window/merge protocol. Changes to any of those inputs produce a different
plan or request identity. Authentication recomputes each reviewer's
applicability and exact window plan from the pinned session, descriptor, and
policy before external work.

All model rubrics are session-level. Each applicable reviewer gets a window plan
sized to its pinned context capacity. Every model reserves the pinned prompt,
output, and safety allowance; surrounding-digest and finding overhead increase
that reserve as the chunk count grows. At the 200,000-token context threshold,
the generation allowance is 4,000 tokens at or below the threshold and 10,000
tokens above it. This API allowance is independent of the smaller validated
artifact limits. Raw windows have separate soft targets of 128,000 tokens above
the same model threshold and 50,000 tokens at or below it. A model catalog
entry may lower that target when its inference harness has a smaller reliable
operational window; this does not change the model's advertised context
capacity. OpenAI catalog
entries use their explicitly pinned `tiktoken` encoding; other model families
use the conservative UTF-8 byte estimator.

The hard raw budget remains a capacity ceiling. Contiguous core chunks are
packed near the soft target, assigning a boundary-crossing turn to the side
closer to that target, so every session turn has exact core coverage. Each raw
window includes up to one neighboring turn on either side when it remains
within the soft target; overlap can shrink to zero without removing core
coverage. Turn boundaries remain indivisible, and a single oversized turn may
exceed the soft target only when it still fits the hard budget. Before planning,
the judge view deterministically compacts only oversized tool arguments and
results, preserving their head, diagnostic excerpts, tail, original character
count, and content hash. Captured user and assistant messages are not compacted. When an intact
projected turn, the required chunk count, or the worst-case merge cannot fit,
that reviewer alone is skipped for the session with
`insufficient_context_capacity`.

For each reviewer, the pipeline first creates one rubric-neutral factual digest
per core chunk. A rubric evaluation then slides across every window: the active
chunk is supplied as raw captured evidence while the rest of the session is
represented by that reviewer's ordered digests. The host replaces the active
raw evidence IDs with short, window-local aliases before inference and resolves
valid citations back to canonical trace identities. Surrounding digest
provenance remains host-owned and is represented by the chunk ID rather than
model-authored citations. Every window returns a bounded set of
positive or negative findings, not a score. Findings are deduplicated in session
order, then one final merge sees the coverage manifest, ordered digests, and
findings and returns the reviewer's anchored session verdict and behavioral
feedback. The merge cites finding IDs; the host derives the final canonical
trace citations from those immutable findings.

W&B digest and verdict-merge requests disable optional model reasoning through
the provider's documented request setting. Those phases compress or consolidate
the substantive rubric analysis already performed in the raw-evidence window
reviews, so spending their output budget on hidden reasoning reduces reliability.
Window review retains each model's default reasoning behavior on its first
attempt and asks for careful internal reasoning followed by concise JSON. W&B
retries use JSON-object mode with optional thinking disabled and receive the
same bound schema and canonical examples in the prompt; canonical runtime
validation remains unchanged. Every structured judge request receives one
schema-owned example, plus a second only for a materially different valid mode;
validation correction repeats that exact contract.

Raw rendering includes captured user, assistant, tool, event, status, token, and
trace-identity evidence while omitting the generating model's identity from the
judge-facing content. Parent-session judging retains each subagent's identity
and internal tool-call count but omits its internal tool payloads; the parent's
delegation call still carries the input and returned output that influenced the
parent. The complete hydrated trace remains available outside this judge view.
Digests, findings, verdicts, and behavioral feedback are schema-bounded. Models
author semantic fields plus short citation references only where needed. The
host adds protocol, chunk, and window identity and resolves aliases before using
the result. Window and merge schemas enumerate only the references authorized
for that phase; the same constraints are validated again after inference. Complete
window-finding artifacts receive a 10,000-token budget. Digest prompts request a
soft target of about 1,000 tokens, but digest text is not rejected for crossing
that estimate. Actual inference calls still enforce model context capacity.
Judge-authored semantic text has no character-count limits.

The current session rubrics evaluate verification, error recovery, tool choice,
state consistency, outcome, and autonomy. A rubric may return
`insufficient_evidence` when its required opportunity is absent even though the
session itself has complete raw coverage. That abstention remains distinct from
missing, malformed, or unauthenticated pipeline evidence.

## Verdict contract

Every reviewer's final merge returns the closed versioned verdict contract:

- status is `scored` or `insufficient_evidence`;
- a scored verdict uses exactly one anchor from
  `0, 0.25, 0.5, 0.75, 1`;
- an insufficient-evidence verdict has no score;
- rationale is required;
- a scored verdict cites at least one supplied finding, from which the host
  derives canonical session evidence IDs, and
  includes at least one nonblank success, problem, or desired-behavior field;
  and
- an insufficient-evidence verdict has no citations or behavioral feedback.

`insufficient_evidence` is authoritative: transport output is canonicalized to
the scoreless, citation-free, feedback-free abstention before persistence.

Chunk provenance is the host-authenticated chunk ID; digest models do not repeat
source IDs. Window findings cite only short aliases visible in that raw window,
are capped in count and by the complete artifact token budget, and receive
canonical identities from their response order. Aggregation scopes each identity
to its authenticated window after deduplicating exact semantic findings in
response order, then deduplicates again across windows in session order. A
finding may include a short exact quote; the host retains it only when it occurs
in the active raw window. An unknown citation, blank required text, or oversized
artifact fails closed. The
unknown-citation diagnostic reports at most three escaped, bounded identifiers
without retaining the invalid response. The merge prompt asks for feedback
about what the agent did or should do, while
reflection separately decides whether and how to edit managed instructions.
Imperfect semantic wording is not itself a validation failure.

Boolean, missing, nonnumeric, non-finite, out-of-range, non-anchor, malformed,
or improperly cited output is rejected. Scores are never clamped or coerced.
Abstention remains distinct from transport or validation failure. Successful
reviewer anchors are mean-pooled, so a final multi-reviewer rating may be any
finite value in `[0, 1]`.

Structured output is requested when the backend supports it. A recorded JSON
object fallback is allowed when the provider explicitly rejects structured
schema capability or on a W&B recovery attempt. Invalid model content does not
weaken canonical runtime validation. W&B retries preserve evidence constraints
while using JSON-object mode with optional thinking disabled;
other backends retry the same strict request. Invalid JSON that consumes the
exact generation allowance retries once under the same budget. A second
exhaustion fails closed as explicit
output-limit exhaustion while retaining both attempts' usage and request counts.
After a W&B model and schema pair requires recovery, later calls for that pair
within the same run start in concise JSON recovery mode instead of repeating the
known-expensive failing path. Exact schema prompting and runtime validation are
unchanged.
If returned JSON parses but violates the canonical output contract, regardless
of structured or fallback output mode, one or two correction calls receive the
bounded validation error and prior object.
Each corrected object is validated normally; a third violation fails closed.
Backend-specific schema emission may omit a keyword that the backend
rejects when the same invariant remains enforced by the canonical runtime
validator; the canonical schema and artifact contract are not weakened.
Antigravity's prompt-only CLI has no protocol-level schema channel, so its
prompt places the schema before the potentially large user payload. A single
Markdown JSON fence around the whole response is treated as an Antigravity
transport envelope and removed before exact-object parsing; prose, multiple
objects, and invalid content remain rejected. A successful Antigravity process
that omits top-level required fields retries the same strict request in a fresh
workspace and fails closed after the bounded transport attempts.

## Judge panel

Runs use guided model catalogs with recommended selections, but the saved
ordered choices are explicit and user-overridable. Start-time validation pins
the model and rubric catalog versions, exact descriptors, rubric thresholds,
the ordered panel, and visible family-overlap warnings. A panel contains one,
two, or three unique judges. The runtime honors the chosen order, plans or skips
each selected judge per session, and never silently replaces one.

One versioned catalog is authoritative for proposal writers, run judges, A/B
verification judges, and proposal evaluators. Each globally unique model ID is
provider-qualified, while its descriptor separately pins the provider's exact
model name. Role capability is part of the descriptor. A panel may therefore
mix providers without a separate backend selection, and execution dispatches
each selected model through its pinned provider.

Model selection keeps separate provider-qualified entries when multiple
inference harnesses expose the same underlying model. W&B Inference entries are
ordered and recommended ahead of local CLI variants; fitting local and direct
API variants remain explicit alternatives. Every option shows its pinned input
context capacity and, once sessions are selected, whether the model can support
the estimated judging requests. A non-fitting option is disabled and a selected
configuration that becomes non-fitting cannot start.

The W&B entries cover its generally available Serverless Inference catalog and
exclude models that W&B marks deprecated. Their provider IDs, families, context
capacities, and role capabilities are maintained in the versioned local catalog:
W&B's live models endpoint reports accessible IDs but not the metadata required
to pin a reproducible run. Catalog construction therefore remains deterministic
and offline; explicit catalog updates track W&B lifecycle changes.

Selection-time capacity estimation hydrates only the selected sessions, renders
their compact judging evidence once, and counts it with each catalog token-counter
identity. For each selected session and model, the largest rendered turn estimates the minimum indivisible
raw chunk, while the total rendered evidence estimates only the number of chunks
and therefore the digest and merge buffers. The estimator uses the pinned context
policy's raw-window target, prompt/output/safety reserve, digest limit, finding
limit, and maximum chunk count. Its estimated largest request is the greater of
the largest raw-window request and the final merge request; that value must not
exceed the descriptor's input context capacity. The runtime still derives and
authenticates the exact rendered-turn plan before inference. Borderline exact
requests are attempted. An explicit provider context-limit rejection becomes an
`insufficient_context_capacity` skip for that reviewer; authentication, rate,
schema, and other provider errors remain failures.

Only planned reviewers call inference. A capacity skip remains in the judging
plan and attempt audit, but does not count as a completed reviewer attempt or an
inference failure. Within one session, rubric panels and their one through three
selected reviewers are eligible to run concurrently. Each reviewer still
creates one rubric-neutral digest set and shares it across that reviewer's
rubrics. Live provider calls are bounded independently of cached work: W&B
Inference permits four active calls and Codex permits two. Sessions remain
sequential, and outcomes are aggregated in pinned rubric and reviewer order so
scheduling cannot change persisted results. A scored rubric is complete when
every selected reviewer returns a valid score. It is `degraded` when at least
one reviewer scores and the rest either validly abstain or were skipped; its
rating is the arithmetic mean of available scores, with minimum, maximum,
spread, abstentions, and skips retained for audit. When no reviewer is
applicable, or every applicable reviewer validly abstains, the rubric is not
evaluable and writes no feedback.

A failed invocation or invalid attempt by an applicable reviewer still fails
coverage even when another reviewer returned a valid score. After the bounded
transport retries for that attempt are exhausted, the runtime cancels pending
work across the session. Active local CLI processes are terminated; an active
HTTP request may settle, but cannot start another digest, window, or merge.
Concurrent outcomes that settled before cancellation remain ordered by pinned
rubric and reviewer identity. Any coverage failure then fails the session and
stops remaining sessions.
Explicit run cancellation remains global and interrupts provider-gate waits.
The HTTP transport makes at most three attempts for retryable timeouts, rate
limits, and transient server failures, with backoff, and permits a 300-second
response wait per attempt. Every provider request start and completion plus
retry, recovery, and terminal transport events are emitted as they happen. They
identify the phase or chunk, attempt count, per-attempt duration, safe error
category, HTTP status when available, and bounded token or finish diagnostics
for output exhaustion. The run UI derives each model's average request time from
those completion events. Digest, window, merge, retry, recovery, and terminal
events are visible normally; low-level request-attempt and cache-reuse events
require debug events. Output is represented only by its digest. The terminal
panel event leads with the first safe provider error and
summarizes each reviewer's score, abstention, skip, or failure and
provider-request count.

Every attempt retains its requested and resolved model, outcome, rationale or
safe error, evidence citations, structured-output mode, usage, bounded
behavioral feedback, and ordered digest/window/merge step audit. Scores are
buffered until every session rubric resolves to a complete or degraded rating,
or a unanimous not-evaluable outcome. Only the resulting ratings are written
under the run's cancellation barrier.

## Judge-call resume and failure behavior

Successful digest, window, and merge model payloads are persisted as request-keyed
judge calls; host-owned identities and resolved citations are reconstructed on
validation. Each identity hashes the requested and provider models, exact
messages, response schema, generation options, and protocol version. On
restart, only a matching reusable success with a phase-valid result is reused;
failed, audit-only, historical, or malformed calls are never reused.

Progress counts unique persisted call identities, so reuse cannot inflate completed
work. Concurrent panel activity and call completion are serialized before
persistence. Partial model outputs never become scores, and cancelled or
unattempted work after a fail-fast stop is not reported as completed. Any
planned rubric failure prevents all judge feedback writes for that run;
successful scores are written only after complete coverage is known. Forced
replacement creates new feedback before deleting prior matches, and write or
cleanup failures remain visible.

## Inference trust boundary

Catalog-backed HTTP evaluation supports direct OpenAI and W&B Inference. Local
evaluation supports Claude, Codex, and Antigravity (`agy`) from the same
provider-qualified catalog. Every local provider receives a minimal child
environment and an isolated working directory. Claude is tool-disabled. Codex
uses isolated state and a deny-by-default filesystem profile; its
repository-check override only removes a CLI precondition.

Antigravity receives each request through a unique, mode-`0600` temporary
prompt file rather than a process argument, so the host argument limit no
longer bounds request size. Because Antigravity's
headless file reader requires accept-edits mode with permission auto-approval,
the process also runs inside a deny-by-default host filesystem sandbox. That
sandbox permits the ephemeral prompt workspace, the minimal runtime and
read-only access to the user's macOS Keychains directory for its system keyring
helper, the helper's `/dev/null` output, and provider network access. Each
attempt receives an isolated temporary home outside the model workspace. The
Antigravity OAuth token is streamed once through a mode-`0600` FIFO and the
path is unlinked as soon as the CLI opens it, leaving no credential file for
model-generated tools; symlinked or non-regular token sources fail closed.
Persistent histories, knowledge, and other `~/.gemini` state are not exposed.
The sandbox denies arbitrary repository file-content reads and writes and other
user-library data. Antigravity startup does require path-metadata access, so
filenames and existence are not hidden. Keychain item access remains subject to
macOS Keychain controls; the filesystem sandbox does not make the keyring a
general evaluation input. Permission auto-approval explicitly trusts
Antigravity with these authentication paths but not repository checkouts.
The temporary workspace contains no repository checkout, is removed after
every attempt, and is not reused across retries or concurrent reviewers. If the
host confinement mechanism is unavailable or the prompt cannot be read, the
request fails closed before inference.

The short bootstrap instruction authorizes only reading the named prompt and
returning its requested response. The prompt file contains the complete system
instruction, response-schema instruction, and user request in that order so
the schema is present in the first bounded file-reader page. Trace text remains
untrusted and cannot expand the host sandbox authority. These controls do not
weaken evidence pinning or permit arbitrary parent secrets and proxy variables
to enter the child environment.

Prompts, raw model output, credentials, and arbitrary parent environment values
are not written to application logs or run errors. Audit records retain safe
structured outcomes and content digests instead.

## Research basis and limits

The design is informed by:

- [PoLL](https://arxiv.org/abs/2404.18796), for heterogeneous judges and score
  pooling;
- [MT-Bench judge analysis](https://arxiv.org/abs/2306.05685), for
  conversation context and boundary sensitivity;
- [self-recognition and preference](https://arxiv.org/abs/2404.13076), for
  hiding generator identity from judge-facing evidence;
- [Agent-as-a-Judge](https://arxiv.org/abs/2410.10934), for inspectable
  trajectory evidence;
- [PReMISE](https://arxiv.org/abs/2605.30803), for rubric-specific abstention;
  and
- [RuVerBench](https://arxiv.org/abs/2606.29920), for the limits and diminishing
  returns of agentic coding judges.

These papers motivate the architecture but do not calibrate this product. There
is no human-labeled validation set, and neither the three-judge cap nor the
context-budget policy is claimed as an empirical optimum. These are product
cost bounds, not claims that larger contexts or another correlated judge could
never add information.

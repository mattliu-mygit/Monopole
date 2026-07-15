# Standalone Signal monitoring design

## Goal

Create a small standalone system that continuously scores agent turns with five
focused W&B Weave monitors and hydrates low-scoring turn feedback into a ranked
list of conversations worth inspecting. This list will later feed Monopole's
Sessions view, but the first version stops at a stable CLI and JSON boundary.

The system favors recall over precision. A low Signal score recommends that a
conversation is interesting; it does not assert that the agent failed. Monopole
remains responsible for complete, evidence-pinned session evaluation.

## Boundary

All Signal-monitoring behavior lives under the root `signal_monitoring/` folder
with its own package metadata, dependencies, `weave-signal-monitor` command,
tests, and documentation. It does not import Monopole modules, reuse Monopole's
internal models, write Monopole state, start Monopole evaluation runs, or modify
the Monopole API or frontend.

The first version has only two responsibilities:

1. install the versioned Signal monitors in Weave; and
2. read their feedback and hydrate low-scoring turns into conversation summaries.

It has no service process, scheduler, local database, synchronization cursor,
webhook, Slack integration, or background worker. Each hydration invocation
queries a bounded time window directly from Weave.

## Signal catalog

The catalog contains five independent turn-level ratings:

- **User frustration** detects annoyance, impatience, confusion, or
  dissatisfaction expressed by the user.
- **User correction or rejection** detects an explicit claim that prior agent
  work was wrong, an explicit rejection, or a redirect intended to correct it.
- **Explicit repeat or rephrase cue** detects language showing that the user is
  repeating or restating an unresolved request. It does not claim to detect
  semantic repetition when the current turn contains no such cue.
- **Stalled or deferred response** detects when the agent unnecessarily hands
  work back to the user, postpones action, or stops despite having enough
  information and authority to continue.
- **Low-quality response** detects a response that is materially irrelevant,
  evasive, repetitive, plainly incomplete, or otherwise fails to address the
  visible request.

Overlap is allowed because each rating preserves a distinct reason for later
inspection. The catalog does not include neutral opportunity markers such as a
completion claim because they should not independently recommend costly review.

Every rating uses the same closed higher-is-better anchors:

- `1.0`: no evidence of the problem;
- `0.75`: weak or ambiguous indication;
- `0.5`: plausible indication;
- `0.25`: strong indication; and
- `0.0`: explicit or severe evidence.

The scorer must return exactly one anchor and a short reason grounded in the
scored turn. The initial recommendation threshold is `<= 0.5`. Signal identity
includes a stable name and catalog version so later prompt changes cannot be
silently mixed with earlier results.

The monitors score every eligible agent turn in the initial personal deployment.
Sampling remains an explicit catalog setting and starts at `1.0`.

## Installation

`weave-signal-monitor install` creates and activates the exact catalog monitors
through a supported Weave monitor API. Installation is idempotent: an existing
monitor with the same versioned identity and definition is reused, while a
conflicting definition fails visibly instead of being overwritten or duplicated.

Installation may safely resume after a partial external failure. It reports the
monitor identities that already exist and those created during the invocation;
it does not attempt a distributed rollback of successfully created Weave
objects.

If the current public Weave API cannot create the newer Agents-view Signal type,
the implementation may use the supported generic Monitor plus LLM-judge scorer
only if its feedback is attached to the agent turn and can be hydrated through
the same documented query boundary. The live compatibility check must establish
that contract before the installer is considered complete.

## Hydration

`weave-signal-monitor hydrate` queries a caller-selected recent time range,
defaulting to the preceding 24 hours. It requests feedback for the exact catalog
identities, keeps ratings at or below the configured threshold, and resolves each
triggering turn to its conversation.

Only conversations with matching low ratings are hydrated. Each conversation is
returned once with:

- conversation identifier;
- recorded display name, or a bounded preview of the first user request;
- start and last-activity times;
- triggering Signal names, versions, ratings, and reasons;
- triggering turn identifiers;
- the lowest observed rating; and
- a W&B link for inspecting the conversation.

Repeated feedback for the same Signal and turn is resolved deterministically in
favor of the newest successfully completed score. Triggering evidence remains
ordered by turn time. Conversations are ranked by lowest rating first and most
recent activity second.

The default output is a compact human-readable table. JSON output follows one
stable versioned schema intended to become the later Monopole integration
boundary. Hydration is read-only and never marks feedback as consumed.

## Failure behavior

Authentication, incomplete pagination, missing required turn-to-conversation
identity, malformed eligible feedback, or inconsistent duplicate identities
fail the command visibly. The command does not emit a partial result that could
be mistaken for the complete recommendation list.

An empty complete query is successful and produces an empty list. Ratings above
the threshold, feedback from unknown Signal versions, and incomplete scorer
attempts are ignored rather than interpreted as healthy or unhealthy evidence.

Errors must identify the failed boundary without logging credentials, scorer
prompts, or unrestricted trace content.

## Verification

Automated checks cover:

- the exact catalog, anchors, threshold, version, and sampling configuration;
- strict rating and reason parsing;
- idempotent installation and conflict rejection;
- bounded time filtering and complete pagination;
- exact Signal-version filtering;
- deterministic duplicate resolution;
- turn-to-conversation grouping, evidence ordering, ranking, and display-name
  fallback;
- stable JSON output and human-readable table output; and
- fail-closed behavior for incomplete or malformed external data.

Tests use mocked Weave responses. A live smoke check in the personal project must
confirm that all five monitors attach readable feedback to agent turns and that
one known low-scoring turn hydrates into the expected conversation and W&B link.

## Deferred work

The following work is deliberately excluded until installation and hydration are
proven against live Weave data:

- Monopole Sessions-view integration;
- automatic Monopole run creation;
- Slack or webhook alerts;
- persistent recommendation or acknowledgement state;
- conversation-level semantic repetition or trajectory analysis; and
- scheduling, polling, or incremental synchronization.

These additions must consume the versioned hydrated-conversation output rather
than reaching into the standalone package's internals.

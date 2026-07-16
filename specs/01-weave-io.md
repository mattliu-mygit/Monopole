# Weave I/O

This boundary turns Weave trace-server records into the exact turn and session
views used by evaluation, then attaches custom score feedback to the refs those
scores describe.

## Identity model

A turn is rooted at an `invoke_agent` span whose parent ID is the empty string.
A session groups turn roots by `conversation_id`; there is no independent
session-root span in this product.

Turn scores attach to the exact trace ref. Session scores attach to the encoded
conversation ref. Evaluation runs additionally pin ordered trace identities and
their conversation and start-time metadata, so later stages cannot absorb new
turns or silently substitute changed evidence.

## Discovery and hydration

Discovery reads lightweight turn roots for listing and selection. Detail
hydration then fetches each requested trace's full root and children. Hydration
is trace-scoped, never conversation-scoped, because several turns in one
conversation must not contaminate one another.

Adapter metadata is requested explicitly and flattened from the trace server's
typed custom-attribute maps. Root spans often omit token totals or effective
model identity, so those values are recovered from child chat spans. User and
assistant messages are similarly backfilled only from detailed evidence for the
same trace.

Pagination preserves chronological order and deduplicates trace IDs. It fails
when an inclusive timestamp cursor cannot advance, rather than returning an
apparently complete partial result. Evaluation-run rehydration preserves the
run's pinned order.

Session discovery also batch-reads feedback for the exact turn refs in the
returned page. Completed W&B Agent Signal feedback whose typed rating matches
its output is recognized by its monitor scorer identity. Low evidence is
attached to the owning session in turn-time order; newer feedback supersedes an
older row for the same signal version and turn. Feedback for omitted sessions
is not read. Legacy or incomplete monitor rows without a typed rating are not
Signal evidence and are ignored.

Hydration is fail closed:

- every requested trace must contain its detailed `invoke_agent` root;
- a response at a server result ceiling is treated as possibly truncated;
- session feedback reads use exact refs in bounded batches and reject a batch at
  its response ceiling; and
- no part of a batch is evaluated until the complete batch passes validation.

These rules keep missing detail from becoming neutral or fabricated evidence.
Malformed Agent Signal feedback that contains a typed rating fails session
discovery rather than silently hiding a completed review recommendation.

## Feedback

Custom feedback is named `weave_agent_signals.<scorer_name>`. Its payload
contains the rating, granularity, scorer version, score time, reason, tags, and
scorer-specific details. Deterministic scores may include confidence heuristics;
model judgments do not claim calibrated confidence.

Execution time, stored with the score details, determines trend chronology.
Feedback-write time does not.

Writes check existing feedback by scorer and ref before mutation. A matching
row observed by a non-forced write is skipped. Replacement queries the exact
matches, creates the new feedback first, then purges every prior match by
feedback ID. A failed create leaves every prior row intact. Failed cleanup is
reported after the new feedback is safely durable and leaves any unpurged prior
rows available for audit and later cleanup.
Reflection reads feedback only from its pinned turn and session refs and records
the identity and digest of every consumed record.

## External correctness constraints

The following details reflect Weave and adapter behavior and must remain
explicit:

1. Root turns use `parent_span_id == ""`, not null or a negative predicate.
2. Adapter metadata requires requested custom-attribute columns and flattening
   of all typed maps.
3. Trace-server `DateTime64(6)` bounds use UTC ISO strings without `Z` or a
   numeric offset.
4. Detail columns require detail mode and cannot be combined with grouped
   queries.
5. Children are hydrated by exact `trace_id`; every detail batch must include
   the requested roots.
6. Bash results may be JSON envelopes whose stdout and stderr need unwrapping.
   Status is often unset, and an explicit null exit code means unknown.
7. Token totals and effective model identity may exist only on child chat spans.
8. Feedback query records are returned under `result`.
9. Forced feedback replacement creates the new row before purging any prior
   match. Cleanup purges one feedback ID at a time only after creation succeeds.
10. Custom scores use a flat payload. Monitor-specific scorer columns require
    runnable, call, and trigger refs this product does not create.
11. W&B Inference uses bearer W&B authentication plus the entity/project header;
    HTTP 402 means inference credits are unavailable.
12. Conversation IDs are URL-encoded when constructing session refs.

Credentials come from `WANDB_API_KEY` or the stored W&B netrc entry. Trace
reads and score writes use the user's configured entity and project.

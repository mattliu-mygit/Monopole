# Monopole Signal Callouts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Annotate existing Monopole session summaries with low standalone Agent Signal feedback and show one compact review callout in both session-selection surfaces.

**Architecture:** The existing Sessions route keeps owning trace discovery and conversation grouping. It batch-reads feedback for only the returned turn refs, recognizes versioned standalone Signal scorer refs, and adds ordered `signal_evidence`; one small React component derives the visible aggregate and is reused by Sessions and Run Selection.

**Tech Stack:** Python 3.11+, FastAPI, existing Weave HTTP client, React, TypeScript, TanStack Query, Vitest, pytest.

## Global Constraints

- Add no endpoint, subprocess, standalone-package import, scheduler, cache, database, or persistence.
- Preserve existing `conversation_id`, date filtering, recency ordering, truncation, navigation, and selection behavior.
- Read feedback only for sessions returned under the existing limit.
- Recognize only `wandb.agent_monitor` rows whose scorer object name matches `agent-signal-<slug>-<version>-scorer`.
- Accept only numeric, non-boolean anchors `0.0`, `0.25`, `0.5`, `0.75`, and `1.0`; display only ratings `<= 0.5`.
- Resolve duplicate Signal feedback per turn to the newest `(created_at, id)` row.
- Fail the Sessions request for malformed eligible feedback or incomplete exact-ref feedback hydration.
- Return `signal_evidence: []` when the complete read contains no low Signal.
- Use one amber callout; never present Signal evidence as a confirmed failure.
- Keep sessions manually selectable; do not re-rank or auto-select them.
- Execute inline without subagents, as requested.

---

### Task 1: Hydrate low Signal feedback into session summaries

**Files:**
- Modify: `src/weave_agent_signals/routes/inspection.py`
- Test: `tests/interfaces/routes/test_inspection.py`

**Interfaces:**
- Consumes: `TurnSpan.ref_for(entity, project)` and `WeaveClient.query_all_feedback_batch(refs)`.
- Produces: every `/api/sessions` item contains `signal_evidence: list[dict]` with `signal`, `version`, `rating`, `reason`, `turn_id`, and `turn_started_at`.

- [ ] **Step 1: Add failing route tests for low evidence and exact-ref batching**

Extend `FakeClient` so listing feedback can be configured by exact ref. Add a test with two visible turns in one conversation and feedback shaped as:

```python
{
    "id": "feedback-low",
    "weave_ref": "weave:///weave-team/agent-sessions/agent_turn/trace-a",
    "feedback_type": "wandb.agent_monitor",
    "runnable_ref": (
        "weave:///weave-team/agent-sessions/object/"
        "agent-signal-user-frustration-v1-scorer:digest"
    ),
    "created_at": "2026-07-14T12:02:00Z",
    "payload": {
        "output": {
            "value": 0.25,
            "reason": "The user explicitly says they are frustrated.",
        }
    },
}
```

Assert the session contains exactly:

```python
[
    {
        "signal": "user-frustration",
        "version": "v1",
        "rating": 0.25,
        "reason": "The user explicitly says they are frustrated.",
        "turn_id": "trace-a",
        "turn_started_at": "2026-07-14T12:00:00+00:00",
    }
]
```

Also assert listing performs one feedback batch containing only refs for returned, non-synthetic sessions.

- [ ] **Step 2: Run the focused backend test and confirm failure**

Run:

```bash
.venv/bin/python -m pytest tests/interfaces/routes/test_inspection.py -q
```

Expected: FAIL because session summaries do not contain `signal_evidence` and listing does not query feedback.

- [ ] **Step 3: Implement strict Signal parsing in the inspection boundary**

Add private constants and helpers in `inspection.py`:

```python
_SIGNAL_SCORER_RE = re.compile(
    r"^agent-signal-(?P<signal>.+)-(?P<version>v[1-9][0-9]*)-scorer(?:[:].+)?$"
)
_SIGNAL_ANCHORS = {0.0, 0.25, 0.5, 0.75, 1.0}
_SIGNAL_THRESHOLD = 0.5
```

The helpers must:

1. URL-decode the scorer ref's final path component and return `(signal, version)` only for the regex above.
2. Ignore non-`wandb.agent_monitor` rows and unrelated scorer refs.
3. Require an object at `payload.output`, accept `rating` or `value`, reject conflicting aliases, booleans, non-anchors, and blank reasons.
4. Parse timezone-aware `created_at` strings or datetimes for duplicate ordering.
5. Select the newest row per `(signal, version, turn_id)`.
6. Return evidence ordered by `(turn_started_at, signal, version, turn_id)`.

Raise `ValueError` with a boundary-specific message for malformed eligible rows; do not coerce malformed data.

- [ ] **Step 4: Join feedback after limiting the visible sessions**

Refactor only the final portion of `get_sessions`:

1. Keep grouped sessions and summaries associated until after recency sorting.
2. Slice to `limit`.
3. Build exact turn refs for those visible sessions.
4. Call `query_all_feedback_batch` once with those refs.
5. Add each conversation's ordered evidence to its existing summary.
6. Return the original `total`, `truncated`, and `limit` values.

Do not hydrate feedback for omitted or synthetic sessions.

- [ ] **Step 5: Add duplicate, healthy, unrelated, and malformed cases**

Add focused tests proving:

- a newer `rating` alias replaces an older `value` alias for one Signal and turn;
- ratings `0.75` and `1.0` produce an empty list;
- unrelated feedback types and unrelated Agent Monitor scorer refs are ignored; and
- an eligible row with a boolean rating, invalid anchor, conflicting aliases, blank reason, or naive timestamp fails the request.

- [ ] **Step 6: Run the focused backend tests**

Run:

```bash
.venv/bin/python -m pytest tests/interfaces/routes/test_inspection.py tests/io/test_client.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit the backend boundary**

```bash
git add src/weave_agent_signals/routes/inspection.py tests/interfaces/routes/test_inspection.py
git commit -m "feat: hydrate session signal evidence"
```

### Task 2: Add one shared Signal callout component

**Files:**
- Create: `frontend/src/components/SignalCallout.tsx`
- Create: `frontend/tests/components/SignalCallout.test.tsx`
- Modify: `frontend/src/types.ts`

**Interfaces:**
- Consumes: `SignalEvidence[]` from `SessionSummary.signal_evidence`.
- Produces: `<SignalCallout evidence={...} />`, rendering nothing for an empty list and one accessible amber callout otherwise.

- [ ] **Step 1: Add the frontend contract and failing component tests**

Add:

```typescript
export interface SignalEvidence {
  signal: string
  version: string
  rating: number
  reason: string
  turn_id: string
  turn_started_at: string
}
```

Add `signal_evidence: SignalEvidence[]` to `SessionSummary`.

Write component tests with duplicate `user-frustration` evidence and one
`low-quality-response` item. Assert:

- the callout says `Needs review`;
- the lowest value appears as `0.25`;
- labels are deduplicated and rendered as `user frustration` and `low quality response`;
- its accessible label describes a review recommendation; and
- empty evidence renders no element.

- [ ] **Step 2: Run the component test and confirm failure**

Run:

```bash
cd frontend
npm test -- SignalCallout.test.tsx
```

Expected: FAIL because `SignalCallout` does not exist.

- [ ] **Step 3: Implement the minimal presentational component**

Implement a pure component that:

- computes `Math.min(...evidence.map(item => item.rating))`;
- preserves first-seen distinct Signal order;
- replaces hyphens with spaces for labels; and
- renders one amber rounded badge with text:

```text
Needs review · 0.25 · user frustration, low quality response
```

Use an accessible label beginning `Signal review recommendation` and do not add state, hooks, icons, tooltips, or click behavior.

- [ ] **Step 4: Run the component tests**

Run:

```bash
cd frontend
npm test -- SignalCallout.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit the shared frontend contract**

```bash
git add frontend/src/types.ts frontend/src/components/SignalCallout.tsx frontend/tests/components/SignalCallout.test.tsx
git commit -m "feat: add signal review callout"
```

### Task 3: Show callouts without changing session behavior

**Files:**
- Modify: `frontend/src/pages/Sessions.tsx`
- Modify: `frontend/src/features/runs/RunSelection.tsx`
- Modify: `frontend/tests/pages/Sessions.test.tsx`
- Modify: `frontend/tests/runs/RunSelection.test.tsx`
- Modify: `frontend/tests/pages/RunDetail.test.tsx`

**Interfaces:**
- Consumes: the component and `SessionSummary.signal_evidence` from Task 2.
- Produces: identical Signal callouts in both existing session surfaces.

- [ ] **Step 1: Update fixtures and add failing integration assertions**

Add `signal_evidence: []` to every `SessionSummary` fixture. Give one session low evidence and assert both Sessions and Run Selection render `Needs review`, `0.25`, and the human-readable Signal label.

Keep existing assertions for encoded links, truncation, checked state, and selection callbacks unchanged.

- [ ] **Step 2: Run the focused frontend tests and confirm failure**

Run:

```bash
cd frontend
npm test -- Sessions.test.tsx RunSelection.test.tsx RunDetail.test.tsx
```

Expected: FAIL because neither session surface renders `SignalCallout`.

- [ ] **Step 3: Render the shared component in both surfaces**

Import `SignalCallout` in `Sessions.tsx` and `RunSelection.tsx`. Render it once per session, directly below the request preview in Sessions and directly below the preview/metadata block in Run Selection:

```tsx
<SignalCallout evidence={session.signal_evidence} />
```

Do not modify links, labels, checkbox handlers, sorting, query keys, selection defaults, or API calls.

- [ ] **Step 4: Run the focused frontend tests**

Run:

```bash
cd frontend
npm test -- SignalCallout.test.tsx Sessions.test.tsx RunSelection.test.tsx RunDetail.test.tsx
```

Expected: PASS.

- [ ] **Step 5: Commit the two integrations**

```bash
git add frontend/src/pages/Sessions.tsx frontend/src/features/runs/RunSelection.tsx frontend/tests/pages/Sessions.test.tsx frontend/tests/runs/RunSelection.test.tsx frontend/tests/pages/RunDetail.test.tsx
git commit -m "feat: call out low-signal sessions"
```

### Task 4: Reconcile canonical specs and verify the complete branch

**Files:**
- Modify: `specs/01-weave-io.md`
- Modify: `specs/03-analysis-monitoring.md`
- Delete: `docs/superpowers/specs/2026-07-15-monopole-signal-callouts-design.md`
- Delete: `docs/superpowers/plans/2026-07-15-monopole-signal-callouts.md`

**Interfaces:**
- Consumes: the verified implementation from Tasks 1-3.
- Produces: canonical documentation describing the current architecture without temporary planning artifacts.

- [ ] **Step 1: Update canonical product boundaries**

In `specs/01-weave-io.md`, add that visible session discovery batch-reads exact
turn feedback and returns ordered low standalone Signal evidence, while malformed
eligible feedback fails closed.

In `specs/03-analysis-monitoring.md`, add that low Agent Signals are high-recall
review recommendations shown on session-selection surfaces; they do not re-rank,
auto-select, or assert failure.

- [ ] **Step 2: Remove superseded temporary artifacts**

Delete the design and this implementation plan after their lasting constraints
are represented in the canonical specs and tests.

- [ ] **Step 3: Run every repository gate**

From the repository root:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src/ tests/
.venv/bin/ruff format --check src/ tests/
```

From `frontend/`:

```bash
npm test
npm run lint
npm run build
```

Expected: all commands exit `0`.

- [ ] **Step 4: Review the final diff for scope and cleanup**

Run:

```bash
git status --short
git diff --check
git diff --stat main...HEAD
```

Expected: only the Sessions feedback join, shared callout, two UI integrations,
focused tests, and canonical spec updates remain. No generated files,
dependencies, temporary plans, or unrelated changes remain.

- [ ] **Step 5: Commit canonical documentation and cleanup**

```bash
git add specs/01-weave-io.md specs/03-analysis-monitoring.md docs/superpowers/specs/2026-07-15-monopole-signal-callouts-design.md docs/superpowers/plans/2026-07-15-monopole-signal-callouts.md
git commit -m "docs: specify session signal callouts"
```

Expected: the worktree is clean.

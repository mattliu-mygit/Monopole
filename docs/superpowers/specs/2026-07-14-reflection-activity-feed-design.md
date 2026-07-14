# Reflection Activity Feed

## Goal

Replace the nearly static Reflecting state on evaluation-run detail pages with a compact activity feed driven by real reflection milestones. A person watching a long reflection should be able to tell what the system is doing, whether it is moving, which model is active, and what candidates have been evaluated without exposing prompts or model output.

## User experience

While reflection is running, the Reflecting card shows:

- the current phase with an in-progress indicator;
- elapsed time;
- determinate candidate progress when a total is known;
- the five newest milestones; and
- a **Show all activity** control that expands the complete retained history.

Example milestones include:

- `Loaded 232 feedback signals from 4 sessions`
- `Prepared coaching digest`
- `Found 3 configurable artifacts`
- `Generating candidate 2 of 6 — claude-sonnet-5`
- `Scoring candidate 2 of 6`
- `Candidate 2 scored 0.81`
- `Selected candidate 2 of 6`

The existing configuration is evaluated separately as the baseline and is not
counted as a generated candidate. If the baseline remains best, the final
milestone says `Kept current configuration — no candidate scored higher`.

The existing two-second run polling interval delivers updates. No WebSocket, streaming endpoint, or simulated message rotation is introduced.

Completed runs retain their activity history. Failed and cancelled runs show the history alongside the existing terminal state so the last successful milestone remains available for diagnosis. Older run records containing only `status_message` continue to render the current fallback presentation.

## Progress contract

`Run.reflecting_progress` remains a JSON object and gains a structured contract:

```json
{
  "phase": "scoring_candidates",
  "status_message": "Scoring candidate 2 of 6",
  "started_at": "2026-07-14T19:20:00+00:00",
  "generated": 2,
  "scored": 1,
  "total_candidates": 6,
  "events": [
    {
      "id": 5,
      "at": "2026-07-14T19:22:14+00:00",
      "phase": "scoring_candidates",
      "message": "Scoring candidate 2 of 6",
      "model": "claude-sonnet-5"
    }
  ]
}
```

Only fields relevant to an event are present. Candidate-score events may include `candidate`, `score`, and `model`. The current top-level counters are copied forward on every update so each poll returns a complete snapshot rather than a patch.

The event history is capped at 100 entries. IDs increase within a run and remain stable across polls. Messages are concise, user-facing summaries; the contract excludes prompts, generated artifact contents, credentials, reasoning, and raw subprocess output.

## Backend design

### Reflector boundary

`run_reflection()` accepts an optional progress callback. The callback receives semantic progress data but has no dependency on the API, `RunStore`, or SQLite.

Instrumentation wraps the two operations the reflector already controls:

1. The reflection language-model callable reports candidate generation immediately before and after each call. For round-robin reflection it reports the actual model selected for that call.
2. The artifact evaluator reports candidate evaluation immediately before the judge call and reports the resulting score afterward.

The reflector emits a final selection event after GEPA returns. Call counters are local to a reflection run. `total_candidates` is the configured proposal budget and excludes GEPA's seed/baseline candidate. The baseline evaluation gets its own `Evaluating current configuration` milestone and does not increment `scored`. The UI does not invent percentage progress when a meaningful total is unavailable.

### API boundary

`_run_reflecting_step()` owns persistence. It creates a small progress recorder that:

- timestamps the start and each event in UTC;
- maintains current phase, status message, counters, and ordered history;
- caps history at 100 events; and
- writes a complete `reflecting_progress` snapshot through `RunStore.update()`.

The API also records preprocessing milestones around trace selection, feedback filtering, coaching-digest construction, and artifact discovery. It passes the recorder callback into `run_reflection()` for generation and scoring milestones.

No new endpoint or database column is required because `reflecting_progress` is already persisted and returned by the run endpoint.

## Frontend design

`RunDetail.tsx` defines the typed reflection-progress/event shapes and renders them through a focused activity-feed component.

The component:

- derives elapsed time from `started_at`, refreshing naturally with existing run polling;
- shows generated/scored counts and a progress bar only when `total_candidates` is useful;
- renders the latest five events initially;
- toggles to the full retained history without another request;
- differentiates the current event from completed milestones using both text/icon treatment and color; and
- preserves keyboard access and visible button labels.

The activity feed remains visible above proposals after completion when events exist. The existing proposal tabs, diff viewer, selection, and apply workflow do not change.

## Failure handling

Progress reporting is observational: the callback does not change candidate content or selection. A reflection failure continues through the existing run failure path, while already-persisted milestones remain intact. Missing or partially populated progress fields are tolerated by the frontend.

The feed never renders untrusted model text as HTML. All event messages are generated by application code and rendered as React text.

## Testing

Backend tests will be written first and will verify:

- preparation milestones are persisted in order;
- candidate generation reports the correct round-robin model and counters;
- evaluation reports candidate scores;
- final selection is reported;
- event history is capped at 100 entries; and
- reflection still works when no callback is supplied.

API tests will exercise the progress recorder through `_run_reflecting_step()` with mocked clients and reflection work. Existing tests will be adjusted only where the new optional callback changes call assertions.

The frontend will be validated with its TypeScript/build checks and the repository's available lint checks. The run-detail UI will also be visually inspected at normal and narrow widths, with both compact and expanded activity histories.

## Out of scope

- Raw or streamed model/subprocess logs
- Prompt, reasoning, or generated-artifact previews during execution
- WebSockets or server-sent events
- Changes to scoring or judging progress UX
- Changes to reflection ranking or candidate-selection behavior
- Reworking reflection cancellation semantics

import { For, Show } from "solid-js";
import type {
  InferenceStepAudit,
  JudgingAttemptSummary,
  JudgingFailureDetail,
  JudgingPlan,
  JudgingProgress,
  ReviewAttempt,
} from "../../../frontend/src/types";
import { Activity } from "./Activity";
import { DEBUG_EVENT_PHASES } from "../../../frontend/src/features/runs/runFormat";
import { fixed, plural, turnRanges, usageText, words } from "./words";

type Event = JudgingProgress["events"][number];

function eventFacts(event: Event): (string | null)[] {
  const count = (value: number | null | undefined, unit: string) => (value == null ? null : `${value.toLocaleString()} ${unit}`);
  return [
    event.model ?? null,
    event.rubric ? words(event.rubric) : null,
    event.conversation_id ? `session ${event.conversation_id}` : null,
    event.request_attempt != null ? `attempt ${event.request_attempt} of ${event.max_attempts ?? "?"}` : null,
    event.item_index != null ? `${event.phase.startsWith("digest") ? "chunk" : "window"} ${event.item_index} of ${event.item_total ?? "?"}` : null,
    event.elapsed_seconds != null ? `${event.elapsed_seconds.toFixed(1).replace(/\.0$/, "")}s` : null,
    event.error_category ? words(event.error_category) : null,
    event.retry_reason ? `reason ${words(event.retry_reason)}` : null,
    event.provider_status != null ? `provider ${event.provider_status}` : null,
    event.provider_error_code ? words(event.provider_error_code) : null,
    event.finish_reason ? `finish ${words(event.finish_reason)}` : null,
    count(event.completion_tokens, "completion tokens"),
    count(event.reasoning_tokens, "reasoning tokens"),
    count(event.estimated_input_tokens, "estimated input tokens"),
    count(event.model_context_tokens, "token context"),
    event.status ? words(event.status) : null,
  ];
}

function EventDetail(props: { event: Event }) {
  const e = () => props.event;
  return (
    <>
      <Show when={e().provider_error_message}>{(text) => <pre class="m-output m-failed">{text()}</pre>}</Show>
      <Show when={e().output_sha256}>
        <details class="m-fold">
          <summary>Output identity</summary>
          <dl class="m-kv">
            <dt>Output sha256</dt>
            <dd class="m-mono m-break">{e().output_sha256}</dd>
            <Show when={e().exit_code != null}>
              <dt>Exit code</dt>
              <dd>{e().exit_code}</dd>
            </Show>
            <Show when={e().prompt_characters != null}>
              <dt>Prompt</dt>
              <dd>{e().prompt_characters!.toLocaleString()} characters</dd>
            </Show>
            <Show when={e().stdout_chars != null || e().stderr_chars != null}>
              <dt>Output</dt>
              <dd>
                {e().stdout_chars ?? 0} stdout · {e().stderr_chars ?? 0} stderr characters
              </dd>
            </Show>
            <Show when={e().output_mode}>
              <dt>Output mode</dt>
              <dd>{words(e().output_mode!)}</dd>
            </Show>
          </dl>
        </details>
      </Show>
    </>
  );
}

function Step(props: { step: InferenceStepAudit }) {
  const s = () => props.step;
  return (
    <li class="m-step">
      <div>
        <strong>{s().phase}</strong> · {s().reused ? "reused" : "this run"} · <span class="m-mono">{s().schema_name}</span>
      </div>
      <dl class="m-kv">
        <dt>Model</dt>
        <dd>{s().resolved_model ?? s().requested_model}</dd>
        <dt>Requests</dt>
        <dd>{s().transport_request_count}</dd>
        <dt>Usage</dt>
        <dd>{usageText(s().usage)}</dd>
        <Show when={s().schema_fallback_reason}>
          <dt>Schema fallback</dt>
          <dd>{s().schema_fallback_reason}</dd>
        </Show>
        <For each={s().response_diagnostics ?? []}>
          {(response, index) => (
            <>
              <dt>Response {index() + 1}</dt>
              <dd>
                {response.finish_reason ? `finish ${words(response.finish_reason)} · ` : ""}
                {response.content_characters.toLocaleString()} characters · {usageText(response.usage ?? {})}
              </dd>
            </>
          )}
        </For>
      </dl>
    </li>
  );
}

/** One reviewer's attempt at one rubric: its verdict and reasons, and folded, how it reached the
 * model and each inference step. */
function Attempt(props: { attempt: ReviewAttempt }) {
  const a = () => props.attempt;
  return (
    <li class="m-attempt">
      <div class="m-attempt-head">
        <strong>
          Judge {a().position} · {a().requested_model}
        </strong>
        <span class={{ "m-failed": a().status === "failed" }}>
          {words(a().trigger)} · {words(a().status)}
          {a().score !== null ? ` · ${fixed(a().score)}` : ""}
        </span>
      </div>
      <Show when={a().resolved_model && a().resolved_model !== a().requested_model}>
        <div class="m-quiet">Answered by {a().resolved_model}</div>
      </Show>
      <Show when={a().rationale}>{(text) => <p>{text()}</p>}</Show>
      <Show when={a().behavioral_feedback}>
        {(feedback) => (
          <dl class="m-kv">
            <Show when={feedback().success}>
              <dt>Went well</dt>
              <dd>{feedback().success}</dd>
            </Show>
            <Show when={feedback().problem}>
              <dt>Problem</dt>
              <dd>{feedback().problem}</dd>
            </Show>
            <Show when={feedback().desired_behavior}>
              <dt>Wanted instead</dt>
              <dd>{feedback().desired_behavior}</dd>
            </Show>
          </dl>
        )}
      </Show>
      <Show when={a().status === "failed"}>
        <p class="m-failed">
          {a().error_type ?? "Error"}: {a().message ?? "No error detail returned"}
        </p>
      </Show>
      <Show when={a().status === "skipped"}>
        <p>Skipped: {words(a().skip_reason ?? "insufficient_context_capacity")}</p>
      </Show>
      <details class="m-fold">
        <summary>
          How it reached the model · {plural(a().transport_request_count, "request")} · {plural(a().steps.length, "step")}
        </summary>
        <dl class="m-kv">
          <dt>Asked for</dt>
          <dd>
            {a().requested_family} · {a().requested_backend}
          </dd>
          <dt>Answered as</dt>
          <dd>{a().resolved_family ?? "—"}</dd>
          <dt>Output mode</dt>
          <dd>{a().output_mode ? words(a().output_mode!) : "—"}</dd>
          <dt>Schema</dt>
          <dd>
            {a().schema_name ?? "—"}
            {a().verdict_schema_version ? ` v${a().verdict_schema_version}` : ""}
          </dd>
          <dt>Usage</dt>
          <dd>{usageText(a().usage)}</dd>
          <Show when={a().evidence_ids.length > 0}>
            <dt>Evidence IDs</dt>
            <dd class="m-mono">{a().evidence_ids.join(", ")}</dd>
          </Show>
          <Show when={a().schema_fallback_reason}>
            <dt>Schema fallback</dt>
            <dd>{a().schema_fallback_reason}</dd>
          </Show>
          <Show when={a().raw_output_digest}>
            <dt>Output digest</dt>
            <dd class="m-mono m-break">{a().raw_output_digest}</dd>
          </Show>
        </dl>
        <Show when={a().steps.length > 0}>
          <ol class="m-steps" aria-label="Inference steps">
            <For each={a().steps}>{(step) => <Step step={step} />}</For>
          </ol>
        </Show>
      </details>
    </li>
  );
}

function Failure(props: { failure: JudgingFailureDetail; label: (rubric: string) => string }) {
  const f = () => props.failure;
  const root = () => f().attempts.find((attempt) => attempt.status === "failed");
  return (
    <li class="m-failure">
      <strong class="m-failed">{props.label(f().rubric)}</strong>
      <Show when={root()}>
        {(attempt) => (
          <p>
            Cause: {attempt().requested_model} · {attempt().error_type ?? "Error"}: {attempt().message ?? "No error detail returned"}
          </p>
        )}
      </Show>
      <p class="m-quiet">
        {f().error_type}: {f().message ?? "No error detail returned"}
      </p>
      <Show when={f().attempts.length > 0}>
        <details class="m-fold">
          <summary>Every judge's attempt · {f().attempts.length}</summary>
          <ol class="m-attempts">
            <For each={f().attempts}>{(attempt) => <Attempt attempt={attempt} />}</For>
          </ol>
        </details>
      </Show>
    </li>
  );
}

function Review(props: { record: JudgingAttemptSummary; label: (rubric: string) => string }) {
  const r = () => props.record;
  return (
    <details class="m-fold m-review">
      <summary>
        <strong>{props.label(r().rubric)}</strong> ·{" "}
        <span class={{ "m-failed": r().review_status === "failed" }}>{words(r().review_status)}</span>
        {r().rating === null ? "" : ` · ${fixed(r().rating)}`} · {plural(r().attempt_count, "attempt")}
      </summary>
      <p class="m-quiet">
        {words(r().scope)} · session <span class="m-mono">{r().conversation_id}</span>
      </p>
      <ol class="m-attempts">
        <For each={r().attempts}>{(attempt) => <Attempt attempt={attempt} />}</For>
      </ol>
    </details>
  );
}

type WindowPlan = NonNullable<JudgingPlan["sessions"][number]["reviewers"][number]["window_plan"]>;

/** One pinned window, named by the turns it judges, the turns it adds for context, and its size. */
function WindowRow(props: { plan: WindowPlan; window: WindowPlan["windows"][number] }) {
  const turnsOf = (ids: readonly string[]) => {
    const positions = new Map(props.plan.raw_turns.map((turn) => [turn.trace_id, turn.position]));
    return turnRanges(ids.flatMap((id) => positions.get(id) ?? []));
  };
  const context = () => {
    const core = new Set(props.window.core_trace_ids);
    return props.window.raw_trace_ids.filter((id) => !core.has(id));
  };
  return (
    <li>
      <strong>Window {props.window.index}</strong> · {turnsOf(props.window.core_trace_ids)}
      {context().length > 0 ? `, plus ${turnsOf(context())} for context` : ""} · {props.window.raw_tokens.toLocaleString()} estimated tokens
    </li>
  );
}

/** One reviewer's pinned sliding windows for one session, each named by its turns and estimated tokens. */
function Windows(props: { plan: JudgingPlan }) {
  return (
    <For each={props.plan.sessions}>
      {(session) => (
        <details class="m-fold">
          <summary>
            Session <span class="m-mono">{session.conversation_id.slice(0, 13)}</span> · {plural(session.turn_count, "turn")} ·{" "}
            {plural(session.reviewers.length, "judge")}
          </summary>
          <For each={session.reviewers}>
            {(reviewer) => (
              <div class="m-reviewer">
                <strong>
                  Judge {reviewer.ordinal} · {reviewer.judge.label}
                </strong>
                <Show
                  when={reviewer.status !== "skipped"}
                  fallback={<p>Skipped · {words(reviewer.skip_reason ?? "insufficient_context_capacity")}</p>}
                >
                  <p class="m-quiet">
                    at most {reviewer.work_bounds.digest_calls} digests · {reviewer.work_bounds.window_calls_per_rubric} windows per rubric ·{" "}
                    {reviewer.work_bounds.merge_calls_per_rubric} merge per rubric
                  </p>
                  <Show when={reviewer.window_plan}>
                    {(windowPlan) => (
                      <ol class="m-windows">
                        <For each={windowPlan().windows}>{(window) => <WindowRow plan={windowPlan()} window={window} />}</For>
                      </ol>
                    )}
                  </Show>
                </Show>
              </div>
            )}
          </For>
        </details>
      )}
    </For>
  );
}

/**
 * Judging, rewritten from the web UI's JudgingProgress: the activity and its events, how far the
 * reviews got, the reviews that failed with their cause first, each session's pinned sliding
 * windows, and every rubric review with each judge's attempt.
 */
export function Judging(props: {
  plan: JudgingPlan | null;
  progress: JudgingProgress | null;
  result: JudgingProgress | null;
  active: boolean;
  label: (rubric: string) => string;
}) {
  const state = () => props.result ?? props.progress;
  const planned = () => state()?.planned_rubrics ?? props.plan?.totals.planned_rubrics ?? 0;
  const max = (key: "maximum_digest_calls" | "maximum_window_calls" | "maximum_merge_calls") => props.plan?.totals[key] ?? 0;
  return (
    <Show when={props.plan || state()} fallback={<p class="m-quiet">Judging hasn't started.</p>}>
      <Activity
        label="Judging activity"
        status={state()?.status_message ?? `Planned ${planned()} rubric reviews`}
        startedAt={state()?.started_at ?? null}
        active={props.active}
        events={state()?.events ?? []}
        facts={eventFacts}
        detail={(event) => <EventDetail event={event} />}
        debug={DEBUG_EVENT_PHASES}
      />
      <div class="m-progress" role="progressbar" aria-label="Rubrics reviewed" aria-valuemin={0} aria-valuemax={planned()} aria-valuenow={state()?.rubrics_completed ?? 0}>
        <i style={{ width: `${planned() ? Math.round(Math.min(1, (state()?.rubrics_completed ?? 0) / planned()) * 100) : 0}%` }} />
      </div>
      <dl class="m-stats">
        <div>
          <dt>Rubrics reviewed</dt>
          <dd>
            {state()?.rubrics_completed ?? 0} of {planned()}
          </dd>
          <dd class="m-quiet">
            {state()?.rated_rubrics ?? 0} rated · {state()?.not_evaluable_rubrics ?? 0} not evaluable
          </dd>
        </div>
        <div>
          <dt>Judge attempts</dt>
          <dd>{state()?.reviewer_attempts_completed ?? 0}</dd>
          <dd class="m-quiet">
            {state()?.minimum_reviewer_attempts ?? props.plan?.totals.minimum_reviewer_attempts ?? 0} to{" "}
            {state()?.maximum_reviewer_attempts ?? props.plan?.totals.maximum_reviewer_attempts ?? 0} planned
          </dd>
        </div>
        <div>
          <dt>Model calls</dt>
          <dd>
            {state()?.digest_steps_completed ?? 0} of {state()?.maximum_digest_steps ?? max("maximum_digest_calls")} digests
          </dd>
          <dd>
            {state()?.window_steps_completed ?? 0} of {state()?.maximum_window_steps ?? max("maximum_window_calls")} windows ·{" "}
            {state()?.merge_steps_completed ?? 0} of {state()?.maximum_merge_steps ?? max("maximum_merge_calls")} merges
          </dd>
        </div>
        <div>
          <dt>Scores written</dt>
          <dd>{state()?.scores_written ?? 0}</dd>
          <dd class={{ "m-failed": !!(state()?.failure_count || state()?.write_failure_count) }}>
            {plural(state()?.failure_count ?? 0, "review failure")} · {plural(state()?.write_failure_count ?? 0, "write failure")}
          </dd>
        </div>
      </dl>
      <Show when={props.plan}>
        {(plan) => (
          <p class="m-quiet">
            {plural(plan().totals.sessions_planned, "session")} · {plural(plan().totals.turns_considered, "turn")} ·{" "}
            {plural(plan().totals.windows_planned, "pinned window")}
          </p>
        )}
      </Show>
      <Show when={(state()?.failure_details.length ?? 0) > 0}>
        <div class="m-sub">
          <h4 class="m-failed">Failed reviews · {state()!.failure_detail_count}</h4>
          <ul class="m-failures">
            <For each={state()!.failure_details}>{(failure) => <Failure failure={failure} label={props.label} />}</For>
          </ul>
          <Show when={state()!.failure_details_truncated}>
            <p class="m-quiet">
              Showing {state()!.failure_details.length} of {state()!.failure_detail_count} failures.
            </p>
          </Show>
        </div>
      </Show>
      <Show when={props.plan && props.plan.sessions.length > 0}>
        <div class="m-sub">
          <h4>Pinned sliding windows · {props.plan!.totals.windows_planned}</h4>
          <Windows plan={props.plan!} />
        </div>
      </Show>
      <Show when={(state()?.attempt_summaries.length ?? 0) > 0}>
        <div class="m-sub">
          <h4>Rubric reviews · {state()!.attempt_summary_count}</h4>
          <For each={state()!.attempt_summaries}>{(record) => <Review record={record} label={props.label} />}</For>
          <Show when={state()!.attempt_summaries_truncated}>
            <p class="m-quiet">
              Showing {state()!.attempt_summaries.length} of {state()!.attempt_summary_count} reviews.
            </p>
          </Show>
        </div>
      </Show>
    </Show>
  );
}

import { For, Show } from "solid-js";
import type {
  EvaluatorRecord,
  GenerationAttempt,
  ReflectingProgress,
  ReflectionCandidate,
  ReflectionResult,
  SuccessfulReflectionResult,
} from "../../../frontend/src/types";
import { signed } from "../format";
import { Activity } from "./Activity";
import { fixed, plural, usageText, words } from "./words";

type ReflectionEvent = ReflectingProgress["events"][number];
type Challenge = NonNullable<SuccessfulReflectionResult["challenge"]>;
type Arm = NonNullable<Challenge["baseline"]>;

const EXCERPT_LIMIT = 2_000;
const excerpt = (text: string) => (text.length <= EXCERPT_LIMIT ? text : `${text.slice(0, EXCERPT_LIMIT)}…`);

function eventFacts(event: ReflectionEvent): (string | null)[] {
  const rejected = event.phase === "candidate_rejected";
  return [
    event.candidate != null ? `${rejected ? "Proposal attempt" : "Proposal"} ${event.candidate}` : null,
    event.model ?? null,
    event.acting_role ? words(event.acting_role) : null,
    event.status ?? null,
    event.error_type ?? null,
    event.score != null ? `predicted score ${fixed(event.score, 3)}` : null,
  ];
}

function EventDetail(props: { event: ReflectionEvent }) {
  const e = () => props.event;
  return (
    <>
      <Show when={e().changed_paths?.length}>
        <p class="m-mono m-quiet">{(e().changed_paths as string[]).join(", ")}</p>
      </Show>
      <Show when={e().error}>{(text) => <p class="m-failed">{text()}</p>}</Show>
      <Show when={e().phase === "candidate_rejected" && e().response_excerpt}>
        <details class="m-fold">
          <summary>Rejected output</summary>
          <pre class="m-output">{excerpt(e().response_excerpt!)}</pre>
        </details>
      </Show>
    </>
  );
}

function Evaluator(props: { label: string; record: EvaluatorRecord }) {
  const r = () => props.record;
  return (
    <details class="m-fold">
      <summary>
        {props.label} · predicted {fixed(r().score, 3)} by {r().resolved_model}
      </summary>
      <p>{r().rationale}</p>
      <dl class="m-kv">
        <dt>Asked for</dt>
        <dd>
          {r().requested_model} · {r().requested_family} · {r().requested_backend}
        </dd>
        <dt>Answered by</dt>
        <dd>
          {r().resolved_model} · {r().resolved_family} · {r().resolved_backend}
        </dd>
        <dt>Usage</dt>
        <dd>{usageText(r().usage)}</dd>
        <dt>Evaluation</dt>
        <dd class="m-mono m-break">{r().evaluation_id}</dd>
      </dl>
    </details>
  );
}

function Candidates(props: { result: SuccessfulReflectionResult }) {
  const r = () => props.result;
  const marker = (candidate: ReflectionCandidate) =>
    candidate.candidate_id === r().recommended_candidate_id ? "recommended" : candidate.candidate_id === r().provisional_candidate_id ? "best, not confirmed" : "";
  return (
    <div class="m-sub">
      <h4>Proposals · {r().candidates.length}</h4>
      <table class="m-table">
        <thead>
          <tr>
            <th scope="col">Proposal</th>
            <th scope="col">Writer</th>
            <th scope="col">Predicted</th>
            <th scope="col">Change</th>
            <th scope="col" />
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Current instructions</td>
            <td>—</td>
            <td class="m-mono">{fixed(r().baseline_score, 3)}</td>
            <td />
            <td />
          </tr>
          <For each={r().candidates}>
            {(candidate, index) => (
              <tr>
                <td>Proposal {index() + 1}</td>
                <td>{candidate.resolved_writer_model}</td>
                <td class="m-mono">{fixed(candidate.score, 3)}</td>
                <td class="m-mono">{signed(candidate.score_delta)}</td>
                <td>{marker(candidate)}</td>
              </tr>
            )}
          </For>
        </tbody>
      </table>
      <Evaluator label="Current instructions" record={r().baseline_evaluation} />
      <For each={r().candidates}>{(candidate, index) => <Evaluator label={`Proposal ${index() + 1}`} record={candidate.evaluation} />}</For>
    </div>
  );
}

function RejectedAttempts(props: { attempts: readonly GenerationAttempt[] }) {
  const failed = () => props.attempts.filter((attempt) => attempt.status === "failed");
  return (
    <Show when={failed().length > 0}>
      <div class="m-sub">
        <h4>Rejected proposal attempts · {failed().length}</h4>
        <For each={failed()}>
          {(attempt) => (
            <details class="m-fold">
              <summary>
                Attempt {attempt.number} · {attempt.error_type ?? "rejected"}: {attempt.error ?? "no message recorded"}
              </summary>
              <dl class="m-kv">
                <dt>Writer</dt>
                <dd>{attempt.resolved_model ?? attempt.requested_writer.label}</dd>
                <dt>Changed</dt>
                <dd class="m-mono">{attempt.changed_paths.join(", ") || "nothing reported"}</dd>
                <dt>Usage</dt>
                <dd>{usageText(attempt.usage)}</dd>
              </dl>
              <Show when={attempt.response_excerpt}>{(text) => <pre class="m-output">{excerpt(text())}</pre>}</Show>
            </details>
          )}
        </For>
      </div>
    </Show>
  );
}

function ArmEvidence(props: { label: string; arm: Arm | null }) {
  return (
    <Show when={props.arm}>
      {(arm) => (
        <details class="m-fold">
          <summary>
            {props.label}: {arm().status} · exit {arm().exit_code ?? "none"} · {arm().duration_seconds.toFixed(1)}s ·{" "}
            {plural(arm().artifact_changes?.length ?? 0, "changed file")}
          </summary>
          <Show when={arm().infrastructure_error}>{(text) => <p class="m-failed">{text()}</p>}</Show>
          <pre class="m-output">{arm().transcript || "No transcript output."}</pre>
          <For each={arm().artifact_changes ?? []}>
            {(change) => (
              <>
                <p class="m-mono">
                  {change.action} {change.path}
                </p>
                <pre class="m-output">{change.diff}</pre>
              </>
            )}
          </For>
        </details>
      )}
    </Show>
  );
}

function challengeTitle(challenge: Challenge): string {
  if (challenge.status !== "complete") return `A/B check ${words(challenge.status)}`;
  if (challenge.winner === "candidate") return "A/B check: the proposal won";
  if (challenge.winner === "baseline") return "A/B check: the current instructions won";
  return "A/B check: a tie";
}

/** The A/B check in the sandbox: both instruction sets ran the same authored task, and blinded
 * judges compared the results. */
function AbCheck(props: { challenge: Challenge }) {
  const c = () => props.challenge;
  return (
    <div class="m-sub">
      <h4 class={{ "m-failed": c().status !== "complete" }}>{challengeTitle(c())}</h4>
      <Show when={c().reason}>{(text) => <p>{text()}</p>}</Show>
      <Show when={c().task}>
        {(task) => (
          <dl class="m-kv">
            <dt>Task</dt>
            <dd>{task().prompt}</dd>
            <dt>Goal</dt>
            <dd>{task().goal}</dd>
            <Show when={task().judging_criteria?.length}>
              <dt>Judged on</dt>
              <dd>{task().judging_criteria!.join("; ")}</dd>
            </Show>
          </dl>
        )}
      </Show>
      <p class="m-quiet">
        {c().execution.model} via {c().execution.harness} {c().execution.harness_version} · {c().execution.effort ?? "default effort"} ·{" "}
        {c().execution.timeout_seconds}s limit · network {c().execution.network_enabled ? "on" : "off"}
      </p>
      <ArmEvidence label="Current instructions" arm={c().baseline} />
      <ArmEvidence label="Proposal" arm={c().candidate} />
      <Show when={c().judges.length > 0}>
        <details class="m-fold">
          <summary>Judges' verdicts · {c().judges.length}</summary>
          <For each={c().judges}>
            {(judge) => (
              <div class="m-reviewer">
                <strong>
                  Judge {judge.position} · {judge.resolved_model}: {words(judge.winner)}
                </strong>
                <Show when={judge.task_valid === false}>
                  <p class="m-failed">Invalid task: {judge.task_invalid_reason}</p>
                </Show>
                <p>{judge.rationale}</p>
                <ul class="m-mono">
                  <For each={judge.rubrics}>
                    {(rubric) => (
                      <li>
                        {rubric.rubric_id}: current {fixed(rubric.baseline_score)} / proposal {fixed(rubric.candidate_score)} ({words(rubric.winner)})
                      </li>
                    )}
                  </For>
                </ul>
              </div>
            )}
          </For>
        </details>
      </Show>
    </div>
  );
}

/**
 * Reflection, rewritten from the web UI's ReflectionActivity and ReflectionReview without its
 * editing and promotion: the activity, the proposals with the evaluator's predicted scores and
 * reasons, the A/B check, and the rejected attempts. Reviewing and promoting stays in the web UI.
 */
export function Reflection(props: { progress: ReflectingProgress | null; result: ReflectionResult | null; active: boolean }) {
  const success = () => {
    const result = props.result;
    return result && "baseline" in result ? result : null;
  };
  return (
    <Show when={props.progress || props.result} fallback={<p class="m-quiet">Reflection hasn't started.</p>}>
      <Show when={props.progress}>
        {(progress) => (
          <>
            <Activity
              label="Reflection activity"
              status={progress().status_message}
              startedAt={progress().started_at}
              active={props.active}
              events={progress().events}
              facts={eventFacts}
              detail={(event) => <EventDetail event={event} />}
            />
            <Show when={progress().total_attempts > 0}>
              <p class="m-quiet">
                {progress().attempted} of {progress().total_attempts} proposal attempts · {progress().valid} valid · {progress().rejected} rejected ·{" "}
                {progress().scored} scored · writer {progress().proposal_writer} · evaluator {progress().proposal_evaluator}
              </p>
            </Show>
          </>
        )}
      </Show>
      <Show when={props.result && !success()}>
        <p>{props.result!.reason}: reflection ended before the current instructions were evaluated.</p>
      </Show>
      <Show when={success()}>
        {(result) => (
          <>
            <Show when={result().reason}>{(text) => <p>{text()}</p>}</Show>
            <Candidates result={result()} />
            <Show when={result().challenge}>{(challenge) => <AbCheck challenge={challenge()} />}</Show>
            <RejectedAttempts attempts={result().generation_attempts} />
          </>
        )}
      </Show>
    </Show>
  );
}

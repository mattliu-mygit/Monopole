import { For, Show } from "solid-js";
import type { DataSelection, EffectiveRunConfig } from "../../../frontend/src/types";
import { shortTime } from "../format";
import { plural } from "./words";

type Cohort = Record<string, unknown> | null;
const field = <T,>(cohort: Cohort, key: string, kind: "string" | "number") => {
  const value = cohort?.[key];
  return typeof value === kind ? (value as T) : null;
};

function range(selection: DataSelection): string {
  if (!selection.since && !selection.until) return "Chosen sessions, any date";
  const day = (value: string | null) => (value ? value.slice(0, 10) : "…");
  return `${day(selection.since)} to ${day(selection.until)}${selection.timezone ? ` (${selection.timezone})` : ""}`;
}

/**
 * What the run evaluates and with what, rewritten from the web UI's PinnedSelectionAudit and
 * RunConfigAudit: the pinned sessions and turns, the judges, writer, and evaluator with their
 * budgets, the rubrics, and the catalog and pipeline versions.
 */
export function Setup(props: { selection: DataSelection | null; cohort: Cohort; config: EffectiveRunConfig | null }) {
  const turns = () => field<number>(props.cohort, "turn_count", "number");
  const sessions = () => field<number>(props.cohort, "session_count", "number");
  const pinnedAt = () => field<string>(props.cohort, "pinned_at", "string");
  return (
    <>
      <dl class="m-kv">
        <dt>Evidence</dt>
        <dd>{turns() !== null && sessions() !== null ? `${plural(turns()!, "turn")} across ${plural(sessions()!, "session")}` : "Not pinned yet"}</dd>
        <Show when={props.selection}>
          {(selection) => (
            <>
              <dt>Dates</dt>
              <dd>{range(selection())}</dd>
            </>
          )}
        </Show>
        <Show when={pinnedAt()}>
          {(at) => (
            <>
              <dt>Pinned</dt>
              <dd>{shortTime(at())}</dd>
            </>
          )}
        </Show>
        <Show when={field<string>(props.cohort, "cohort_id", "string")}>
          {(id) => (
            <>
              <dt>Cohort</dt>
              <dd class="m-mono m-break">{id()}</dd>
            </>
          )}
        </Show>
      </dl>
      <Show when={props.selection && props.selection.session_ids.length > 0}>
        <details class="m-fold">
          <summary>Session IDs · {props.selection!.session_ids.length}</summary>
          <ul class="m-mono">
            <For each={props.selection!.session_ids}>{(id) => <li>{id}</li>}</For>
          </ul>
        </details>
      </Show>
      <Show when={props.config} fallback={<p class="m-quiet">The settings are pinned when scoring starts.</p>}>
        {(config) => (
          <>
            <table class="m-table">
              <caption>Models</caption>
              <thead>
                <tr>
                  <th scope="col">Role</th>
                  <th scope="col">Model</th>
                  <th scope="col">Provider</th>
                  <th scope="col">Input limit</th>
                  <th scope="col">Window budget</th>
                </tr>
              </thead>
              <tbody>
                <For each={config().models.judges}>
                  {(judge) => (
                    <tr>
                      <td>Judge {judge.position}</td>
                      <td>{judge.label}</td>
                      <td>{judge.provider}</td>
                      <td>{judge.max_input_tokens?.toLocaleString() ?? "—"}</td>
                      <td>{judge.window_budget_tokens?.toLocaleString() ?? "—"}</td>
                    </tr>
                  )}
                </For>
                <For each={config().models.challenge_judges}>
                  {(judge) => (
                    <tr>
                      <td>A/B check judge {judge.position}</td>
                      <td>{judge.label}</td>
                      <td>{judge.provider}</td>
                      <td>{judge.max_input_tokens?.toLocaleString() ?? "—"}</td>
                      <td>—</td>
                    </tr>
                  )}
                </For>
                <tr>
                  <td>Proposal writer</td>
                  <td>{config().models.proposal_writer.label}</td>
                  <td>{config().models.proposal_writer.provider}</td>
                  <td>{config().models.proposal_writer.max_input_tokens.toLocaleString()}</td>
                  <td>—</td>
                </tr>
                <tr>
                  <td>Evaluator</td>
                  <td>{config().models.proposal_evaluator.label}</td>
                  <td>{config().models.proposal_evaluator.provider}</td>
                  <td>{config().models.proposal_evaluator.max_input_tokens.toLocaleString()}</td>
                  <td>—</td>
                </tr>
              </tbody>
            </table>
            <dl class="m-kv">
              <dt>Rubrics</dt>
              <dd>{config().rubrics.map((rubric) => `${rubric.label} (pass ${rubric.pass_threshold.toFixed(2)})`).join(", ") || "—"}</dd>
              <dt>Proposals</dt>
              <dd>
                at most {plural(config().candidate_budget, "attempt")}
                {config().force ? " · replaces existing Scores" : ""}
              </dd>
              <dt>Sliding windows</dt>
              <dd>
                {config().judging_context.digest_max_tokens?.toLocaleString() ?? "—"} digest tokens · {config().judging_context.finding_max_tokens?.toLocaleString() ?? "—"}{" "}
                finding tokens · {plural(config().judging_context.overlap_turns ?? 0, "turn")} overlap
              </dd>
              <dt>Versions</dt>
              <dd class="m-mono">
                pipeline {config().pipeline_version} · models {config().model_catalog_version} · rubrics {config().rubric_catalog_version}
              </dd>
            </dl>
            <Show when={config().selection_warnings.length > 0}>
              <ul class="m-list" aria-label="Warnings about these models">
                <For each={config().selection_warnings}>{(warning) => <li>{warning.message}</li>}</For>
              </ul>
            </Show>
          </>
        )}
      </Show>
    </>
  );
}

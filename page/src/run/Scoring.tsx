import { For, Show } from "solid-js";
import type { ScoringProgress, ScoringResult, ScoringTurnDetail } from "../../../frontend/src/types";
import { plural } from "./words";

function Turn(props: { detail: ScoringTurnDetail }) {
  const d = () => props.detail;
  const scores = () => Object.entries(d().scores as Record<string, number>);
  return (
    <details class="m-fold">
      <summary>
        <span class="m-clip">{d().user_input || d().trace_id}</span> · {plural(scores().length, "score")}
        <Show when={d().errors > 0}>
          <span class="m-failed"> · {plural(d().errors, "error")}</span>
        </Show>
      </summary>
      <p class="m-quiet">
        turn <span class="m-mono">{d().trace_id}</span> · session <span class="m-mono">{d().conversation_id}</span> · {d().model ?? "model unknown"} ·{" "}
        {d().tokens.toLocaleString()} tokens · {plural(d().tool_count, "tool call")}
      </p>
      <dl class="m-scores">
        <For each={scores()}>
          {([scorer, value]) => (
            <>
              <dt>{scorer}</dt>
              <dd class="m-mono">{value.toFixed(2)}</dd>
            </>
          )}
        </For>
      </dl>
    </details>
  );
}

/** Scoring, rewritten from the web UI's ScoringProgress: how far it got or what it wrote, and each
 * scored turn with its Scores. */
export function Scoring(props: { progress: ScoringProgress | null; result: ScoringResult | null }) {
  const details = () => props.result?.turn_details ?? props.progress?.turn_details ?? [];
  return (
    <Show when={props.progress || props.result} fallback={<p class="m-quiet">Scoring hasn't started.</p>}>
      <Show when={!props.result && props.progress}>
        {(progress) => (
          <>
            <p role="status">
              {progress().status_message} · {progress().scored} of {progress().total} turns scored · {progress().written} Scores written
            </p>
            <div class="m-progress" role="progressbar" aria-label="Turns scored" aria-valuemin={0} aria-valuemax={progress().total} aria-valuenow={progress().scored}>
              <i style={{ width: `${progress().total ? Math.round((progress().scored / progress().total) * 100) : 0}%` }} />
            </div>
          </>
        )}
      </Show>
      <Show when={props.result}>
        {(result) => (
          <dl class="m-stats">
            <div>
              <dt>Turns scored</dt>
              <dd>{result().turns_scored}</dd>
            </div>
            <div>
              <dt>Sessions scored</dt>
              <dd>{result().sessions_scored}</dd>
            </div>
            <div>
              <dt>Scores written</dt>
              <dd>{result().scores_written}</dd>
            </div>
            <div>
              <dt>Errors</dt>
              <dd class={{ "m-failed": result().errors > 0 }}>{result().errors}</dd>
            </div>
          </dl>
        )}
      </Show>
      <Show when={details().length > 0}>
        <div class="m-sub">
          <h4>Scored turns · {details().length}</h4>
          <For each={details()}>{(detail) => <Turn detail={detail} />}</For>
        </div>
      </Show>
    </Show>
  );
}

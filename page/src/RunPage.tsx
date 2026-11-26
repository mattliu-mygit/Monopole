import { createSignal, For, Show } from "solid-js";
import { Button, StatusDot } from "@prompting-time/ui";
import { meta, ratings, score } from "./format";
import type { RunDetail } from "./protocol";
import { ProposalRow } from "./ProposalRow";
import type { PageStore } from "./store";

const STAGE_WORDS = { done: "Done", running: "Running", failed: "Failed", not_started: "Not started" } as const;

/**
 * One run, read-only: its status and why it failed, its stages, its rubric ratings, its judges and
 * models, and its proposal. Delete asks Prompting Time, which confirms with the user first.
 */
export function RunPage(props: { store: PageStore; runId: string }) {
  const store = props.store;
  return (
    <Show
      when={store.state.run?.run_id === props.runId ? store.state.run : null}
      fallback={
        <Show when={store.state.loadError} fallback={<p class="m-hint" role="status">Reading the run…</p>}>
          {(error) => <p class="m-error" role="alert">{error()}</p>}
        </Show>
      }
    >
      {(run) => <RunBody store={store} run={run()} />}
    </Show>
  );
}

function RunBody(props: { store: PageStore; run: RunDetail }) {
  const run = () => props.run;
  const [deleting, setDeleting] = createSignal(false);
  const [deleteError, setDeleteError] = createSignal<string | null>(null);

  async function remove() {
    setDeleting(true);
    setDeleteError(null);
    try {
      await props.store.deleteRun(run().run_id);
    } catch (error) {
      setDeleteError(error instanceof Error ? error.message : String(error));
    } finally {
      setDeleting(false);
    }
  }

  return (
    <article class="m-run">
      <header class="m-run-header">
        <div class="m-run-title">
          <h2>{run().title}</h2>
          <div class="m-meta m-mono">{meta(run())}</div>
        </div>
        <span class="m-status">
          <StatusDot status={run().dot} />
          {run().label}
        </span>
        <Button onClick={remove} disabled={!run().deletable || deleting()} title={run().deletable ? undefined : "A running run can't be deleted"}>
          Delete run
        </Button>
      </header>
      <Show when={deleteError()}>{(error) => <p class="m-error" role="alert">{error()}</p>}</Show>
      <Show when={run().failure}>
        {(failure) => (
          <section class="m-section">
            <h3 class="m-failed">Stopped at {failure().stopped_at}</h3>
            <p>{failure().cause}</p>
          </section>
        )}
      </Show>
      <Show when={run().review_state === "review-needed"}>
        <p class="m-hint">Review and promote the proposal in Monopole's web UI.</p>
      </Show>
      <Show when={run().proposal}>{(proposal) => <ProposalRow proposal={proposal()} />}</Show>
      <section class="m-section">
        <h3>Stages</h3>
        <ul class="m-list">
          <For each={run().stages}>
            {(stage) => (
              <li>
                <strong>{stage.stage[0].toUpperCase() + stage.stage.slice(1)}</strong> · {STAGE_WORDS[stage.state]}
                <Show when={stage.message}>{(text) => <span class="m-quiet"> · {text()}</span>}</Show>
              </li>
            )}
          </For>
        </ul>
      </section>
      <Show when={run().rubrics.length > 0}>
        <section class="m-section">
          <h3>Rubrics{ratings(run()) ? ` · ${ratings(run())}` : ""}</h3>
          <ul class="m-list">
            <For each={run().rubrics}>
              {(rubric) => (
                <li>
                  {rubric.label} · {rubric.rating === null ? rubric.review_status : score(rubric.rating)}
                </li>
              )}
            </For>
          </ul>
        </section>
      </Show>
      <section class="m-section">
        <h3>Models</h3>
        <dl class="m-kv">
          <dt>Judges</dt>
          <dd>{run().judges.map((judge) => judge.label).join(", ") || "—"}</dd>
          <dt>Proposal writer</dt>
          <dd>{run().models.proposal_writer ?? "—"}</dd>
          <dt>Evaluator</dt>
          <dd>{run().models.proposal_evaluator ?? "—"}</dd>
          <Show when={run().selection}>
            {(selection) => (
              <>
                <dt>Sessions</dt>
                <dd class="m-mono">{selection().session_ids.join(", ") || "—"}</dd>
              </>
            )}
          </Show>
        </dl>
      </section>
    </article>
  );
}

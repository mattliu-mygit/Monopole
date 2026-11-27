import { createSignal, Show } from "solid-js";
import type { JSX } from "@solidjs/web";
import { Button, StatusDot } from "@prompting-time/ui";
import { meta, ratings } from "../format";
import { PageHeader } from "../PageHeader";
import type { RunDetail } from "../protocol";
import { ProposalRow } from "../ProposalRow";
import { runActions } from "../RunList";
import type { PageStore } from "../store";
import { Judging } from "./Judging";
import { Reflection } from "./Reflection";
import { Scoring } from "./Scoring";
import { Setup } from "./Setup";

const STATE_WORDS = { done: "Done", running: "Running", failed: "Failed", not_started: "Not started" } as const;

/** One stage of the run: its heading with its state and one-line summary, and everything it did. */
function Stage(props: { id: string; title: string; state?: keyof typeof STATE_WORDS; summary?: string | null; children: JSX.Element }) {
  return (
    <section class="m-stage" aria-labelledby={`stage-${props.id}`}>
      <h2 id={`stage-${props.id}`}>
        {props.title}
        <Show when={props.state}>
          {(state) => <span class={{ "m-stage-state": true, "m-failed": state() === "failed" }}> · {STATE_WORDS[state()]}</span>}
        </Show>
      </h2>
      <Show when={props.summary}>{(text) => <p class="m-quiet">{text()}</p>}</Show>
      {props.children}
    </section>
  );
}

/**
 * One run, with the detail of each stage that the web UI's run page shows, so a failed or slow run
 * can be debugged here: why it stopped first, its proposal, then its setup, scoring, judging, and
 * reflection, which refresh as Prompting Time reports changes. Continue, Rerun, and Delete ask
 * Prompting Time, which confirms with the user first.
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
  const [busy, setBusy] = createSignal(false);
  const [actionError, setActionError] = createSignal<string | null>(null);
  const actions = runActions(props.store, setActionError);
  const busyWhile = (work: (runId: string) => Promise<void>) => async () => {
    setBusy(true);
    try {
      await work(run().run_id);
    } finally {
      setBusy(false);
    }
  };
  const stage = (name: string) => run().stages.find((item) => item.stage === name);
  const rubricLabels = () => new Map((run().effective_config?.rubrics ?? []).map((rubric) => [rubric.id, rubric.label]));
  const label = (rubric: string) => rubricLabels().get(rubric) ?? rubric;
  const menu = () => [
    { value: "delete", label: "Delete run", disabled: !run().deletable || busy() || actions.held() },
    { value: "back", label: "Back to runs" },
    { value: "refresh", label: "Read again" },
  ];
  const onMenu = (value: string) => {
    if (value === "delete") void busyWhile(actions.remove)();
    else if (value === "back") props.store.open("/");
    else props.store.refresh();
  };

  return (
    <article class="m-run">
      <PageHeader
        title={run().title}
        menuLabel={`More actions for ${run().run_id}`}
        meta={
          <>
            <span class="m-status">
              <StatusDot status={run().dot} />
              {run().label}
            </span>
            <span class="m-mono"> · {meta(run())}</span>
          </>
        }
        menu={menu()}
        onMenu={onMenu}
      >
        <Show when={run().next_stage}>
          <Button variant="primary" onClick={busyWhile(actions.continueRun)} disabled={busy() || actions.held()}>
            Continue
          </Button>
        </Show>
        <Button onClick={busyWhile(actions.rerun)} disabled={busy() || actions.held()}>
          Rerun
        </Button>
        <Button onClick={() => actions.editAndRerun(run().run_id)}>Edit and rerun</Button>
      </PageHeader>
      <div class="m-column">
        <Show when={actionError()}>{(error) => <p class="m-error" role="alert">{error()}</p>}</Show>
        <Show when={run().failure}>
          {(failure) => (
            <section class="m-verdict" aria-label="Why the run stopped">
              <h2 class="m-failed">Stopped at {failure().stopped_at}</h2>
              <p>{failure().cause}</p>
            </section>
          )}
        </Show>
        <Show when={run().shortened}>
          <p class="m-hint">This run is too large to show whole here, so long texts and lists are cut. Monopole's web UI shows all of it.</p>
        </Show>
        <Show when={run().review_state === "review-needed"}>
          <p class="m-hint">Review and promote the proposal in Monopole's web UI.</p>
        </Show>
        <Show when={run().proposal}>{(proposal) => <ProposalRow proposal={proposal()} note />}</Show>
        <Stage id="setup" title="Sessions and settings">
          <Setup selection={run().data_selection} cohort={run().turn_cohort} config={run().effective_config} />
        </Stage>
        <Stage id="scoring" title="Scoring" state={stage("scoring")?.state} summary={stage("scoring")?.message}>
          <Scoring progress={run().scoring_progress} result={run().scoring_result} />
        </Stage>
        <Stage id="judging" title="Judging" state={stage("judging")?.state} summary={ratings(run())}>
          <Judging
            plan={run().judging_plan}
            progress={run().judging_progress}
            result={run().judging_result}
            active={run().status === "judging" && !run().current_stage_succeeded}
            label={label}
          />
        </Stage>
        <Stage id="reflecting" title="Reflection" state={stage("reflecting")?.state}>
          <Reflection progress={run().reflecting_progress} result={run().reflecting_result} active={run().status === "reflecting"} />
        </Stage>
      </div>
    </article>
  );
}

import { For, Show } from "solid-js";
import { score, signed } from "./format";
import { PageHeader } from "./PageHeader";
import type { RunDetail } from "./protocol";
import type { PageStore } from "./store";

/** One row of the comparison: its name and each run's value, null when the run has none. */
export interface CompareRow {
  label: string;
  values: [number | null, number | null];
}

const mean = (values: number[]) => (values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null);

/** Each rubric's rating, averaged over a run's sessions, in the order the runs first name them;
 * then the mean rating, and the proposal's predicted evaluator score for the current instructions
 * and for the proposal. */
export function compareRows(runs: [RunDetail, RunDetail]): CompareRow[] {
  const labels = new Map<string, string>();
  for (const run of runs) for (const review of run.rubrics) if (!labels.has(review.rubric)) labels.set(review.rubric, review.label);
  const rating = (run: RunDetail, rubric: string) =>
    mean(run.rubrics.filter((review) => review.rubric === rubric && review.rating !== null).map((review) => review.rating!));
  return [
    ...[...labels].map(([rubric, label]) => ({ label, values: [rating(runs[0], rubric), rating(runs[1], rubric)] as [number | null, number | null] })),
    { label: "Mean rating", values: [runs[0].mean_score, runs[1].mean_score] },
    { label: "Proposal: current instructions", values: [runs[0].proposal?.current ?? null, runs[1].proposal?.current ?? null] },
    { label: "Proposal: proposed", values: [runs[0].proposal?.proposed ?? null, runs[1].proposal?.proposed ?? null] },
  ];
}

const cell = (value: number | null) => (value === null ? "—" : score(value));

/**
 * Two runs checked in the list, side by side: each rubric's rating, the mean rating, and how each
 * run's proposal scored against the current instructions, by the evaluator's prediction, with the
 * second run's change from the first. Each run's ID opens it.
 */
export function Compare(props: { store: PageStore; runIds: [string, string] }) {
  const store = props.store;
  const runs = () => {
    const pair = store.state.compare;
    return pair && pair[0].run_id === props.runIds[0] && pair[1].run_id === props.runIds[1] ? pair : null;
  };
  return (
    <section class="m-tab">
      <PageHeader
        title="Compare runs"
        menu={[
          { value: "back", label: "Back to runs" },
          { value: "refresh", label: "Read again" },
        ]}
        onMenu={(value) => (value === "back" ? store.open("/") : store.refresh())}
      />
      <div class="m-column">
        <Show
          when={runs()}
          fallback={
            <Show when={store.state.loadError} fallback={<p class="m-hint" role="status">Reading the runs…</p>}>
              {(error) => <p class="m-error" role="alert">{error()}</p>}
            </Show>
          }
        >
          {(pair) => (
            <>
              <ol class="m-list">
                <For each={pair()}>
                  {(run) => (
                    <li>
                      <button type="button" class="m-link m-mono" onClick={() => store.open(`/runs/${encodeURIComponent(run.run_id)}`)}>
                        {run.run_id}
                      </button>{" "}
                      · {run.label} · <span class="m-clip">{run.title}</span>
                    </li>
                  )}
                </For>
              </ol>
              <table class="m-table m-wide">
                <caption>Scores, higher is better</caption>
                <thead>
                  <tr>
                    <th scope="col">Rubric or score</th>
                    <th scope="col" class="m-mono">{pair()[0].run_id}</th>
                    <th scope="col" class="m-mono">{pair()[1].run_id}</th>
                    <th scope="col">Change</th>
                  </tr>
                </thead>
                <tbody>
                  <For each={compareRows(pair())}>
                    {(row) => (
                      <tr>
                        <th scope="row">{row.label}</th>
                        <td>{cell(row.values[0])}</td>
                        <td>{cell(row.values[1])}</td>
                        <td>{row.values[0] === null || row.values[1] === null ? "—" : signed(row.values[1] - row.values[0])}</td>
                      </tr>
                    )}
                  </For>
                </tbody>
              </table>
              <p class="m-hint">Proposal scores are the evaluator's predictions, not a verification run.</p>
            </>
          )}
        </Show>
      </div>
    </section>
  );
}

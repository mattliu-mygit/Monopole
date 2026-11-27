import { For, Show } from "solid-js";
import { Button } from "@prompting-time/ui";
import { score, shortTime, signed } from "./format";
import { PageHeader } from "./PageHeader";
import type { Analysis } from "./protocol";
import type { PageStore } from "./store";

type Trend = Analysis["trends"][number];
type Summary = Analysis["summary"][number];

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;

function TrendRow(props: { trend: Trend }) {
  const t = () => props.trend;
  const regression = () => t().direction === "regression";
  return (
    <li class="m-trend">
      <strong>{t().scorer}</strong>
      <span class={regression() ? "m-failed" : "m-good"}>{regression() ? "Regression" : "Improvement"}</span>
      <span class="m-mono">
        {score(t().older_mean)} → {score(t().recent_mean)}
      </span>
      <span class="m-mono">{signed(t().delta)}</span>
      <span class="m-quiet">
        {t().significant ? "significant" : "not significant"} · n={t().sample_count}
      </span>
    </li>
  );
}

function ScoreTable(props: { summary: readonly Summary[]; trends: readonly Trend[] }) {
  const trendOf = (scorer: string) => props.trends.find((trend) => trend.scorer === scorer);
  return (
    <table class="m-table m-wide">
      <thead>
        <tr>
          <th scope="col">Scorer</th>
          <th scope="col">Mean</th>
          <th scope="col">Pass rate</th>
          <th scope="col">95% interval</th>
          <th scope="col">Count</th>
          <th scope="col">Recent change</th>
        </tr>
      </thead>
      <tbody>
        <For each={props.summary}>
          {(row) => (
            <tr>
              <th scope="row">{row.scorer}</th>
              <td class="m-mono">{score(row.mean)}</td>
              <td class="m-mono">{row.pass_rate === null ? "—" : percent(row.pass_rate)}</td>
              <td class="m-mono">
                {score(row.ci[0])}–{score(row.ci[1])}
              </td>
              <td>
                {row.count}
                {row.confident ? "" : " · too few to trust"}
              </td>
              <td class="m-mono">{trendOf(row.scorer) ? signed(trendOf(row.scorer)!.delta) : "—"}</td>
            </tr>
          )}
        </For>
      </tbody>
    </table>
  );
}

/** The A/B leaderboard as one table: a row per scorer, a column per instruction configuration, and
 * the highest mean in each row in bold. */
function ByConfiguration(props: { entries: Analysis["ab_leaderboard"] }) {
  const scorers = () => [...new Set(props.entries.flatMap((entry) => Object.keys(entry.scores)))].sort();
  const best = (scorer: string) => Math.max(...props.entries.map((entry) => entry.scores[scorer]?.mean ?? -Infinity));
  return (
    <div class="m-table-wrap">
      <table class="m-table m-wide">
        <thead>
          <tr>
            <th scope="col">Scorer</th>
            <For each={props.entries}>
              {(entry) => (
                <th scope="col" class="m-mono" title={entry.config_version}>
                  {entry.config_version.slice(0, 8)}
                  <div class="m-quiet">{entry.evaluated_target_count} evaluated</div>
                </th>
              )}
            </For>
          </tr>
        </thead>
        <tbody>
          <For each={scorers()}>
            {(scorer) => (
              <tr>
                <th scope="row">{scorer}</th>
                <For each={props.entries}>
                  {(entry) => (
                    <td class="m-mono">
                      <Show when={entry.scores[scorer]} fallback="—">
                        {(cell) => (
                          <>
                            <span class={{ "m-best": props.entries.length > 1 && cell().mean === best(scorer) }}>{score(cell().mean)}</span>
                            <span class="m-quiet"> n={cell().count}</span>
                          </>
                        )}
                      </Show>
                    </td>
                  )}
                </For>
              </tr>
            )}
          </For>
        </tbody>
      </table>
    </div>
  );
}

/**
 * Trends, rewritten from the web UI's Analysis and Dashboard: which scores moved recently, every
 * scorer's mean, pass rate, and interval, the scores by instruction configuration, and coaching.
 * The server reads every Score's statistics from CES, which can take minutes, so the page reads
 * when Trends first opens and on Read again.
 */
export function Trends(props: { store: PageStore }) {
  const store = props.store;
  const meta = () =>
    store.state.readingTrends
      ? "Reading every Score from CES; this can take a few minutes"
      : store.state.trendsReadAt
        ? `Read ${shortTime(store.state.trendsReadAt)}`
        : "Not read yet";
  return (
    <section class="m-tab">
      <PageHeader title="Trends" meta={meta()} menu={[{ value: "tools", label: "Check for regressions in Tools" }]} onMenu={() => store.open("/tools")}>
        <Button onClick={() => store.refresh()} disabled={store.state.readingTrends}>
          {store.state.readingTrends ? "Reading…" : "Read again"}
        </Button>
      </PageHeader>
      <div class="m-column">
        <Show when={store.state.loadError}>
          {(error) => (
            <p class="m-error" role="alert">
              {error()}
            </p>
          )}
        </Show>
        <Show when={store.state.trends} fallback={<Show when={store.state.readingTrends}><p class="m-hint" role="status">Reading trends…</p></Show>}>
          {(analysis) => (
            <Show when={analysis().summary.length > 0} fallback={<p class="m-hint">No Scores yet. Finish a run to see trends here.</p>}>
              <section class="m-stage" aria-labelledby="trends-changes">
                <h2 id="trends-changes">Recent changes · {analysis().trends.length}</h2>
                <p class="m-quiet">Each scorer's last 7 rated days against the 7 days before them; changes of 0.20 or more show.</p>
                <Show when={analysis().trends.length > 0} fallback={<p class="m-hint">No scorer moved enough to show.</p>}>
                  <ul class="m-trends" aria-label="Recent changes">
                    <For each={analysis().trends}>{(trend) => <TrendRow trend={trend} />}</For>
                  </ul>
                </Show>
              </section>
              <section class="m-stage" aria-labelledby="trends-scores">
                <h2 id="trends-scores">Scores · {analysis().summary.length}</h2>
                <div class="m-table-wrap">
                  <ScoreTable summary={analysis().summary} trends={analysis().trends} />
                </div>
              </section>
              <section class="m-stage" aria-labelledby="trends-configs">
                <h2 id="trends-configs">By instruction configuration · {analysis().ab_leaderboard.length}</h2>
                <Show
                  when={analysis().ab_leaderboard.length > 0}
                  fallback={<p class="m-hint">Sessions from more than one configuration are needed to compare them.</p>}
                >
                  <ByConfiguration entries={analysis().ab_leaderboard} />
                </Show>
              </section>
              <section class="m-stage" aria-labelledby="trends-coaching">
                <h2 id="trends-coaching">Coaching</h2>
                <pre class="m-prose">{analysis().coaching_markdown || "No coaching recommendations yet."}</pre>
              </section>
            </Show>
          )}
        </Show>
      </div>
    </section>
  );
}

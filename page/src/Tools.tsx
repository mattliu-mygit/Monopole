import { createSignal, For, Show } from "solid-js";
import type { JSX } from "@solidjs/web";
import { Button, Select, StatusDot, Tabs, TextField } from "@prompting-time/ui";
import { shortTime } from "./format";
import type { ToolRun, ToolTable } from "./protocol";
import type { PageStore } from "./store";

const INSIGHTS = [
  { value: "subagents", label: "Subagents" },
  { value: "judges", label: "Judge agreement" },
  { value: "segments", label: "Segments" },
];

/** A tool's last run: running since when, or what it printed when it finished. */
function LastRun(props: { run: ToolRun | undefined }) {
  return (
    <Show when={props.run}>
      {(run) => (
        <>
          <p class="m-status">
            <StatusDot status={run().state === "running" ? "running" : run().state === "done" ? "success" : "error"} />
            {run().state === "running"
              ? `Running since ${shortTime(run().started_at)}`
              : `${run().state === "done" ? "Finished" : run().state === "failed" ? "Failed" : "Stopped"} ${shortTime(run().finished_at ?? run().started_at)}`}
          </p>
          <Show when={run().tables?.length} fallback={<Show when={run().output}>{(output) => <pre class="m-output">{output()}</pre>}</Show>}>
            <For each={run().tables!}>{(table) => <ResultTable table={table} />}</For>
          </Show>
        </>
      )}
    </Show>
  );
}

function ResultTable(props: { table: ToolTable }) {
  return (
    <div class="m-table-wrap">
      <table class="m-table">
        <caption>{props.table.title}</caption>
        <thead>
          <tr>
            <For each={props.table.columns}>{(column) => <th scope="col">{column}</th>}</For>
          </tr>
        </thead>
        <tbody>
          <For each={props.table.rows}>
            {(row) => (
              <tr>
                <For each={row}>{(cell) => <td>{cell}</td>}</For>
              </tr>
            )}
          </For>
        </tbody>
      </table>
    </div>
  );
}

function ToolCard(props: { title: string; note: string; children: JSX.Element; error: string | null; run: ToolRun | undefined }) {
  return (
    <div class="m-panel">
      <h3>{props.title}</h3>
      <p class="m-quiet">{props.note}</p>
      {props.children}
      <Show when={props.error}>
        {(text) => (
          <p class="m-error" role="alert">
            {text()}
          </p>
        )}
      </Show>
      <LastRun run={props.run} />
    </div>
  );
}

/**
 * The commands that were terminal-only: score recent sessions, backfill a date range, check for
 * regressions, and analyze's three insight tables. Score, Backfill, and Check now write Scores or
 * post alerts, so Prompting Time confirms them first; the insights only read. Each tool's last run
 * stays shown after the page reloads, and a running one can't start twice.
 */
export function Tools(props: { store: PageStore }) {
  const store = props.store;
  const [errors, setErrors] = createSignal<Record<string, string | null>>({});
  const [hours, setHours] = createSignal("24");
  const [start, setStart] = createSignal("");
  const [end, setEnd] = createSignal("");
  const [kind, setKind] = createSignal("subagents");
  // The tools this page started that haven't replied yet, including while Prompting Time asks to
  // confirm; a tool started elsewhere shows as running from its last run.
  const [pending, setPending] = createSignal<string[]>([]);
  const last = (tool: string) => store.state.tools?.tools[tool];
  const running = (tool: string) => pending().includes(tool) || last(tool)?.state === "running";

  async function run(tool: string, args: string[]) {
    setErrors((all) => ({ ...all, [tool]: null }));
    setPending((all) => [...all, tool]);
    try {
      await store.runTool(tool, args);
    } catch (failure) {
      setErrors((all) => ({ ...all, [tool]: failure instanceof Error ? failure.message : String(failure) }));
    } finally {
      setPending((all) => all.filter((other) => other !== tool));
    }
  }

  return (
    <section class="m-tools">
      <header class="m-run-header">
        <div class="m-run-title">
          <h2>Tools</h2>
        </div>
      </header>
      <Show when={store.state.loadError}>
        {(text) => (
          <p class="m-error" role="alert">
            {text()}
          </p>
        )}
      </Show>
      <div class="m-grid">
        <ToolCard title="Score recent sessions" note="Writes deterministic Scores for turns that have none. Same as score." error={errors().score ?? null} run={last("score")}>
          <Select
            label="Since"
            options={[
              { value: "24", label: "Last 24 hours" },
              { value: "168", label: "Last 7 days" },
            ]}
            value={hours()}
            onChange={(event) => setHours(event.currentTarget.value)}
          />
          <Button variant="primary" disabled={running("score") || store.held()} onClick={() => void run("score", ["--hours", hours()])}>
            {running("score") ? "Running…" : "Score"}
          </Button>
        </ToolCard>
        <ToolCard
          title="Backfill a date range"
          note="Scores past days, and fills the day field trends need. Same as backfill."
          error={errors().backfill ?? null}
          run={last("backfill")}
        >
          <TextField label="Start" placeholder="2026-10-01" value={start()} onInput={(event) => setStart(event.currentTarget.value)} />
          <TextField label="End" placeholder="2026-10-31" value={end()} onInput={(event) => setEnd(event.currentTarget.value)} />
          <Button
            variant="primary"
            disabled={running("backfill") || store.held() || !start().trim()}
            onClick={() => void run("backfill", ["--start", start().trim(), ...(end().trim() ? ["--end", end().trim()] : [])])}
          >
            {running("backfill") ? "Running…" : "Backfill"}
          </Button>
        </ToolCard>
        <ToolCard
          title="Regression alerts"
          note="Significant score drops. Same as monitor; new alerts go to the webhook in Monopole's settings."
          error={errors().monitor ?? null}
          run={last("monitor")}
        >
          <Button variant="primary" disabled={running("monitor") || store.held()} onClick={() => void run("monitor", [])}>
            {running("monitor") ? "Running…" : "Check now"}
          </Button>
        </ToolCard>
      </div>
      <div class="m-panel">
        <Tabs label="Insights" tabs={INSIGHTS} value={kind()} onChange={setKind}>
          <div class="m-section">
            <p class="m-quiet">analyze's insight tables, read-only.</p>
            <div class="m-actions">
              <Button disabled={running("insights") || store.held()} onClick={() => void run("insights", ["--kind", kind()])}>
                {running("insights") ? "Reading…" : "Read"}
              </Button>
            </div>
            <Show when={errors().insights}>
              {(text) => (
                <p class="m-error" role="alert">
                  {text()}
                </p>
              )}
            </Show>
            <LastRun run={last("insights")?.args.kind === kind() ? last("insights") : undefined} />
          </div>
        </Tabs>
      </div>
    </section>
  );
}

import { For, Match, Switch } from "solid-js";
import { blocking, statusLine } from "./format";
import { NewRun } from "./NewRun";
import { Providers } from "./Providers";
import { RunList } from "./RunList";
import { RunPage } from "./run/RunPage";
import type { PageStore } from "./store";
import { Tools } from "./Tools";
import { Trends } from "./Trends";

/**
 * The page's views, by route, with Monopole's own tab sidebar, since Prompting Time folds its
 * sidebar while a plugin page is open: the run list at / (and one run at /runs/<id>), Trends at
 * /trends, New run at /new (and /new?from=<id> for Edit and rerun), /providers, and /tools.
 * Prompting Time's `open {route}` selects the tab.
 */
export function App(props: { store: PageStore }) {
  const store = props.store;
  const kind = () => store.place().kind;
  const runId = () => {
    const place = store.place();
    return place.kind === "run" ? place.runId : null;
  };
  const signedOut = () => store.state.providers?.providers.filter(blocking).length ?? 0;
  const tabs = () => [
    { kind: "runs", route: "/", label: "Runs", count: store.state.runs ? String(store.state.runs.runs.length) : "" },
    { kind: "trends", route: "/trends", label: "Trends", count: "" },
    { kind: "new", route: "/new", label: "New run", count: "" },
    { kind: "providers", route: "/providers", label: "Providers", count: signedOut() ? `${signedOut()} signed out` : "" },
    { kind: "tools", route: "/tools", label: "Tools", count: "" },
  ];
  const current = (tab: string) => kind() === tab || (tab === "runs" && kind() === "run");
  return (
    <div class="m-page">
      <aside class="m-side">
        <div class="m-brand">
          <strong>Monopole</strong>
          <span>{statusLine(store.state.runs)}</span>
        </div>
        <nav aria-label="Monopole">
          <For each={tabs()}>
            {(tab) => (
              <button type="button" class="m-tab-button" aria-current={current(tab.kind) ? "page" : undefined} onClick={() => store.open(tab.route)}>
                <span>{tab.label}</span>
                <span class={{ "m-count": true, "m-failed": tab.kind === "providers" && signedOut() > 0 }}>{tab.count}</span>
              </button>
            )}
          </For>
        </nav>
      </aside>
      <main class="m-main">
        <Switch>
          <Match when={runId()} keyed>
            {(id: string) => <RunPage store={store} runId={id} />}
          </Match>
          <Match when={kind() === "trends"}>
            <Trends store={store} />
          </Match>
          <Match when={kind() === "new"}>
            <NewRun store={store} />
          </Match>
          <Match when={kind() === "providers"}>
            <Providers store={store} />
          </Match>
          <Match when={kind() === "tools"}>
            <Tools store={store} />
          </Match>
          <Match when={true}>
            <RunList store={store} />
          </Match>
        </Switch>
      </main>
    </div>
  );
}

import { For, Match, Show, Switch } from "solid-js";
import { NewRun } from "./NewRun";
import { Providers } from "./Providers";
import { RunList } from "./RunList";
import { RunPage } from "./RunPage";
import type { PageStore } from "./store";
import { Tools } from "./Tools";

const NAV = [
  { kind: "runs", route: "/", label: "Runs" },
  { kind: "new", route: "/new", label: "New run" },
  { kind: "providers", route: "/providers", label: "Providers" },
  { kind: "tools", route: "/tools", label: "Tools" },
] as const;

/** The page's views, by route: the run list at /, one run at /runs/<id>, New run at /new (and
 * /new?from=<id> for Edit and rerun), /providers, and /tools. */
export function App(props: { store: PageStore }) {
  const store = props.store;
  const kind = () => store.place().kind;
  const runId = () => {
    const place = store.place();
    return place.kind === "run" ? place.runId : null;
  };
  return (
    <div class="m-page">
      <nav class="m-nav" aria-label="Monopole">
        <For each={NAV}>
          {(item) => (
            <button
              type="button"
              aria-current={kind() === item.kind || (item.kind === "runs" && kind() === "run") ? "page" : undefined}
              onClick={() => store.open(item.route)}
            >
              {item.label}
            </button>
          )}
        </For>
      </nav>
      <Show when={store.state.loadError && kind() === "runs"}>
        <p class="m-error" role="alert">
          {store.state.loadError}
        </p>
      </Show>
      <Switch>
        <Match when={runId()} keyed>
          {(id: string) => <RunPage store={store} runId={id} />}
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
    </div>
  );
}

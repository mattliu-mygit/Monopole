import { Match, Show, Switch } from "solid-js";
import { Providers } from "./Providers";
import { RunList } from "./RunList";
import { RunPage } from "./RunPage";
import type { PageStore } from "./store";

/** The page's views, by route: the run list at /, one run at /runs/<id>, and /providers. */
export function App(props: { store: PageStore }) {
  const store = props.store;
  const runId = () => {
    const place = store.place();
    return place.kind === "run" ? place.runId : null;
  };
  return (
    <div class="m-page">
      <nav class="m-nav" aria-label="Monopole">
        <button type="button" aria-current={store.place().kind !== "providers" ? "page" : undefined} onClick={() => store.open("/")}>
          Runs
        </button>
        <button type="button" aria-current={store.place().kind === "providers" ? "page" : undefined} onClick={() => store.open("/providers")}>
          Providers
        </button>
      </nav>
      <Show when={store.state.loadError && store.place().kind === "runs"}>
        <p class="m-error" role="alert">
          {store.state.loadError}
        </p>
      </Show>
      <Switch>
        <Match when={runId()} keyed>
          {(id: string) => <RunPage store={store} runId={id} />}
        </Match>
        <Match when={store.place().kind === "providers"}>
          <Providers store={store} />
        </Match>
        <Match when={true}>
          <RunList store={store} />
        </Match>
      </Switch>
    </div>
  );
}

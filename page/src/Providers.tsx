import { For, Show } from "solid-js";
import { Button, StatusDot, type Status } from "@prompting-time/ui";
import { PROVIDER_WORDS, shortTime } from "./format";
import type { ProviderCheck } from "./protocol";
import type { PageStore } from "./store";

/** Red for a provider that stops runs, green for one that works, no color otherwise. A missing
 * W&B key stops every run, since CES needs it; a missing OpenAI key stops only runs that pick
 * OpenAI models. */
function dotFor(provider: ProviderCheck["providers"][number]): Status {
  if (provider.state === "signed_in") return "success";
  if (provider.state === "signed_out" || (provider.state === "no_key" && provider.id === "wandb")) return "error";
  return "idle";
}

/** Whether each provider Monopole calls is signed in, from the last check, with Check all. */
export function Providers(props: { store: PageStore }) {
  const store = props.store;
  return (
    <section class="m-providers">
      <header class="m-run-header">
        <div class="m-run-title">
          <h2>Providers</h2>
          <div class="m-meta">
            {store.state.providers?.checked_at ? `Checked ${shortTime(store.state.providers.checked_at)}` : "Not checked yet"}
          </div>
        </div>
        <Button variant="primary" onClick={() => void store.checkProviders()} disabled={store.state.checking}>
          {store.state.checking ? "Checking…" : "Check all"}
        </Button>
      </header>
      <Show when={store.state.loadError}>
        {(error) => (
          <p class="m-error" role="alert">
            {error()}
          </p>
        )}
      </Show>
      <Show when={store.state.providers?.providers.length}>
        <ul class="m-list m-provider-list">
          <For each={store.state.providers!.providers}>
            {(provider) => (
              <li>
                <span class="m-status">
                  <StatusDot status={dotFor(provider)} />
                  <strong>{provider.title}</strong>
                </span>
                <span>{PROVIDER_WORDS[provider.state]}</span>
                <Show when={provider.detail}>{(detail) => <span class="m-quiet">{detail()}</span>}</Show>
              </li>
            )}
          </For>
        </ul>
      </Show>
      <p class="m-hint">Checks run each CLI's status command and ask W&B and OpenAI to list models; they call no model.</p>
    </section>
  );
}

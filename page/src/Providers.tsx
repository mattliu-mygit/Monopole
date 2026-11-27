import { createSignal, For, Show } from "solid-js";
import { Button, StatusDot, TextField, type Status } from "@prompting-time/ui";
import { PROVIDER_WORDS, shortTime } from "./format";
import type { ProviderCheck, SignIn } from "./protocol";
import type { PageStore } from "./store";

type Provider = ProviderCheck["providers"][number];

const CLIS = new Set(["claude", "codex", "devin"]);
const KEYS = new Set(["wandb", "openai"]);
const COMMANDS: Record<string, string> = { claude: "claude auth login", codex: "codex login", devin: "devin auth login" };
export const CODEX_WARNING = "Starting sign-in removes the current Codex login, even if you cancel.";

/** Red for a provider that stops runs, green for one that works, no color otherwise. A missing
 * W&B key stops every run, since CES needs it; a missing OpenAI key stops only runs that pick
 * OpenAI models. */
function dotFor(provider: Provider): Status {
  if (provider.state === "signed_in") return "success";
  if (provider.state === "signed_out" || (provider.state === "no_key" && provider.id === "wandb")) return "error";
  return "idle";
}

const SIGN_IN_WORDS: Record<SignIn["state"], string> = {
  waiting: "Waiting for the browser…",
  signed_in: "Signed in.",
  failed: "The sign-in failed.",
  timed_out: "The sign-in timed out.",
  cancelled: "You cancelled the sign-in.",
  stopped: "The sign-in stopped without finishing.",
};

/**
 * Whether each provider Monopole calls is signed in, from the last check, with Check all. A CLI
 * that isn't signed in gets Sign in, which runs its own login after Prompting Time's confirmation
 * and waits for the browser; Codex warns first, since starting its login removes the current one.
 * W&B and OpenAI get a field for a key, which Monopole checks and saves; the page never shows it.
 */
export function Providers(props: { store: PageStore }) {
  const store = props.store;
  // The provider whose Codex warning or key field is open.
  const [asking, setAsking] = createSignal<string | null>(null);
  const [signingIn, setSigningIn] = createSignal<string | null>(null);
  const [error, setError] = createSignal<string | null>(null);
  const [saved, setSaved] = createSignal<string | null>(null);

  async function signIn(provider: string) {
    setAsking(null);
    setError(null);
    setSigningIn(provider);
    try {
      await store.signIn(provider);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setSigningIn(null);
    }
  }

  const signInState = () => store.state.providers?.sign_in ?? null;
  const waiting = () => {
    const found = signInState();
    return found?.state === "waiting" ? found.provider : signingIn();
  };

  return (
    <section class="m-providers">
      <header class="m-run-header">
        <div class="m-run-title">
          <h2>Providers</h2>
          <div class="m-meta">
            {store.state.providers?.checked_at ? `Checked ${shortTime(store.state.providers.checked_at)}` : "Not checked yet"}
          </div>
        </div>
        <Button variant="primary" onClick={() => void store.checkProviders()} disabled={store.state.checking || store.held()}>
          {store.state.checking ? "Checking…" : "Check all"}
        </Button>
      </header>
      <Show when={store.state.loadError ?? error()}>
        {(text) => (
          <p class="m-error" role="alert">
            {text()}
          </p>
        )}
      </Show>
      <Show when={saved()}>{(text) => <p class="m-hint">{text()}</p>}</Show>
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
                <span class="m-actions">
                  <Show when={provider.detail}>{(detail) => <span class="m-quiet">{detail()}</span>}</Show>
                  <Show when={CLIS.has(provider.id) && provider.state !== "not_installed" && (provider.state !== "signed_in" || provider.id === "codex")}>
                    <Button
                      disabled={waiting() !== null || store.held()}
                      onClick={() => (provider.id === "codex" ? setAsking("codex") : void signIn(provider.id))}
                    >
                      {provider.state === "signed_in" ? "Sign in again" : "Sign in"}
                    </Button>
                  </Show>
                  <Show when={KEYS.has(provider.id)}>
                    <Button onClick={() => setAsking(provider.id)}>{provider.state === "signed_in" ? "Replace key" : "Add key"}</Button>
                  </Show>
                </span>
              </li>
            )}
          </For>
        </ul>
      </Show>
      <Show when={asking() === "codex"}>
        <div class="m-panel" role="dialog" aria-label="Sign in to Codex">
          <h3>Sign in to Codex</h3>
          <p class="m-warning">{CODEX_WARNING}</p>
          <p class="m-quiet">Monopole runs {COMMANDS.codex}; it finishes when the browser returns to localhost:1455.</p>
          <div class="m-actions">
            <Button variant="primary" onClick={() => void signIn("codex")} disabled={store.held()}>
              Start sign-in
            </Button>
            <Button onClick={() => setAsking(null)}>Cancel</Button>
          </div>
        </div>
      </Show>
      <Show when={asking() !== null && KEYS.has(asking()!)}>
        <KeyPanel
          store={store}
          provider={store.state.providers!.providers.find((p) => p.id === asking())!}
          onDone={(text) => {
            setAsking(null);
            setSaved(text);
          }}
          onCancel={() => setAsking(null)}
        />
      </Show>
      <Show when={waiting()}>{(provider) => <SignInPanel store={store} provider={provider()} state={signInState()} />}</Show>
      <Show when={!waiting() && signInState()}>
        {(last) => (
          <p class="m-hint">
            Last sign-in, {COMMANDS[last().provider] ?? last().provider}: {SIGN_IN_WORDS[last().state]}
            {last().message && last().state !== "signed_in" ? ` ${last().message}` : ""}
          </p>
        )}
      </Show>
      <p class="m-hint">Checks run each CLI's status command and ask W&B and OpenAI to list models; they call no model.</p>
    </section>
  );
}

function SignInPanel(props: { store: PageStore; provider: string; state: SignIn | null }) {
  const [code, setCode] = createSignal("");
  const [error, setError] = createSignal<string | null>(null);
  const current = () => (props.state?.provider === props.provider ? props.state : null);

  async function attempt(work: () => Promise<unknown>) {
    setError(null);
    try {
      await work();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    }
  }

  return (
    <div class="m-panel" role="dialog" aria-label={`Signing in with ${COMMANDS[props.provider]}`}>
      <h3>Signing in with {COMMANDS[props.provider]}</h3>
      <p class="m-status">
        <StatusDot status="running" />
        Waiting for the browser… Finish signing in there; this closes on its own.
      </p>
      <Show when={current()?.link}>
        {(link) => (
          <p class="m-quiet">
            Browser didn't open? Copy this link into it: <span class="m-mono">{link()}</span>
          </p>
        )}
      </Show>
      <div class="m-actions">
        <TextField label="Code from the browser, if it shows one" value={code()} onInput={(event) => setCode(event.currentTarget.value)} />
        <Button
          disabled={!code().trim()}
          onClick={() =>
            void attempt(async () => {
              await props.store.sendSignInCode(props.provider, code().trim());
              setCode("");
            })
          }
        >
          Submit code
        </Button>
        <Button onClick={() => void attempt(() => props.store.cancelSignIn(props.provider))}>Cancel sign-in</Button>
      </div>
      <Show when={error()}>
        {(text) => (
          <p class="m-error" role="alert">
            {text()}
          </p>
        )}
      </Show>
    </div>
  );
}

function KeyPanel(props: { store: PageStore; provider: Provider; onDone: (saved: string) => void; onCancel: () => void }) {
  const [key, setKey] = createSignal("");
  const [saving, setSaving] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);

  async function save() {
    setSaving(true);
    setError(null);
    try {
      const saved = await props.store.setKey(props.provider.id, key().trim());
      setKey("");
      props.onDone(saved);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div class="m-panel" role="dialog" aria-label={`${props.provider.title} key`}>
      <h3>{props.provider.state === "signed_in" ? "Replace" : "Add"} the {props.provider.title} key</h3>
      <p class="m-quiet">
        Monopole checks the key with GET /v1/models, which costs nothing, then saves it. Afterwards the page shows only that a key works, never
        the key.
      </p>
      <input
        class="m-key"
        type="password"
        aria-label="API key"
        autocomplete="off"
        value={key()}
        onInput={(event) => setKey(event.currentTarget.value)}
      />
      <Show when={error()}>
        {(text) => (
          <p class="m-error" role="alert">
            {text()}
          </p>
        )}
      </Show>
      <div class="m-actions">
        <Button variant="primary" onClick={save} disabled={saving() || props.store.held() || !key().trim()}>
          {saving() ? "Checking…" : "Check and save"}
        </Button>
        <Button onClick={props.onCancel}>Cancel</Button>
      </div>
    </div>
  );
}

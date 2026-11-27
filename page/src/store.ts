import { createSignal, createStore } from "solid-js";
import {
  CallError,
  PROTOCOL_VERSION,
  readHostMessage,
  type Catalog,
  type ProviderCheck,
  type RunConfig,
  type RunDetail,
  type RunList,
  type Selection,
  type SessionList,
  type ToolRun,
  type ToolRuns,
} from "./protocol";

/** How the page reaches its host: in Prompting Time, the parent window; in tests, the fake host. */
export interface HostLink {
  post(message: unknown): void;
  /** Calls receive with each message the host sends, until the returned function is called. */
  listen(receive: (data: unknown) => void): () => void;
}

/** The parent window. Only messages from it count, and the page posts only to it. The page's
 * sandbox gives it an opaque origin, so it posts to any origin. */
export function parentLink(): HostLink {
  return {
    post: (message) => window.parent.postMessage(message, "*"),
    listen: (receive) => {
      const onMessage = (event: MessageEvent) => {
        if (event.source === window.parent) receive(event.data);
      };
      window.addEventListener("message", onMessage);
      return () => window.removeEventListener("message", onMessage);
    },
  };
}

/** What a route shows: the run list, one run, New run (blank, or filled in from the run in
 * `?from=`), the provider checks, or Tools. Any other route shows the list. */
export type Place =
  | { kind: "runs" }
  | { kind: "run"; runId: string }
  | { kind: "new"; from: string | null }
  | { kind: "providers" }
  | { kind: "tools" };

export function placeOf(route: string): Place {
  if (route === "/providers") return { kind: "providers" };
  if (route === "/tools") return { kind: "tools" };
  const fresh = /^\/new(?:\?from=([^&]+))?$/.exec(route);
  if (fresh) return { kind: "new", from: fresh[1] ? decodeURIComponent(fresh[1]) : null };
  const match = /^\/runs\/([^/]+)$/.exec(route);
  return match ? { kind: "run", runId: decodeURIComponent(match[1]) } : { kind: "runs" };
}

/** What New run reads before it shows its form. */
export interface NewRunData {
  catalog: Catalog;
  sessions: SessionList;
  /** The run Edit and rerun started from. */
  from: RunDetail | null;
}

export interface RunSetup {
  selection: Selection;
  config: RunConfig;
  auto_run: boolean;
}

export interface PageState {
  route: string;
  runs: RunList | null;
  /** The open run, or null while it is read. */
  run: RunDetail | null;
  providers: ProviderCheck | null;
  newRun: NewRunData | null;
  tools: ToolRuns | null;
  /** Why the last read failed, or null. */
  loadError: string | null;
  /** Whether check-providers runs now. */
  checking: boolean;
}

export type PageStore = ReturnType<typeof createPageStore>;

/** How long the action buttons stay disabled after an action's reply, or after an open that arrives
 * in that time: long enough for Prompting Time to open the next item the action moved past. */
export const HOLD_MS = 600;

const message = (error: unknown) => (error instanceof Error ? error.message : String(error));

/**
 * The page's state, read from Monopole through the host with page protocol 2: each read calls one
 * of the manifest's actions and returns a promise that settles with what it printed, or rejects with
 * the host's error. The page shows the route Prompting Time opens, starting at the run list, and
 * reads again when files in Monopole's state folder change.
 */
export function createPageStore(link: HostLink) {
  const [state, setState] = createStore<PageState>({
    route: "/",
    runs: null,
    run: null,
    providers: null,
    newRun: null,
    tools: null,
    loadError: null,
    checking: false,
  });
  const waiting = new Map<number, { resolve: (json: unknown) => void; reject: (error: Error) => void }>();
  let lastId = 0;
  // The open route, kept beside the store, which applies a change only at its next flush, so reads
  // that finish after the user moved on are dropped.
  let route = "/";

  function call(action: string, args: string[] = []): Promise<unknown> {
    const id = ++lastId;
    return new Promise((resolve, reject) => {
      waiting.set(id, { resolve, reject });
      link.post({ version: PROTOCOL_VERSION, type: "call", id, action, args });
    });
  }

  /** Reads what the route shows; a read that lands after the route changed is dropped. */
  async function read() {
    const asked = route;
    const place = placeOf(asked);
    try {
      if (place.kind === "runs") {
        const runs = (await call("runs")) as RunList;
        if (route === asked)
          setState((s) => {
            s.runs = runs;
            s.loadError = null;
          });
      } else if (place.kind === "run") {
        const shown = ((await call("run", ["--item", place.runId])) as { run: RunDetail }).run;
        if (route === asked)
          setState((s) => {
            s.run = shown;
            s.loadError = null;
          });
      } else if (place.kind === "new") {
        const [catalog, sessions, from] = await Promise.all([
          call("catalog") as Promise<Catalog>,
          call("sessions", ["--limit", "50"]) as Promise<SessionList>,
          place.from ? (call("run", ["--item", place.from]) as Promise<{ run: RunDetail }>).then((r) => r.run) : null,
        ]);
        if (route === asked)
          setState((s) => {
            s.newRun = { catalog, sessions, from };
            s.loadError = null;
          });
      } else if (place.kind === "tools") {
        const tools = (await call("tools")) as ToolRuns;
        if (route === asked)
          setState((s) => {
            s.tools = tools;
            s.loadError = null;
          });
      } else {
        const providers = (await call("providers")) as ProviderCheck;
        if (route === asked)
          setState((s) => {
            s.providers = providers;
            s.loadError = null;
          });
      }
    } catch (error) {
      if (route === asked)
        setState((s) => {
          s.loadError = message(error);
        });
    }
  }

  // While an action the user pressed runs, and for HOLD_MS after its reply, the page holds every
  // action button disabled. Prompting Time may open the next item in that time, and an open in that
  // time holds them HOLD_MS more, so a second click can't act on the item it opened. The plain
  // variables decide; the signal only draws, since the store applies a write at its next flush.
  // A sign-in waits up to 10 minutes for the browser and a tool can run for minutes, so those hold
  // the buttons only after their reply (`holdingAfter`); while they run, only their own button says
  // so. The sign-in's code and cancel buttons aren't held.
  const [held, setHeld] = createSignal(false);
  let running = 0;
  let release: ReturnType<typeof setTimeout> | undefined;
  function holdFor(ms: number) {
    clearTimeout(release);
    release = setTimeout(() => {
      release = undefined;
      setHeld(false);
    }, ms);
  }
  async function holding<T>(work: () => Promise<T>): Promise<T> {
    running++;
    clearTimeout(release);
    release = undefined;
    setHeld(true);
    try {
      return await work();
    } finally {
      if (--running === 0) holdFor(HOLD_MS);
    }
  }

  async function holdingAfter<T>(work: () => Promise<T>): Promise<T> {
    try {
      return await work();
    } finally {
      // An action that holds while it runs starts the hold itself when it replies.
      if (running === 0) {
        setHeld(true);
        holdFor(HOLD_MS);
      }
    }
  }

  function open(next: string) {
    if (release !== undefined) holdFor(HOLD_MS);
    if (next !== route) {
      route = next;
      setState((s) => {
        s.route = next;
        s.run = null;
        s.newRun = null;
        s.loadError = null;
      });
    }
    void read();
  }

  const stop = link.listen((data) => {
    const host = readHostMessage(data);
    if (!host) return;
    switch (host.type) {
      case "open":
        open(host.route);
        break;
      case "changed":
        // New run reads CES through the server, too slow to repeat each time a run writes
        // progress; it reads once when it opens.
        if (placeOf(route).kind !== "new") void read();
        break;
      case "result":
      case "error": {
        const request = host.reply_to === undefined ? undefined : waiting.get(host.reply_to);
        if (!request) return;
        waiting.delete(host.reply_to!);
        // Monopole writes a refusal's reason on stderr; the page shows it, or the host's message
        // when there is none.
        if (host.type === "error")
          request.reject(new CallError(host.stderr?.trim() || host.message, { stage: host.stage, exit_code: host.exit_code, stderr: host.stderr }));
        else request.resolve(host.json);
        break;
      }
    }
  });

  return {
    state,
    place: () => placeOf(state.route),
    /** Tells the host the page listens, and reads the run list it starts on. */
    ready: () => {
      link.post({ version: PROTOCOL_VERSION, type: "ready" });
      void read();
    },
    open,
    /** Whether the action buttons are disabled: while an action runs, and for HOLD_MS after. */
    held,
    /** Asks Prompting Time to delete the run, which it does only after the user confirms. Resolves
     * when the run is gone; rejects with the host's message, such as a cancelled confirmation. */
    deleteRun: async (runId: string) => {
      await holding(() => call("delete", ["--item", runId]));
      open("/");
    },
    /** Asks Prompting Time to rerun the run with the same settings, which it does only after the
     * user confirms, and opens the new run. */
    rerun: async (runId: string) => {
      const started = (await holding(() => call("rerun", ["--item", runId]))) as { route: string };
      open(started.route);
    },
    /** Asks Prompting Time to start the stage the waiting run waits for, which it does only after
     * the user confirms, since judging and reflection call paid models, and shows the run. */
    continueRun: async (runId: string) => {
      const advanced = (await holding(() => call("continue", ["--item", runId]))) as { route: string };
      open(advanced.route);
    },
    /** Starts New run's settings after Prompting Time's confirmation, and opens the new run. */
    startRun: async (setup: RunSetup) => {
      const started = (await holding(() => call("create-run", [JSON.stringify(setup)]))) as { route: string };
      open(started.route);
    },
    /** Runs the provider's login, after Prompting Time's confirmation, until it finishes; then
     * shows the provider check the action ran after it. `deviceAuth` runs Codex's device-code login. */
    signIn: async (provider: string, deviceAuth = false) => {
      const args = ["--item", provider, ...(deviceAuth ? ["--device-auth"] : [])];
      const done = (await holdingAfter(() => call("sign-in", args))) as { sign_in: ProviderCheck["sign_in"]; providers: ProviderCheck };
      setState((s) => {
        s.providers = { ...done.providers, sign_in: done.sign_in };
      });
      return done.sign_in;
    },
    sendSignInCode: (provider: string, code: string) => call("sign-in-code", ["--item", provider, "--code", code]),
    cancelSignIn: (provider: string) => call("sign-in-cancel", ["--item", provider]),
    /** Hands the key to stage-key, whose process exits at once, so the key is in the process list
     * only briefly; then set-key, given no key, checks and saves the key stage-key held. The page
     * keeps no copy of it. */
    setKey: async (provider: string, key: string) => {
      const done = (await holding(async () => {
        await call("stage-key", ["--item", provider, "--key", key]);
        return call("set-key", ["--item", provider]);
      })) as { saved: string; providers: ProviderCheck };
      setState((s) => {
        s.providers = { ...done.providers, sign_in: s.providers?.sign_in ?? null };
      });
      return done.saved;
    },
    /** Runs one of the Tools actions; the write tools confirm in Prompting Time first. */
    runTool: async (action: string, args: string[]) => {
      const run = (await holdingAfter(() => call(action, args))) as ToolRun & { tool: string };
      setState((s) => {
        s.tools = { format: 1, tools: { ...(s.tools?.tools ?? {}), [action]: run } };
      });
      return run;
    },
    /** Checks every provider now and shows the result. */
    checkProviders: async () => {
      setState((s) => {
        s.checking = true;
      });
      try {
        const providers = (await holding(() => call("check-providers"))) as ProviderCheck;
        setState((s) => {
          s.providers = { ...providers, sign_in: s.providers?.sign_in ?? null };
          s.loadError = null;
        });
      } catch (error) {
        setState((s) => {
          s.loadError = message(error);
        });
      } finally {
        setState((s) => {
          s.checking = false;
        });
      }
    },
    dispose: () => {
      stop();
      clearTimeout(release);
    },
  };
}

import { createSignal, createStore, reconcile } from "solid-js";
import {
  CallError,
  PROTOCOL_VERSION,
  readHostMessage,
  type Analysis,
  type Batch,
  type Catalog,
  type ProviderCheck,
  type RunConfig,
  type RunDetail,
  type RunList,
  type Selection,
  type SessionEstimates,
  type SessionList,
  type Settings,
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

/** What a route shows: the run list, one run, two runs side by side, Trends, New run (blank, or
 * filled in from the run in `?from=`), the provider checks, Tools, or Settings. Any other route
 * shows the list. */
export type Place =
  | { kind: "runs" }
  | { kind: "run"; runId: string }
  | { kind: "compare"; runIds: [string, string] }
  | { kind: "trends" }
  | { kind: "new"; from: string | null }
  | { kind: "providers" }
  | { kind: "tools" }
  | { kind: "settings" };

export function placeOf(route: string): Place {
  if (route === "/providers") return { kind: "providers" };
  if (route === "/tools") return { kind: "tools" };
  if (route === "/settings") return { kind: "settings" };
  if (route === "/trends") return { kind: "trends" };
  const fresh = /^\/new(?:\?from=([^&]+))?$/.exec(route);
  if (fresh) return { kind: "new", from: fresh[1] ? decodeURIComponent(fresh[1]) : null };
  const pair = /^\/compare\/([^/]+)\/([^/]+)$/.exec(route);
  if (pair) return { kind: "compare", runIds: [decodeURIComponent(pair[1]), decodeURIComponent(pair[2])] };
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
  /** The newest runs, which the list and the tab sidebar's count and status line read. */
  runs: RunList | null;
  /** The open run, or null while it is read. */
  run: RunDetail | null;
  /** The two runs Compare shows, or null while they are read. */
  compare: [RunDetail, RunDetail] | null;
  providers: ProviderCheck | null;
  newRun: NewRunData | null;
  tools: ToolRuns | null;
  /** The plugin's settings as get-settings prints them, the values Prompting Time's own Settings
   * tab shows. */
  settings: Settings | null;
  /** The web UI's analysis, kept across tabs since it takes up to minutes to read, and when it was read. */
  trends: Analysis | null;
  trendsReadAt: string | null;
  /** Whether trends reads now. */
  readingTrends: boolean;
  /** Why the last read failed, or null. */
  loadError: string | null;
  /** Whether check-providers runs now. */
  checking: boolean;
  /** The run list's search text and status-group filter, kept while the user opens a run. */
  runQuery: string;
  runGroup: RunGroup | null;
  /** The run j and k selected in the list, kept while the user opens it. */
  runSelected: string | null;
  /** The runs checked in the list for an action on all of them; a new search or filter clears it. */
  runChecked: string[];
}

/** The actions the run list runs on every checked run. */
export type BatchAction = "delete" | "rerun" | "continue";

/** How a batch action went: the runs it acted on, and the runs the server refused, with why. */
export interface BatchOutcome {
  done: string[];
  failed: { item: string; error: string }[];
}

export type RunGroup = NonNullable<RunList["runs"][number]["group"]>;

export type PageStore = ReturnType<typeof createPageStore>;

/** The run in what `run --item` printed. Prompting Time hands the page any output that isn't whole
 * JSON as text, such as output past the first MiB, which it doesn't keep. */
export function runIn(printed: unknown, runId: string): RunDetail {
  const run = typeof printed === "object" && printed !== null ? (printed as { run?: unknown }).run : undefined;
  if (typeof run === "object" && run !== null && (run as { run_id?: unknown }).run_id === runId) return run as RunDetail;
  throw new Error(`Monopole printed ${runId} in a form the page can't read; open it in Monopole's web UI.`);
}

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
    settings: null,
    trends: null,
    trendsReadAt: null,
    readingTrends: false,
    loadError: null,
    checking: false,
    runQuery: "",
    runGroup: null,
    runSelected: null,
    runChecked: [],
    compare: null,
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

  const failed = (asked: string, error: unknown) => {
    if (route === asked)
      setState((s) => {
        s.loadError = message(error);
      });
  };

  /** Reads the runs and the last provider check, which the tab sidebar counts on every tab and
   * the run list and Providers show. Both read files only. */
  function readSidebar() {
    const asked = route;
    const reads = [
      (call("runs") as Promise<RunList>).then(
        (runs) =>
          setState((s) => {
            // Reconciled, so cards that didn't change keep their elements and an open ⋯ menu
            // or popover stays open when the runs are read again.
            if (s.runs) reconcile(runs, "run_id")(s.runs);
            else s.runs = runs;
            if (placeOf(route).kind === "runs") s.loadError = null;
          }),
        (error) => placeOf(asked).kind === "runs" && failed(asked, error),
      ),
      (call("providers") as Promise<ProviderCheck>).then(
        (providers) =>
          setState((s) => {
            s.providers = providers;
          }),
        (error) => placeOf(asked).kind === "providers" && failed(asked, error),
      ),
    ];
    return Promise.all(reads);
  }

  /** Reads Trends: the server's analysis, which reads CES and can take minutes, so it is read when
   * Trends first opens and on Read again, not on every change. */
  async function readTrends() {
    const asked = route;
    setState((s) => {
      s.readingTrends = true;
    });
    try {
      const trends = (await call("trends")) as Analysis;
      setState((s) => {
        s.trends = trends;
        s.trendsReadAt = new Date().toISOString();
        if (placeOf(route).kind === "trends") s.loadError = null;
      });
    } catch (error) {
      failed(asked, error);
    } finally {
      setState((s) => {
        s.readingTrends = false;
      });
    }
  }

  /** Reads what the route shows besides the sidebar; a read that lands after the route changed is
   * dropped. */
  async function readPlace() {
    const asked = route;
    const place = placeOf(asked);
    try {
      if (place.kind === "run") {
        const shown = runIn(await call("run", ["--item", place.runId]), place.runId);
        if (route === asked)
          setState((s) => {
            // Reconciled into the shown run, so a fold the user opened stays open while the run
            // writes progress and the page reads it again.
            if (s.run?.run_id === shown.run_id) reconcile(shown, "id")(s.run);
            else s.run = shown;
            s.loadError = null;
          });
      } else if (place.kind === "compare") {
        const shown = await Promise.all(place.runIds.map((id) => call("run", ["--item", id]).then((printed) => runIn(printed, id))));
        if (route === asked)
          setState((s) => {
            s.compare = [shown[0], shown[1]];
            s.loadError = null;
          });
      } else if (place.kind === "new") {
        const [catalog, sessions, from] = await Promise.all([
          call("catalog") as Promise<Catalog>,
          call("sessions", ["--limit", "50"]) as Promise<SessionList>,
          place.from ? call("run", ["--item", place.from]).then((printed) => runIn(printed, place.from!)) : null,
        ]);
        if (route === asked)
          setState((s) => {
            s.newRun = { catalog, sessions, from };
            s.loadError = null;
          });
      } else if (place.kind === "settings") {
        const settings = await readSettings();
        if (route === asked)
          setState((s) => {
            s.settings = settings;
            s.loadError = null;
          });
      } else if (place.kind === "tools") {
        const tools = (await call("tools")) as ToolRuns;
        if (route === asked)
          setState((s) => {
            s.tools = tools;
            s.loadError = null;
          });
      }
    } catch (error) {
      failed(asked, error);
    }
  }

  async function readSettings(): Promise<Settings> {
    const settings = await call("get-settings");
    if (typeof settings !== "object" || settings === null || Array.isArray(settings)) throw new Error("Monopole didn't print its settings as JSON.");
    return settings as Settings;
  }

  /** Reads the sidebar and what the route shows. Trends reads only when it has nothing yet. */
  function read() {
    void readSidebar();
    if (placeOf(route).kind === "trends") {
      if (!state.trends && !state.readingTrends) void readTrends();
    } else void readPlace();
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
        s.compare = null;
        s.newRun = null;
        s.loadError = null;
      });
    }
    read();
  }

  const stop = link.listen((data) => {
    const host = readHostMessage(data);
    if (!host) return;
    switch (host.type) {
      case "open":
        open(host.route);
        break;
      case "changed":
        // New run and Trends read CES through the server, too slow to repeat each time a run
        // writes progress; they read when they open and on Read again.
        void readSidebar();
        if (placeOf(route).kind !== "new" && placeOf(route).kind !== "trends") void readPlace();
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
      read();
    },
    open,
    /** Sets the run list's search text and status-group filter, and clears the checked runs, so
     * an action never reaches a checked run the list no longer shows. */
    filterRuns: (query: string, group: RunGroup | null) =>
      setState((s) => {
        s.runQuery = query;
        s.runGroup = group;
        s.runChecked = [];
      }),
    checkRuns: (runIds: string[]) =>
      setState((s) => {
        s.runChecked = runIds;
      }),
    selectRun: (runId: string | null) =>
      setState((s) => {
        s.runSelected = runId;
      }),
    /** Reads the open tab again, including Trends and New run, which changes don't reread. */
    refresh: () => {
      void readSidebar();
      if (placeOf(route).kind === "trends") void readTrends();
      else void readPlace();
    },
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
    /** Asks Prompting Time to run the action on every run in runIds, which it does after one
     * confirmation covering them all, and says which runs it acted on and which the server
     * refused. Rejects as one run's action does, such as for a cancelled confirmation. */
    runBatch: async (action: BatchAction, runIds: string[]): Promise<BatchOutcome> => {
      const printed = await holding(() => call(action, runIds.flatMap((id) => ["--item", id])));
      // One run prints what its single action prints, and delete prints text; a refusal rejected.
      if (runIds.length === 1 || action === "delete") return { done: runIds, failed: [] };
      const outcome: BatchOutcome = { done: [], failed: [] };
      for (const result of (printed as Batch).results) {
        if ("error" in result) outcome.failed.push(result);
        else outcome.done.push(result.item);
      }
      return outcome;
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
    /** The selected sessions' judging token estimates, which New run's capacity hints read. */
    sessionEstimates: (ids: string[]) => call("session-estimates", ids) as Promise<SessionEstimates>,
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
    /** Saves the changed settings with save-settings, as Prompting Time's Settings tab does, and
     * shows what Monopole saved. Rejects with Monopole's reason when it refuses a value. */
    saveSettings: async (changes: Settings) => {
      await call("save-settings", [JSON.stringify(changes)]);
      const settings = await readSettings();
      setState((s) => {
        s.settings = settings;
      });
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

import { createStore } from "solid-js";
import { PROTOCOL_VERSION, readHostMessage, type ProviderCheck, type RunDetail, type RunList } from "./protocol";

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

/** What a route shows: the run list, one run, or the provider checks. Any other route shows the list. */
export type Place = { kind: "runs" } | { kind: "run"; runId: string } | { kind: "providers" };

export function placeOf(route: string): Place {
  if (route === "/providers") return { kind: "providers" };
  const match = /^\/runs\/([^/]+)$/.exec(route);
  return match ? { kind: "run", runId: decodeURIComponent(match[1]) } : { kind: "runs" };
}

export interface PageState {
  route: string;
  runs: RunList | null;
  /** The open run, or null while it is read. */
  run: RunDetail | null;
  providers: ProviderCheck | null;
  /** Why the last read failed, or null. */
  loadError: string | null;
  /** Whether check-providers runs now. */
  checking: boolean;
}

export type PageStore = ReturnType<typeof createPageStore>;

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

  function open(next: string) {
    if (next !== route) {
      route = next;
      setState((s) => {
        s.route = next;
        s.run = null;
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
        void read();
        break;
      case "result":
      case "error": {
        const request = host.reply_to === undefined ? undefined : waiting.get(host.reply_to);
        if (!request) return;
        waiting.delete(host.reply_to!);
        if (host.type === "error") request.reject(new Error(host.message));
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
    /** Asks Prompting Time to delete the run, which it does only after the user confirms. Resolves
     * when the run is gone; rejects with the host's message, such as a cancelled confirmation. */
    deleteRun: async (runId: string) => {
      await call("delete", ["--item", runId]);
      open("/");
    },
    /** Checks every provider now and shows the result. */
    checkProviders: async () => {
      setState((s) => {
        s.checking = true;
      });
      try {
        const providers = (await call("check-providers")) as ProviderCheck;
        setState((s) => {
          s.providers = providers;
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
    dispose: stop,
  };
}

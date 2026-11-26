import * as fixtures from "./fixtures";
import { PROTOCOL_VERSION, readPageMessage, type Card, type HostMessage, type HostUpdate, type PageMessage, type ProviderCheck } from "./protocol";

// A stand-in for Prompting Time, for development (dev/index.html) and the page's tests. It answers
// the page's calls of the manifest's actions from the invented runs in fixtures.ts, as bin/monopole
// would: delete asks `confirm` first, as Prompting Time's confirmation does, and refuses a running
// run; check-providers returns a new check. It refuses an action the manifest does not declare and
// a page that passes --print or --expect itself, as Prompting Time does.

export interface FakeHostOptions {
  /** Prompting Time's confirmation of a delete; without it every delete is cancelled. */
  confirm?: (runId: string) => boolean | Promise<boolean>;
  /** The route the host opens once the page is ready, as when the user opened a row. */
  openOnReady?: string;
}

/** The actions prompting-time-plugin.json declares, and which of them need a confirmation. */
const DECLARED = new Set(["status", "runs", "run", "delete", "providers", "check-providers", "get-settings", "save-settings"]);
const CONFIRM = new Set(["delete"]);

export function createFakeHost(send: (message: HostMessage) => void, options: FakeHostOptions = {}) {
  let runs: Card[] = fixtures.cards();
  let providers: ProviderCheck = { format: 1, checked_at: null, providers: [] };
  /** Every message the page sent that fits the protocol, in order. */
  const received: PageMessage[] = [];
  const post = (update: HostUpdate) => send({ ...update, version: PROTOCOL_VERSION });

  function item(args: string[]): string {
    const at = args.indexOf("--item");
    if (at < 0 || !args[at + 1]) throw new Error("--item <run id> is required");
    return args[at + 1];
  }

  async function run(action: string, args: string[]): Promise<unknown> {
    if (!DECLARED.has(action)) throw new Error(`Monopole declares no ${action} action.`);
    if (CONFIRM.has(action) && args.some((arg) => arg === "--print" || arg === "--expect")) {
      throw new Error(`${action} asks to confirm first, so Prompting Time adds --print and --expect itself.`);
    }
    switch (action) {
      case "runs":
        return { format: 1, runs, truncated: false };
      case "run": {
        const id = item(args);
        const found = runs.find((card) => card.run_id === id);
        if (!found) throw new Error(`There is no run ${id}.`);
        return { format: 1, run: fixtures.detail(found) };
      }
      case "delete": {
        const id = item(args);
        const found = runs.find((card) => card.run_id === id);
        if (!found) throw new Error(`There is no run ${id}.`);
        if (!found.deletable) throw new Error(`${id} is ${found.status}; cancel it in Monopole before deleting it.`);
        if (!(await options.confirm?.(id))) throw new Error("Nothing ran: the confirmation was cancelled.");
        runs = runs.filter((card) => card.run_id !== id);
        queueMicrotask(() => post({ type: "changed", files: ["runs-changed"] }));
        return `Deleted ${id}.`;
      }
      case "providers":
        return providers;
      case "check-providers":
        providers = fixtures.providerCheck();
        return providers;
      default:
        throw new Error(`The fake host does not run ${action}.`);
    }
  }

  return {
    received,
    /** Runs what the page sent, dropping a message that does not fit protocol 2. */
    receive(data: unknown) {
      const message = readPageMessage(data);
      if (!message) return;
      received.push(message);
      if (message.type === "ready") {
        if (options.openOnReady) post({ type: "open", route: options.openOnReady });
        return;
      }
      run(message.action, message.args).then(
        (json) => post({ type: "result", reply_to: message.id, json }),
        (error) => post({ type: "error", reply_to: message.id, message: error instanceof Error ? error.message : String(error) }),
      );
    },
    open: (route: string) => post({ type: "open", route }),
    changed: (files: string[] | null) => post({ type: "changed", files }),
    /** The calls the page made, as action and arguments. */
    calls: () => received.flatMap((message) => (message.type === "call" ? [[message.action, ...message.args]] : [])),
  };
}

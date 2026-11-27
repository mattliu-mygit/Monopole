import * as fixtures from "./fixtures";
import {
  PROTOCOL_VERSION,
  readPageMessage,
  type Card,
  type HostMessage,
  type HostUpdate,
  type PageMessage,
  type ProviderCheck,
  type SignIn,
  type ToolRun,
} from "./protocol";

// A stand-in for Prompting Time, for development (dev/index.html) and the page's tests. It answers
// the page's calls of the manifest's actions from the invented runs in fixtures.ts, as bin/monopole
// would: an action marked confirm asks `confirm` first, as Prompting Time's confirmation does. It
// refuses an action the manifest does not declare and a page that passes --print or --expect
// itself, as Prompting Time does.

export interface FakeHostOptions {
  /** Prompting Time's confirmation of a confirm action; without it every confirmation is cancelled. */
  confirm?: (action: string, args: string[]) => boolean | Promise<boolean>;
  /** The route the host opens once the page is ready, as when the user opened a row. */
  openOnReady?: string;
  /** Keeps a sign-in waiting until `finishSignIn` or `sign-in-cancel`, as a real browser sign-in does. */
  holdSignIn?: boolean;
}

/** The actions prompting-time-plugin.json declares, and which of them need a confirmation. */
const DECLARED = new Set([
  "status",
  "runs",
  "run",
  "delete",
  "rerun",
  "create-run",
  "catalog",
  "sessions",
  "providers",
  "check-providers",
  "sign-in",
  "sign-in-code",
  "sign-in-cancel",
  "set-key",
  "score",
  "backfill",
  "monitor",
  "insights",
  "tools",
  "get-settings",
  "save-settings",
]);
const CONFIRM = new Set(["delete", "rerun", "create-run", "sign-in", "score", "backfill", "monitor"]);

export function createFakeHost(send: (message: HostMessage) => void, options: FakeHostOptions = {}) {
  let runs: Card[] = fixtures.cards();
  let providers: ProviderCheck = { format: 1, checked_at: null, providers: [] };
  let signIn: SignIn | null = null;
  let finishSignIn: ((state: SignIn["state"]) => void) | null = null;
  const tools: Record<string, ToolRun> = {};
  /** Every message the page sent that fits the protocol, in order. */
  const received: PageMessage[] = [];
  /** Codes the page typed into a waiting sign-in, and the setups New run started. */
  const codes: string[] = [];
  const setups: unknown[] = [];
  const post = (update: HostUpdate) => send({ ...update, version: PROTOCOL_VERSION });
  const changed = (file: string) => queueMicrotask(() => post({ type: "changed", files: [file] }));

  function flag(args: string[], name: string): string {
    const at = args.indexOf(name);
    if (at < 0 || !args[at + 1]) throw new Error(`${name} is required`);
    return args[at + 1];
  }
  const item = (args: string[]) => flag(args, "--item");

  function startRun(): { format: 1; run_id: string; status: string; route: string } {
    const run_id = `run-new${runs.length}`;
    const base = runs[0];
    runs = [{ ...base, run_id, status: "scoring", label: "Scoring", dot: "running", group: "running", proposal: null, deletable: false }, ...runs];
    return { format: 1, run_id, status: "scoring", route: `/runs/${run_id}` };
  }

  function recordTool(tool: string, args: Record<string, unknown>, output: string): ToolRun & { tool: string } {
    const run: ToolRun = { state: "done", args, started_at: "2026-11-26T18:00:00+00:00", finished_at: "2026-11-26T18:01:00+00:00", exit_code: 0, output };
    tools[tool] = run;
    return { tool, ...run };
  }

  async function run(action: string, args: string[]): Promise<unknown> {
    if (!DECLARED.has(action)) throw new Error(`Monopole declares no ${action} action.`);
    if (CONFIRM.has(action)) {
      if (args.some((arg) => arg === "--print" || arg === "--expect")) {
        throw new Error(`${action} asks to confirm first, so Prompting Time adds --print and --expect itself.`);
      }
      if (action === "delete") {
        const found = runs.find((card) => card.run_id === item(args));
        if (!found) throw new Error(`There is no run ${item(args)}.`);
        if (!found.deletable) throw new Error(`${found.run_id} is ${found.status}; cancel it in Monopole before deleting it.`);
      }
      if (!(await options.confirm?.(action, args))) throw new Error("Nothing ran: the confirmation was cancelled.");
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
        runs = runs.filter((card) => card.run_id !== id);
        changed("runs-changed");
        return `Deleted ${id}.`;
      }
      case "rerun":
        return startRun();
      case "create-run":
        setups.push(JSON.parse(args[0]));
        return startRun();
      case "catalog":
        return fixtures.catalog();
      case "sessions":
        return fixtures.sessions();
      case "providers":
        return { ...providers, sign_in: signIn };
      case "check-providers":
        providers = fixtures.providerCheck();
        return providers;
      case "sign-in": {
        const provider = item(args);
        signIn = { provider, state: "waiting", started_at: "2026-11-26T18:00:00+00:00", link: `https://${provider}.example/sign-in`, asks_for_code: true, message: null };
        changed("sign-in.json");
        const state = options.holdSignIn ? await new Promise<SignIn["state"]>((resolve) => (finishSignIn = resolve)) : "signed_in";
        signIn = { ...signIn, state };
        providers = fixtures.providerCheck();
        if (state === "signed_in") providers.providers = providers.providers.map((p) => (p.id === provider ? { ...p, state: "signed_in", detail: null } : p));
        return { format: 1, sign_in: signIn, providers };
      }
      case "sign-in-code":
        if (signIn?.state !== "waiting") throw new Error("No sign-in is waiting for a code.");
        codes.push(flag(args, "--code"));
        return { format: 1, sent: true };
      case "sign-in-cancel":
        finishSignIn?.("cancelled");
        return { format: 1, cancelling: true };
      case "set-key": {
        const provider = item(args);
        if (flag(args, "--key") === "bad-key") throw new Error("https://api.openai.com/v1/models refused the key (HTTP 401); nothing saved.");
        providers = fixtures.providerCheck();
        providers.providers = providers.providers.map((p) => (p.id === provider ? { ...p, state: "signed_in", detail: null } : p));
        return { format: 1, saved: "Saved in Monopole's state folder.", providers };
      }
      case "score":
        return recordTool("score", { hours: Number(flag(args, "--hours")) }, "Scored 3 turns, 1 sessions, 0 subagents (12 scores). Wrote 12.");
      case "backfill":
        return recordTool("backfill", { start: flag(args, "--start"), end: args.includes("--end") ? flag(args, "--end") : null }, "Backfilled 40 scores.");
      case "monitor":
        return recordTool("monitor", {}, "No new significant regressions (0 already alerted, 2 tentative/low-confidence).");
      case "insights":
        return recordTool("insights", { kind: flag(args, "--kind") }, `Table for ${flag(args, "--kind")}`);
      case "tools":
        return { format: 1, tools };
      default:
        throw new Error(`The fake host does not run ${action}.`);
    }
  }

  return {
    received,
    codes,
    setups,
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
    /** Ends a held sign-in as the browser sign-in finishing would. */
    finishSignIn: () => finishSignIn?.("signed_in"),
    /** The calls the page made, as action and arguments. */
    calls: () => received.flatMap((message) => (message.type === "call" ? [[message.action, ...message.args]] : [])),
  };
}

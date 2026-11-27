import * as fixtures from "./fixtures";
import {
  CallError,
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
  "continue",
  "create-run",
  "catalog",
  "sessions",
  "providers",
  "check-providers",
  "sign-in",
  "sign-in-code",
  "sign-in-cancel",
  "stage-key",
  "set-key",
  "score",
  "backfill",
  "monitor",
  "insights",
  "tools",
  "get-settings",
  "save-settings",
]);
const CONFIRM = new Set(["delete", "rerun", "continue", "create-run", "sign-in", "score", "backfill", "monitor"]);

/** A refusal at a confirm action's --print, as Prompting Time reports it: Monopole printed reason
 * on stderr and exited 1, and the confirmation showed it. */
function refusal(reason: string): CallError {
  return new CallError(reason, { stage: "preview", exit_code: 1, stderr: `${reason}\n` });
}

export function createFakeHost(send: (message: HostMessage) => void, options: FakeHostOptions = {}) {
  let runs: Card[] = fixtures.cards();
  let providers: ProviderCheck = { format: 1, checked_at: null, providers: [] };
  let signIn: SignIn | null = null;
  let finishSignIn: ((state: SignIn["state"]) => void) | null = null;
  /** The key stage-key holds for set-key, by provider, as <state>/staged-<provider>-key holds it. */
  const staged = new Map<string, string>();
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
      if (action === "delete" || action === "continue") {
        const found = runs.find((card) => card.run_id === item(args));
        if (!found) throw refusal(`There is no run ${item(args)}.`);
        if (action === "delete" && !found.deletable) throw refusal(`${found.run_id} is ${found.status}; cancel it in Monopole before deleting it.`);
        if (action === "continue" && !found.next_stage) throw refusal(`${found.run_id} is ${found.label}, so it isn't waiting for Continue.`);
      }
      if (!(await options.confirm?.(action, args))) throw new CallError("Nothing ran: the confirmation was cancelled.", { stage: "cancelled" });
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
      case "continue": {
        const id = item(args);
        runs = runs.map((card) =>
          card.run_id === id ? { ...card, status: "judging", label: "Judging", dot: "running", group: "running", progress: null, next_stage: null } : card,
        );
        changed("runs-changed");
        return { format: 1, run_id: id, status: "judging", route: `/runs/${id}` };
      }
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
        signIn = args.includes("--device-auth")
          ? { provider, state: "waiting", started_at: "2026-11-26T18:00:00+00:00", link: "https://auth.openai.com/codex/device", device_code: "BUT6-0M8K4", asks_for_code: false, message: null }
          : { provider, state: "waiting", started_at: "2026-11-26T18:00:00+00:00", link: `https://${provider}.example/sign-in`, device_code: null, asks_for_code: true, message: null };
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
      case "stage-key":
        staged.set(item(args), flag(args, "--key"));
        return { format: 1, staged: item(args) };
      case "set-key": {
        const provider = item(args);
        if (args.includes("--key")) throw new Error("set-key takes no --key; stage-key holds the key.");
        const key = staged.get(provider);
        staged.delete(provider);
        if (!key) throw new Error("No key given; nothing saved.");
        if (key === "bad-key") throw new Error("https://api.openai.com/v1/models refused the key (HTTP 401); nothing saved.");
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
      case "insights": {
        const kind = flag(args, "--kind");
        const run = recordTool("insights", { kind }, `Table for ${kind}`);
        // Judge agreement has a table; the others show their text.
        if (kind === "judges") {
          run.tables = tools.insights.tables = [
            { title: "Judge agreement", columns: ["Judge", "Reference", "Sessions"], rows: [["session_outcome_quality", "outcome.tests_passed", "6"]] },
          ];
        }
        return run;
      }
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
        (error) =>
          post({
            type: "error",
            reply_to: message.id,
            message: error instanceof Error ? error.message : String(error),
            ...(error instanceof CallError ? error.failure : {}),
          }),
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

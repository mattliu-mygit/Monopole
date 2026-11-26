// The messages between Monopole's page and Prompting Time, page protocol 2, which
// prompting-time-plugin.json names as page.protocol. Prompting Time hosts the page in an iframe
// sandboxed to scripts only, so the page reaches nothing else: no network, no Monopole server. Every
// message is a JSON object with `version` 2, `type`, and only the fields its type lists; Prompting
// Time drops a message that does not fit, and so does the page.
//
// Page -> host:
//   ready                      The page listens. The host then sends the route the user opened, if any.
//   call {id, action, args}    Runs one of the manifest's actions with args after its command. The host
//                              answers with result or error, with the call's id as reply_to. An action
//                              marked confirm (delete) runs only after the user confirms it in
//                              Prompting Time, which adds --print and --expect itself.
// Host -> page:
//   result {reply_to, json}    What the action printed: JSON when it printed JSON, else the text.
//   error {reply_to?, message} The action failed or was refused, or the user cancelled the confirmation.
//   open {route}               The user opened an item or blocker of `status`: /runs/<run id> or /providers.
//   changed {files}            Files changed in Monopole's state folder, such as runs-changed, or null.
//
// The actions the page calls (bin/monopole <action>):
//   runs                       the newest runs, as cards
//   run --item <run id>        one run's detail
//   delete --item <run id>     deletes a run after Prompting Time's confirmation
//   providers                  the last provider sign-in check
//   check-providers            checks every provider now and prints the result

export const PROTOCOL_VERSION = 2;

export type Dot = "error" | "attention" | "success" | "running" | "idle";

export interface Judge {
  id: string;
  label: string;
  provider: string;
}

/** How the best proposal compared with the current instructions. Scores are the proposal
 * evaluator's predictions; `check` is the A/B check in the sandbox, null when none ran. */
export interface Proposal {
  current: number | null;
  proposed: number | null;
  change: number | null;
  candidates: number;
  check: { status: string; winner: string | null; reason: string | null } | null;
  recommended: boolean;
  note: string | null;
}

/** One run in `runs`; src/weave_agent_signals/plugin/runs.py `card` writes it. */
export interface Card {
  run_id: string;
  status: "created" | "scoring" | "judging" | "reflecting" | "complete" | "failed" | "cancelled";
  label: string;
  dot: Dot;
  group: "failed" | "review" | "running" | "done" | null;
  created_at: string;
  updated_at: string;
  title: string;
  session_count: number | null;
  turns: number | null;
  judges: Judge[];
  rated: number;
  mean_score: number | null;
  progress: string | null;
  failure: { stage: string; stopped_at: string; cause: string } | null;
  review_state: string;
  proposal: Proposal | null;
  deletable: boolean;
}

export interface RunList {
  format: 1;
  runs: Card[];
  truncated: boolean;
}

/** `run --item <id>`: the card plus what the run page shows. */
export interface RunDetail extends Card {
  selection: { session_ids: string[]; since: string | null; until: string | null; timezone: string | null } | null;
  stages: { stage: string; state: "done" | "running" | "failed" | "not_started"; message: string | null }[];
  rubrics: { rubric: string; label: string; conversation_id: string; rating: number | null; review_status: string }[];
  models: { proposal_writer: string | null; proposal_evaluator: string | null };
  error: string | null;
}

export type ProviderState = "signed_in" | "signed_out" | "no_key" | "not_installed" | "unknown";

export interface ProviderCheck {
  format: 1;
  checked_at: string | null;
  providers: { id: string; title: string; state: ProviderState; detail: string | null }[];
}

export type PageMessage =
  | { version: typeof PROTOCOL_VERSION; type: "ready" }
  | { version: typeof PROTOCOL_VERSION; type: "call"; id: number; action: string; args: string[] };

export type HostUpdate =
  | { type: "result"; reply_to: number; json: unknown }
  | { type: "error"; reply_to?: number; message: string }
  | { type: "open"; route: string }
  | { type: "changed"; files: string[] | null };

export type HostMessage = HostUpdate & { version: typeof PROTOCOL_VERSION };

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** Whether message has exactly the keys required, plus any of optional. */
function hasKeys(message: Record<string, unknown>, required: string[], optional: string[] = []): boolean {
  const keys = Object.keys(message);
  return required.every((key) => keys.includes(key)) && keys.every((key) => required.includes(key) || optional.includes(key));
}

/** The host message in data, or null when it is not one: another version, an unknown type, or a
 * field missing, extra, or of the wrong kind. */
export function readHostMessage(data: unknown): HostMessage | null {
  if (!isRecord(data) || data.version !== PROTOCOL_VERSION) return null;
  switch (data.type) {
    case "result":
      return hasKeys(data, ["version", "type", "reply_to", "json"]) && Number.isSafeInteger(data.reply_to) ? (data as HostMessage) : null;
    case "error":
      return hasKeys(data, ["version", "type", "message"], ["reply_to"]) &&
        typeof data.message === "string" &&
        (data.reply_to === undefined || Number.isSafeInteger(data.reply_to))
        ? (data as HostMessage)
        : null;
    case "open":
      return hasKeys(data, ["version", "type", "route"]) && typeof data.route === "string" && data.route.startsWith("/")
        ? (data as HostMessage)
        : null;
    case "changed":
      return hasKeys(data, ["version", "type", "files"]) &&
        (data.files === null || (Array.isArray(data.files) && data.files.every((file) => typeof file === "string")))
        ? (data as HostMessage)
        : null;
    default:
      return null;
  }
}

/** The page message in data, or null: the check Prompting Time runs before it runs anything, which
 * the fake host runs too. */
export function readPageMessage(data: unknown): PageMessage | null {
  if (!isRecord(data) || data.version !== PROTOCOL_VERSION) return null;
  if (data.type === "ready") return hasKeys(data, ["version", "type"]) ? (data as PageMessage) : null;
  if (data.type !== "call" || !hasKeys(data, ["version", "type", "id", "action", "args"])) return null;
  const fits =
    Number.isSafeInteger(data.id) &&
    typeof data.action === "string" &&
    data.action !== "" &&
    Array.isArray(data.args) &&
    data.args.every((arg) => typeof arg === "string");
  return fits ? (data as PageMessage) : null;
}

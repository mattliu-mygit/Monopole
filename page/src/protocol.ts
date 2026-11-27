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
//   error {reply_to?, message, stage?, exit_code?, stderr?}
//                              The action failed or was refused, or the user cancelled the confirmation.
//                              stage says how far the call got: cancelled, preview (the --print of a
//                              confirm action refused), or run; absent when Prompting Time refused the
//                              call itself. When Monopole's command refused, exit_code (null on a
//                              signal) and stderr, where it wrote its reason, say how.
//   open {route}               The user opened an item or blocker of `status`: /runs/<run id> or /providers.
//   changed {files}            Files changed in Monopole's state folder, such as runs-changed, or null.
//
// The actions the page calls (bin/monopole <action>); those marked * confirm first:
//   runs                       the newest runs, as cards
//   run --item <run id>        one run's detail, with the settings Edit and rerun fills in
//   delete* --item <run id>    deletes a run
//   rerun* --item <run id>     starts a new run with that run's sessions and settings
//   continue* --item <run id>  starts the stage a waiting run waits for (Card.next_stage)
//   create-run* <setup JSON>   starts a new run from New run's {selection, config, auto_run}
//   catalog                    the models and rubrics a new run can choose
//   sessions [--limit n]       the newest sessions a new run can choose
//   providers                  the last provider sign-in check, and the last or waiting sign-in
//   check-providers            checks every provider now and prints the result
//   sign-in* --item <cli>      runs claude, codex, or devin's login and waits for it to finish
//   sign-in-code --item <cli> --code <code>   types a pasted code into the waiting sign-in
//   sign-in-cancel --item <cli>               stops the waiting sign-in
//   stage-key --item wandb|openai --key <key> holds a key for set-key, and exits at once
//   set-key --item wandb|openai               checks the held key with GET /v1/models, then saves it
//   score* --hours 24|168, backfill* --start <day> --end <day>, monitor*, insights --kind <kind>
//   tools                      each tool's last run

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
  group: "failed" | "review" | "waiting" | "running" | "done" | null;
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
  /** The stage Continue starts for a run that waits between stages, or null when it doesn't wait. */
  next_stage: "scoring" | "judging" | "reflecting" | null;
}

export interface RunList {
  format: 1;
  runs: Card[];
  truncated: boolean;
}

/** `run --item <id>`: the card plus what the run page shows. */
export interface Selection {
  session_ids: string[];
  since: string | null;
  until: string | null;
  timezone: string | null;
}

/** What a run asked for: src/weave_agent_signals/run_config.py RunConfig. */
export interface RunConfig {
  model_catalog_version: string;
  rubric_catalog_version: string;
  judge_models: string[];
  challenge_judge_models: string[];
  proposal_model: string;
  proposal_evaluator_model: string;
  rubrics: string[];
  candidate_budget: number;
  force: boolean;
}

export interface RunDetail extends Card {
  selection: Selection | null;
  config: RunConfig | null;
  auto_run: boolean;
  stages: { stage: string; state: "done" | "running" | "failed" | "not_started"; message: string | null }[];
  rubrics: { rubric: string; label: string; conversation_id: string; rating: number | null; review_status: string }[];
  models: { proposal_writer: string | null; proposal_evaluator: string | null };
  error: string | null;
}

export type ProviderState = "signed_in" | "signed_out" | "no_key" | "not_installed" | "unknown";

/** A sign-in the sign-in action runs or ran; `link` is the CLI's fallback sign-in link. */
export interface SignIn {
  provider: string;
  state: "waiting" | "signed_in" | "failed" | "timed_out" | "cancelled" | "stopped";
  started_at: string;
  link: string | null;
  asks_for_code: boolean;
  message: string | null;
}

export interface ProviderCheck {
  format: 1;
  checked_at: string | null;
  providers: { id: string; title: string; state: ProviderState; detail: string | null }[];
  /** `providers` prints the last sign-in; `check-providers` leaves it out. */
  sign_in?: SignIn | null;
}

export interface CatalogModel {
  id: string;
  label: string;
  provider: string;
  supported_roles: string[];
}

/** `catalog`: the server's model and rubric catalogs, with only the fields the page reads. */
export interface Catalog {
  format: 1;
  models: {
    catalog_version: string;
    available_models: CatalogModel[];
    recommended_proposal_model: string | null;
    recommended_judges: string[];
    recommended_challenge_judges: string[];
    proposal_evaluator_preferences: string[];
  };
  rubrics: { catalog_version: string; rubrics: { id: string; label: string }[] };
}

export interface SessionSummary {
  conversation_id: string;
  turn_count: number;
  last_activity: string | null;
  input_preview: string | null;
}

export interface SessionList {
  sessions: SessionSummary[];
  total: number;
  truncated: boolean;
}

/** One tool's last run, from `tools` or the tool's own action. */
export interface ToolRun {
  state: "running" | "done" | "failed" | "stopped";
  args: Record<string, unknown>;
  started_at: string;
  finished_at: string | null;
  exit_code: number | null;
  output: string | null;
}

export interface ToolRuns {
  format: 1;
  tools: Partial<Record<string, ToolRun>>;
}

export type PageMessage =
  | { version: typeof PROTOCOL_VERSION; type: "ready" }
  | { version: typeof PROTOCOL_VERSION; type: "call"; id: number; action: string; args: string[] };

/** How far a failed call got, and when Monopole's command refused, its exit code and stderr. */
export type ActionFailure = { stage?: "cancelled" | "preview" | "run"; exit_code?: number | null; stderr?: string };

/** A failed call as the page's promises reject with it: Monopole's reason from its stderr when it
 * refused, else the host's message, and what else the host said. */
export class CallError extends Error {
  constructor(
    message: string,
    readonly failure: ActionFailure = {},
  ) {
    super(message);
  }
}

export type HostUpdate =
  | { type: "result"; reply_to: number; json: unknown }
  | ({ type: "error"; reply_to?: number; message: string } & ActionFailure)
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
      return hasKeys(data, ["version", "type", "message"], ["reply_to", "stage", "exit_code", "stderr"]) &&
        typeof data.message === "string" &&
        (data.reply_to === undefined || Number.isSafeInteger(data.reply_to)) &&
        (data.stage === undefined || data.stage === "cancelled" || data.stage === "preview" || data.stage === "run") &&
        (data.exit_code === undefined || data.exit_code === null || Number.isSafeInteger(data.exit_code)) &&
        (data.stderr === undefined || typeof data.stderr === "string")
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

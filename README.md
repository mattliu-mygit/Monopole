# Monopole

Monopole evaluates Claude Code and Codex sessions recorded by
[`weave-agent-adapter`](https://github.com/wandb/weave-agent-adapter). It reads
those sessions from the CoreWeave Evaluations service (CES), writes CES Scores on
the same turns and sessions, detects regressions, and prepares instruction
improvements for human review.

The repository, Python package, and CLI entry point are named
`weave-agent-signals`.

## Quick start: run the local web app

This path installs the contributor toolchain and runs the API and React UI in
development mode.

### 1. Install prerequisites

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Node.js `20.19.x` or `22.12+` and npm
- Apple Silicon macOS, or Linux with KVM, for local paired verification
- The self-contained [Smol CLI](https://smolmachines.com/docs/installation), version
  `1.6.13` or newer
- The CES Dataset that `weave-agent-adapter evaluations setup` created

Clone the repository if you have not already:

```bash
git clone https://github.com/mattliu-mygit/Monopole.git
cd Monopole
```

### 2. Install dependencies

From the repository root:

```bash
uv sync --frozen --all-extras
npm --prefix frontend ci
```

The lockfiles provide the exact Python and frontend dependency versions used by
the project.

Install and verify the local microVM runtime separately:

```bash
curl -sSL https://raw.githubusercontent.com/smol-machines/smol/main/scripts/install.sh | bash
smol --version
```

### 3. Configure W&B access

Set `WANDB_API_KEY`, either in your shell or in a repo-root `.env` file:

```dotenv
WANDB_API_KEY=your-key
```

If you already use the W&B CLI, its `api.wandb.ai` entry in `~/.netrc` also
works. Monopole finds the CES scope and Dataset in the adapter's
`~/.weave-agent-adapter/evaluations.json`; override them with `--ces-scope-ref`,
`--ces-dataset-id`, and `--ces-url`, or the matching `WEAVE_AGENT_SIGNALS_CES_*`
env vars. The global `--entity` and `--project` arguments only choose the
CoreWeave Serverless Inference project for judges; by default it is the
adapter's `project` from the same file.

### 4. Create your local instruction-target registry

The target registry defines the Markdown files Monopole may read and update
during reflection and promotion. Start with the safe, repo-relative example:

```bash
cp targets.example.json targets.local.json
```

`targets.local.json` is ignored by Git. Edit it when you want to evaluate a
different repository or a larger set of instruction files. Do not add secrets
or credentials to it.

### 5. Configure the paired Smol Machines runtime

Every web evaluation verifies its provisional instruction proposal in two local
microVMs. Copy the runtime example and set its workspace, image, and exact image
digest:

```bash
cp smol-runtime.example.json smol-runtime.local.json
```

The image may be an OCI `@sha256:` reference or an absolute local
`.smolmachine` artifact whose bytes match `image_digest`. A local artifact also
supports offline/no-network evaluations. It must contain the same Codex and/or
Claude CLI, repository toolchain, and tools used by the recorded agent. The
service copies the source workspace into each VM without a live host mount.
Each arm's environment holds only the shell basics (`PATH`, `LANG`, `LC_ALL`,
`LC_CTYPE`, `TERM`, `TZ`), the harness API keys (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `WANDB_API_KEY`) when the server has them, the names listed in
`passthrough_environment`, and the values in `environment_overrides`, such as
the guest `HOME` and `PWD`. Other server variables, such as cloud or
source-control tokens, stay on the host. Explicit host config or credential
files may be copied through `runtime_files`. Those files are written only
inside each disposable VM; persisted evidence contains their hashes, not their
contents. Keep the runtime file local and never commit credentials.

Monopole drives the Smol CLI rather than its SDK, moves files through bounded,
retried guest commands rather than the CLI's stdin and copy paths, and detaches
each deleted machine's disk image on macOS. The reasons are in
[docs/upstream-smol-issues.md](docs/upstream-smol-issues.md).

Managed instruction files inside the configured workspace stay under
`/workspace`. Managed files under the host home, such as `~/.codex/AGENTS.md`,
are mirrored to the same relative location under the guest home. Dependency
and cache directories omitted from the frozen source are also omitted from
final artifact capture, so installs do not dominate comparison evidence.

The same runtime is used for coding and research tasks. Enable network only
when the evaluated agent should retain network access; forwarded credentials
are reachable by the sandboxed agent when network is enabled.

### 6. Start the API and UI

In one terminal, from the repository root:

```bash
uv run weave-agent-signals serve \
  --target-registry targets.local.json \
  --sandbox-runtime smol-runtime.local.json
```

In a second terminal:

```bash
npm --prefix frontend run dev
```

Open [http://localhost:5173](http://localhost:5173). Vite proxies `/api`
requests to the API at `http://127.0.0.1:8787`.

For a one-process production-style check:

```bash
npm --prefix frontend run build
uv run weave-agent-signals serve \
  --target-registry targets.local.json \
  --sandbox-runtime smol-runtime.local.json
```

Then open [http://127.0.0.1:8787](http://127.0.0.1:8787).

## Run Monopole as a Prompting Time plugin

[Prompting Time](https://github.com/mattliu-mygit/prompting-time) can run Monopole as a
plugin from this folder. `prompting-time-plugin.json` declares it: Prompting Time starts
`bin/monopole-run`, which runs this server; lists runs that failed, wait for review, or wait for
Continue under Needs you, from `bin/monopole status`; and shows the page in `page/`, through the plugin's
actions:

- **Runs**: each run as a card, with Rerun, and Edit and rerun and Delete in its ⋯ menu. Rerun
  starts a new run with the old run's sessions and settings, so turns added to those sessions
  since join it; a judge that has left the catalog stops it with a message naming the judge. A
  run that waits between stages, because it doesn't continue automatically, says "Waiting to
  judge" or "Waiting to reflect", is listed under Needs you as "Waiting for you", and offers
  Continue on its card, its page, and its row in Prompting Time; Continue starts that stage
  through the server, as the web app's Continue does, and its confirmation refuses at once when
  the server isn't answering.
- **New run** (`/new`, or `/new?from=<run id>` for Edit and rerun): sessions, judges, proposal
  writer and evaluator, and Continue automatically. As in the web app, each model says whether it
  fits the chosen sessions, and hints name a chosen model that doesn't and models that share a
  family. Until you pick a model, a blank New run's models follow the chosen sessions with the
  web app's recommendation: the most diverse panel that fits them. Edit and rerun keeps the old
  run's models.
- **Providers**: the last sign-in check, Sign in for Claude, Codex, and Devin, which runs the
  CLI's own login and waits for the browser, or for Codex, a device-code sign-in that shows a link
  and a one-time code (starting either Codex login removes the current Codex login, even if you
  cancel), and a field for the W&B and OpenAI keys, checked with the free `GET /v1/models` before
  they're saved. The W&B key goes in the `api.wandb.ai` entry of `~/.netrc`; the OpenAI key goes in
  the plugin's state folder, and Monopole's server restarts to read it, unless a run is mid-stage.
  Prompting Time passes an action only arguments, so the page hands the key to a `stage-key`
  action that holds it in a mode-600 file and exits in about 0.2 s; `set-key` then checks and
  saves it, so the key isn't in the process list while the check runs.
- **Tools**: `score`, `backfill`, `monitor` (posting to the alert webhook in the plugin's
  settings), and `analyze`'s subagent, judge-agreement, and segment insights, shown as tables.

Rerun, Continue, New run, Delete, Sign in, Score, Backfill, and Check now run only after Prompting
Time's confirmation. Rerun, Continue, New run, and their lists of sessions and models need
Monopole's server to be running. After you press one of the page's action buttons, the page
disables them until 600 ms after the action's reply, and 600 ms more when Prompting Time opens
another item in that time, so a second click can't act on the item Prompting Time moved on to. A
sign-in or a tool holds them only after its reply; while it runs, only its own button waits. Proposal reviews still happen in the web app at
http://127.0.0.1:8787.

To install the development folder:

1. Install [uv](https://docs.astral.sh/uv/), the plugin's one required program.
2. Build the page. Its `@prompting-time/ui` dependency is the tarball a Prompting Time checkout
   builds with `scripts/pack-ui.sh`, at the path `page/package.json` names:

   ```bash
   cd page && npm ci && npm run build
   ```

3. In Prompting Time, install a plugin from a folder and choose this repository.
4. In the plugin's settings, set the target registry and the sandbox runtime, and the run
   database if yours is not `~/.weave-agent-signals/runs.db`. Blank CES fields use the adapter's
   `~/.weave-agent-adapter/evaluations.json`. Restart the plugin to apply them.

The plugin keeps its settings, the last provider check and sign-in, each tool's last run, the
OpenAI key, and a file it rewrites when runs change in `$XDG_STATE_HOME/monopole` (default
`~/.local/state/monopole`), which Prompting Time watches. Only one server may use a run database,
so stop a `serve` you started yourself before Prompting Time starts Monopole on the same database.

Page checks, from `page/`: `npm test`, `npm run typecheck`, and `npm run build`. `npm run dev` serves
the page with a fake host at http://localhost:5176/dev/index.html.

## Model-backed features

Deterministic scoring and trace inspection need only W&B access. Judging and
reflection additionally need a usable model backend:

- CoreWeave Serverless Inference uses `WANDB_API_KEY` and requires
  inference credits.
- OpenAI inference uses `OPENAI_API_KEY`.
- Local Claude, Codex, Antigravity, or Devin inference requires an installed,
  authenticated `claude`, `codex`, `agy`, or `devin` CLI on `PATH`. Devin judges
  run with their own home, `~/.devin-judge/home`, which links only Devin's
  stored login, so your Devin rules, config, and plugins don't reach them.

One provider-qualified model catalog supplies proposal writers, judges, A/B
verification judges, and proposal evaluators. The web UI only offers local
models whose provider executable it detects. Antigravity contributes its
Gemini/Google and open-source models; duplicate Anthropic choices remain under
the Claude provider.

Each catalog model pins its context capacity and four reading budgets, for one
question, one raw window, one whole-thread request, and one merge; a model may
have no window or whole-thread budget. Judging uses only the window budget
today: it packs session turns into windows of up to that size, and any turn
larger than that still gets a window of its own when it fits the context left
after the model's reserved space and the other windows' digests. The question,
whole-thread, and merge budgets are pinned for reading plans that later versions
add. Once sessions are selected, each model selector disables the models that do
not fit that role for those sessions. A model with no window budget, such as
Llama 3.1 8B, cannot be a session judge.

## Try the CLI

Run a read-only inspection first:

```bash
uv run weave-agent-signals inspect --recent 5
```

Run `uv run weave-agent-signals COMMAND --help` for arguments and defaults.

| Command | Purpose |
|---|---|
| `score` | Score recent turns and sessions deterministically |
| `backfill` | Paginate and score a historical date range |
| `judge` | Run sliding-window session model rubrics |
| `inspect` | Inspect recent sessions and their Scores |
| `analyze` | Summarize scores, cohorts, trends, and coaching |
| `monitor` | Alert on new significant regressions |
| `reflect` | Preview candidate instruction bundles and diffs |
| `serve` | Run the API and serve a built frontend |

Standalone `reflect` only previews output. Use an evaluation run in the web UI
to persist candidates, edit a proposal, promote it, or retain a receipt.
Displayed reflection scores are predicted evaluator scores, not verification
runs. The default proposal loop evaluates up to three candidate bundles, then
runs only the selected provisional B against A. Normal session judging defaults
to three recommended reviewers; the independently configured A/B verification
panel defaults to one for cost.

## Target registries

The starter registry allows updates to this repository's `AGENTS.md` and
`CLAUDE.md` without allowing new files:

```json
{
  "schema_version": "1",
  "targets": [
    {
      "kind": "markdown_root",
      "id": "this-repo",
      "description": "Repository-wide agent instructions. Update an existing file for behavior in its scope; create a focused Markdown file only for narrower directory-specific guidance.",
      "root": ".",
      "files": ["AGENTS.md", "CLAUDE.md"],
      "allow_create": false
    }
  ]
}
```

Paths are resolved relative to the registry file. Locators use
`markdown:<id>/<relative-path.md>`. A Markdown root can only access its explicit
`files`; setting `allow_create` to `true` additionally permits new `.md` files
beneath that root. Paths cannot escape the root, cross symlinks, enter
dependency or cache directories, or target non-Markdown files.
The optional, concise `description` tells the proposal writer what the target
contains and when it should update or create instructions there.

## What Monopole does

- Extracts deterministic test, build, lint, install, Git, and command outcomes.
- Measures completion, correction-free rate, and repeated-work efficiency.
- Judges complete sessions with bounded context windows and evidence-cited
  model reviews.
- Compares compatible evaluation cohorts, identifies trends, and emits
  deduplicated regression alerts.
- Runs a reproducible scoring-to-reflection workflow over pinned CES sessions.
- Keeps instruction proposals human-reviewed and records exact promotion
  outcomes.

Scores are CES Scores on the turn's first Case or the session Trajectory they
describe. Local SQLite state tracks evaluation runs and review evidence.

## Development

Run backend checks from the repository root:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src/ tests/
.venv/bin/ruff format --check src/ tests/
```

Run frontend checks from the repository root:

```bash
npm --prefix frontend test
npm --prefix frontend run lint
npm --prefix frontend run build
```

Additional contributor conventions live in [AGENTS.md](AGENTS.md). Start the
architecture tour in [specs/DESIGN.md](specs/DESIGN.md), then use the component
specifications:

- [CES I/O](specs/01-ces-io.md)
- [Evaluation](specs/02-evaluation.md)
- [Analysis and monitoring](specs/03-analysis-monitoring.md)
- [Evaluation runs](specs/04-evaluation-runs.md)

Implementation is under `src/weave_agent_signals/`. The React SPA lives under
`frontend/src/`; backend and frontend tests live under `tests/` and
`frontend/tests/`.

## Local deployment safety

The API is unauthenticated and intended for a trusted local, single-user
environment. It binds to `127.0.0.1` by default and inherits credentials from
the server process. Passing a non-loopback `--host` explicitly exposes that API
and its configured credentials to the reachable network.

On macOS,
[`deploy/com.weave-agent-signals.monitor.plist`](deploy/com.weave-agent-signals.monitor.plist)
can run `score` followed by `monitor` every hour. Review its paths, environment,
and optional webhook before loading it into `~/Library/LaunchAgents/`.

## Troubleshooting

- **`Configuration error: ...` (exit 2):** a setting is missing or wrong, such as
  the CES scope and Dataset, the W&B API key, or a model ID; the message says
  which.
- **`No W&B API key found`:** set `WANDB_API_KEY` or add an `api.wandb.ai`
  entry to `~/.netrc`.
- **The UI cannot reach the API:** keep `serve` running on port `8787`; the Vite
  development server expects that port.
- **No proposal-writer model is available:** install and authenticate Claude,
  Codex, or Antigravity, then restart the API so the model catalog is rebuilt.
- **Paired verification cannot start:** verify that the Smol runtime uses an
  immutable OCI `@sha256:` reference or digest-matching local `.smolmachine`,
  contains the pinned agent CLI and tools, and that every declared runtime file
  exists. Prepared task repositories additionally require host `git` access to
  their exact commit on GitHub, GitLab, Bitbucket, or Codeberg; bootstrap tasks
  require guest network access when cloning or pulling is part of the agent task.
- **CoreWeave Serverless Inference returns HTTP 402:** inference credits are
  disabled for the configured project; choose another backend or enable credits.

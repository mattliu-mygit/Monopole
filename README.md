# Monopole

Monopole evaluates Claude Code and Codex sessions recorded by
[`weave-agent-adapter`](https://github.com/wandb/weave-agent-adapter). It reads
traces from W&B Weave, writes scores beside the same turns and sessions, detects
regressions, and prepares instruction improvements for human review.

The repository, Python package, and CLI entry point are named
`weave-agent-signals`.

## Quick start: run the local web app

This path installs the contributor toolchain and runs the API and React UI in
development mode.

### 1. Install prerequisites

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Node.js `20.19.x` or `22.12+` and npm
- Access to the W&B entity and project that contain your Weave traces

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

### 3. Configure W&B access

Set `WANDB_API_KEY`, either in your shell or in a repo-root `.env` file:

```dotenv
WANDB_API_KEY=your-key
```

If you already use the W&B CLI, its `api.wandb.ai` entry in `~/.netrc` also
works. The default Weave scope is `weave-team/agent-sessions`; pass global
`--entity` and `--project` arguments to use another scope.

### 4. Create your local instruction-target registry

The target registry defines the Markdown files Monopole may read and update
during reflection and promotion. Start with the safe, repo-relative example:

```bash
cp targets.example.json targets.local.json
```

`targets.local.json` is ignored by Git. Edit it when you want to evaluate a
different repository or a larger set of instruction files. Do not add secrets
or credentials to it.

### 5. Start the API and UI

In one terminal, from the repository root:

```bash
uv run weave-agent-signals serve --target-registry targets.local.json
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
uv run weave-agent-signals serve --target-registry targets.local.json
```

Then open [http://127.0.0.1:8787](http://127.0.0.1:8787).

## Model-backed features

Deterministic scoring and trace inspection need only W&B access. Judging and
reflection additionally need a usable model backend:

- W&B Inference uses `WANDB_API_KEY` and requires inference credits.
- OpenAI inference uses `OPENAI_API_KEY`.
- Local Claude or Codex inference requires an installed, authenticated `claude`
  or `codex` CLI on `PATH`.

Reflection proposal writing currently uses an available local Claude or Codex
CLI. The web UI only offers local models whose executables it detects.

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
| `inspect` | Inspect recent trace/session detail and feedback |
| `analyze` | Summarize scores, cohorts, trends, and coaching |
| `monitor` | Alert on new significant regressions |
| `reflect` | Preview candidate instruction bundles and diffs |
| `serve` | Run the API and serve a built frontend |

Standalone `reflect` only previews output. Use an evaluation run in the web UI
to persist candidates, edit a proposal, promote it, or retain a receipt.
Displayed reflection scores are predicted evaluator scores, not verification
runs.

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

## What Monopole does

- Extracts deterministic test, build, lint, install, Git, and command outcomes.
- Measures completion, correction-free rate, and repeated-work efficiency.
- Judges complete sessions with bounded context windows and evidence-cited
  model reviews.
- Compares compatible evaluation cohorts, identifies trends, and emits
  deduplicated regression alerts.
- Runs a reproducible scoring-to-reflection workflow over pinned Weave traces.
- Keeps instruction proposals human-reviewed and records exact promotion
  outcomes.

Scores are custom Weave feedback attached to the turn or conversation they
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

- [Weave I/O](specs/01-weave-io.md)
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

- **`No W&B API key found`:** set `WANDB_API_KEY` or add an `api.wandb.ai`
  entry to `~/.netrc`.
- **The UI cannot reach the API:** keep `serve` running on port `8787`; the Vite
  development server expects that port.
- **No proposal-writer model is available:** install and authenticate either the
  Claude or Codex CLI, then restart the API so the model catalog is rebuilt.
- **W&B Inference returns HTTP 402:** inference credits are disabled for the
  configured project; choose another backend or enable credits.

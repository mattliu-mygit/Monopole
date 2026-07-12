"""TEMPORARY judge backend: shells out to a local coding-agent CLI.

Route B judging with no metered inference. Uses the installed ``claude`` /
``codex`` CLIs (subscription auth) as one-shot LLM judges, so the judge layer
runs while W&B Inference billing is pending. This is a LOCAL developer-machine
stopgap, NOT the production backend — the real path is an HTTP inference backend
with credits (see inference.py) or Weave signals server-side.

Cross-family by construction (see families.py): a Claude-family agent is judged
by the Codex (GPT) CLI and a GPT/Codex-family agent by the Claude CLI, so a model
never judges its own family. The exact CLI flags live in one place (``_build``)
so they are easy to update as the CLIs evolve.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from typing import Any, Callable

from weave_agent_signals.judges.families import model_family
from weave_agent_signals.judges.inference import JudgeResponse

log = logging.getLogger("weave_agent_signals.judges")

# Full model ids → the alias the `claude` CLI's --model flag expects.
_CLAUDE_MODEL_ALIASES = {
    "claude-sonnet-5": "sonnet",
    "claude-opus-4-8": "opus",
    "claude-haiku-4-5": "haiku",
}


def _extract_json(text: str) -> dict:
    """Pull a judge JSON object out of a CLI/LLM text response.

    Agentic CLIs (esp. ``codex exec``) interleave status/reasoning text with the
    answer, so a naive first-``{``-to-last-``}`` slice can span unrelated braces
    and fail. Instead, scan for the first *valid* JSON object anywhere in the
    output, preferring one that carries a ``score`` key. Returns an empty dict
    when nothing parses — callers treat that as "no score".
    """
    if not text:
        return {}
    # Fenced ```json blocks first — the most explicit signal.
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    # Otherwise parse the first valid object at each '{'; prefer one with a score.
    decoder = json.JSONDecoder()
    best: dict | None = None
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            if "score" in obj:
                return obj
            if best is None:
                best = obj
        idx = text.find("{", idx + 1)
    return best if best is not None else {}


class CliJudgeClient:
    """Judge client that invokes a local coding-agent CLI. TEMPORARY (see module docstring).

    Drop-in for InferenceClient: exposes ``chat_json(model, messages) ->
    (dict, JudgeResponse)`` and is a context manager, so the runner is agnostic
    to whether judging goes over HTTP or a local CLI.
    """

    backend = "cli"

    def __init__(self, timeout: float = 180.0, runner: Callable[..., Any] = subprocess.run):
        self._timeout = timeout
        self._run = runner  # injectable for tests

    def chat_json(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> tuple[dict[str, Any], JudgeResponse]:
        system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
        user = "\n\n".join(m["content"] for m in messages if m.get("role") == "user")
        argv, stdin_text, mode = self._build(model, system, user)

        proc = self._run(
            argv,
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=self._timeout,
        )
        if getattr(proc, "returncode", 0) != 0:
            raise RuntimeError(
                f"{argv[0]} judge exited {proc.returncode}: {(proc.stderr or '')[:200]}"
            )

        text = self._extract_text(proc.stdout or "", mode)
        parsed = _extract_json(text)
        return parsed, JudgeResponse(content=text, model=model, usage={})

    def _build(self, model: str, system: str, user: str) -> tuple[list[str], str, str]:
        """Build (argv, stdin_text, mode) for the given judge model.

        Routes by family: anthropic → `claude`, google → `gemini`, else → `codex`.
        ``mode`` is "claude" (answer is inside a --output-format json envelope) or
        "plain" (codex/gemini print the answer directly). The prompt is passed on
        stdin in every case, to avoid OS argument-length limits on big digests.
        """
        fam = model_family(model)
        if fam == "anthropic":
            alias = _CLAUDE_MODEL_ALIASES.get(model, model)
            # `--tools ""` is the definitive no-tools switch; --system-prompt
            # replaces Claude Code's agent prompt with the pure judge prompt;
            # --strict-mcp-config loads no MCP servers; --no-session-persistence
            # avoids writing session files. Prompt is piped on stdin.
            argv = [
                "claude",
                "-p",
                "--output-format",
                "json",
                "--model",
                alias,
                "--tools",
                "",
                "--strict-mcp-config",
                "--no-session-persistence",
            ]
            if system:
                argv += ["--system-prompt", system]
            return argv, user, "claude"

        if fam == "google":
            # Gemini CLI. Prompt goes on stdin (Gemini reads it in non-TTY/headless
            # mode) rather than as a -p argument, so a large digest can't hit the
            # OS arg-length limit. No system-prompt flag (prepend it) and no
            # "disable tools" flag, but default headless mode cannot auto-run tools
            # unless --yolo is passed, which we never do. Plain-text output is just
            # the answer (--output-format json has version-specific bugs).
            prompt = f"{system}\n\n{user}" if system else user
            argv = ["gemini", "-m", model]
            return argv, prompt, "plain"

        # openai / other → Codex CLI. Codex has no "disable tools" flag;
        # --sandbox read-only is the guarantee it can't write or run anything.
        # `-` makes `codex exec` read the prompt from stdin. Model is left to
        # Codex's configured default to avoid model-id mismatches.
        argv = ["codex", "exec", "--sandbox", "read-only", "-"]
        prompt = f"{system}\n\n{user}" if system else user
        return argv, prompt, "plain"

    def _extract_text(self, stdout: str, mode: str) -> str:
        """Unwrap the assistant text from the CLI's stdout.

        ``claude`` wraps the answer in a ``--output-format json`` envelope under
        ``result``; ``plain`` (codex/gemini) prints the answer verbatim.
        """
        if mode == "claude":
            try:
                env = json.loads(stdout)
            except json.JSONDecodeError:
                return stdout
            if isinstance(env, dict) and "result" in env:
                return str(env["result"])
        return stdout

    def close(self) -> None:
        pass

    def __enter__(self) -> "CliJudgeClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

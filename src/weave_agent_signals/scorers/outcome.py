from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from weave_agent_signals.models import Score, ToolSpan, TurnSpan

# --- Outcome dataclasses ---


@dataclass
class TestRunOutcome:
    __test__ = False
    framework: str
    command: str
    exit_code: int | None
    passed: int | None = None
    failed: int | None = None
    errors: int | None = None
    skipped: int | None = None
    total: int | None = None
    duration_s: float | None = None

    @property
    def success(self) -> bool:
        if self.failed is not None and self.failed > 0:
            return False
        if self.errors is not None and self.errors > 0:
            return False
        if self.exit_code is not None:
            return self.exit_code == 0
        if self.passed is not None and self.passed > 0:
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "command": self.command,
            "exit_code": self.exit_code,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "total": self.total,
            "duration_s": self.duration_s,
        }


@dataclass
class BuildOutcome:
    tool: str
    command: str
    exit_code: int | None
    success: bool
    error_count: int | None = None
    warning_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "command": self.command,
            "exit_code": self.exit_code,
            "success": self.success,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
        }


@dataclass
class LintOutcome:
    tool: str
    command: str
    exit_code: int | None
    clean: bool
    issue_count: int | None = None
    fix_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "command": self.command,
            "exit_code": self.exit_code,
            "clean": self.clean,
            "issue_count": self.issue_count,
        }


@dataclass
class InstallOutcome:
    tool: str
    command: str
    exit_code: int | None
    success: bool
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "command": self.command,
            "exit_code": self.exit_code,
            "success": self.success,
            "error_message": self.error_message,
        }


@dataclass
class GitOutcome:
    operation: str
    command: str
    exit_code: int | None
    success: bool
    files_changed: int | None = None
    insertions: int | None = None
    deletions: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "command": self.command,
            "exit_code": self.exit_code,
            "success": self.success,
            "files_changed": self.files_changed,
            "insertions": self.insertions,
            "deletions": self.deletions,
        }


@dataclass
class CommandOutcome:
    """Generic fallback for bash commands not matched by specific classifiers."""

    command: str
    exit_code: int | None
    success: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "exit_code": self.exit_code,
            "success": self.success,
        }


# --- Command classification ---

_TEST_PATTERNS = [
    re.compile(r"(?:python3?|python)\s+(?:-\w\s+)*-m\s+pytest\b"),
    re.compile(r"\bpytest\b"),
    re.compile(r"\bnpm\s+test\b"),
    re.compile(r"\bnpx\s+(?:jest|vitest)\b"),
    re.compile(r"\bcargo\s+test\b"),
    re.compile(r"\bgo\s+test\b"),
    re.compile(r"\bmake\s+(?:test|check)\b"),
    re.compile(r"\bpython3?\s+(?:-\w\s+)*-m\s+unittest\b"),
    re.compile(r"\brspec\b"),
    re.compile(r"\bbundle\s+exec\s+rspec\b"),
]

_BUILD_PATTERNS = [
    re.compile(r"\bcargo\s+build\b"),
    re.compile(r"\bnpm\s+run\s+build\b"),
    re.compile(r"\btsc\b"),
    re.compile(r"\bgo\s+build\b"),
    re.compile(r"\bmake\b(?!\s+(?:test|check|clean))"),
    re.compile(r"\bg(?:cc|\+\+)\b"),
    re.compile(r"\bpython3?\s+(?:-\w\s+)*-m\s+py_compile\b"),
]

_LINT_PATTERNS = [
    re.compile(r"\bruff\s+(?:check|format)\b"),
    re.compile(r"\beslint\b"),
    re.compile(r"\bflake8\b"),
    re.compile(r"\bpylint\b"),
    re.compile(r"\bblack\s+--check\b"),
    re.compile(r"\bprettier\s+--check\b"),
    re.compile(r"\bclippy\b"),
    re.compile(r"\bgolangci-lint\b"),
    re.compile(r"\bpyright\b"),
    re.compile(r"\bmypy\b"),
]

_INSTALL_PATTERNS = [
    re.compile(r"\bnpm\s+install\b"),
    re.compile(r"\bnpm\s+ci\b"),
    re.compile(r"\byarn\s+(?:install|add)\b"),
    re.compile(r"\bpnpm\s+(?:install|add)\b"),
    re.compile(r"\bpip\s+install\b"),
    re.compile(r"\buv\s+(?:pip\s+install|add|sync)\b"),
    re.compile(r"\bpoetry\s+install\b"),
    re.compile(r"\bcargo\s+add\b"),
    re.compile(r"\bgo\s+(?:get|mod\s+download)\b"),
    re.compile(r"\bbundle\s+install\b"),
]

_GIT_MUTATING_OPS = {
    "commit",
    "push",
    "merge",
    "rebase",
    "cherry-pick",
    "revert",
    "reset",
    "stash",
}

_GIT_PATTERN = re.compile(r"\bgit\s+(\S+)")

_EXPLORATION_COMMANDS = {
    "ls",
    "ll",
    "cat",
    "head",
    "tail",
    "less",
    "more",
    "wc",
    "file",
    "find",
    "grep",
    "rg",
    "ag",
    "ack",
    "sed",
    "awk",
    "echo",
    "printf",
    "true",
    "false",
    "pwd",
    "cd",
    "pushd",
    "popd",
    "which",
    "where",
    "whereis",
    "type",
    "command",
    "env",
    "printenv",
    "set",
    "export",
    "unset",
    "date",
    "whoami",
    "hostname",
    "uname",
    "ps",
    "top",
    "htop",
    "df",
    "du",
    "free",
    "tree",
    "stat",
    "readlink",
    "realpath",
    "basename",
    "dirname",
    "diff",
    "cmp",
    "md5sum",
    "sha256sum",
    "shasum",
    "sort",
    "uniq",
    "cut",
    "tr",
    "tee",
    "xargs",
    "touch",
    "mkdir",
    "cp",
    "mv",
    "ln",
    "sleep",
    "wait",
    "man",
    "help",
    "info",
}


def _mutating_git_op(cmd: str) -> str | None:
    # a single Bash call often chains git verbs (`git add -A && git commit`);
    # return the first mutating one rather than only inspecting the leading verb
    for m in _GIT_PATTERN.finditer(cmd):
        if m.group(1) in _GIT_MUTATING_OPS:
            return m.group(1)
    return None


def _strip_prefixes(cmd: str) -> str:
    cmd = cmd.strip()
    # strip env var assignments
    while re.match(r"^[A-Z_][A-Z_0-9]*=\S+\s+", cmd):
        cmd = re.sub(r"^[A-Z_][A-Z_0-9]*=\S+\s+", "", cmd, count=1)
    # strip cd prefix
    cmd = re.sub(r"^cd\s+\S+\s*&&\s*", "", cmd)
    # strip timeout wrapper
    cmd = re.sub(r"^timeout\s+\d+\s+", "", cmd)
    return cmd.strip()


def classify_command(raw_cmd: str) -> str | None:
    # handle pipes: classify the producer (first command)
    cmd = raw_cmd.split("|")[0].strip()
    cmd = _strip_prefixes(cmd)

    for pat in _TEST_PATTERNS:
        if pat.search(cmd):
            return "test"
    for pat in _LINT_PATTERNS:
        if pat.search(cmd):
            return "lint"
    for pat in _BUILD_PATTERNS:
        if pat.search(cmd):
            return "build"
    for pat in _INSTALL_PATTERNS:
        if pat.search(cmd):
            return "install"
    if _mutating_git_op(cmd):
        return "git"
    # read-only git is exploration
    if re.match(r"\bgit\b", cmd):
        return None
    if not cmd:
        return None
    # check if the leading command is pure exploration
    leading = re.match(r"(\S+)", cmd)
    if leading and leading.group(1) in _EXPLORATION_COMMANDS:
        return None
    return "command"


# --- Framework detection ---


_TEST_FRAMEWORK_MATCHERS: tuple[tuple[str | re.Pattern[str], str], ...] = (
    ("pytest", "pytest"),
    ("jest", "jest"),
    ("vitest", "jest"),
    ("cargo test", "cargo"),
    ("go test", "go"),
    ("rspec", "rspec"),
    ("unittest", "unittest"),
    ("npm test", "jest"),
)

_BUILD_TOOL_MATCHERS: tuple[tuple[str | re.Pattern[str], str], ...] = (
    ("tsc", "tsc"),
    ("cargo build", "cargo"),
    ("go build", "go"),
    ("npm run build", "npm"),
    (re.compile(r"\bg(?:cc|\+\+)\b"), "gcc"),
    ("make", "make"),
)

_INSTALL_TOOL_MATCHERS: tuple[tuple[str | re.Pattern[str], str], ...] = (
    ("uv ", "uv"),
    ("pip install", "pip"),
    ("poetry", "poetry"),
    ("pnpm", "pnpm"),
    ("npm", "npm"),
    ("yarn", "yarn"),
    ("cargo add", "cargo"),
    ("go get", "go"),
    ("go mod", "go"),
    ("bundle", "bundler"),
)

_LINT_TOOL_MATCHERS: tuple[tuple[str | re.Pattern[str], str], ...] = (
    ("ruff", "ruff"),
    ("eslint", "eslint"),
    ("black", "black"),
    ("prettier", "prettier"),
    ("flake8", "flake8"),
    ("pylint", "pylint"),
    ("clippy", "clippy"),
    ("golangci-lint", "golangci-lint"),
    ("pyright", "pyright"),
    ("mypy", "mypy"),
)


def _first_match(
    command: str,
    ordered_matchers: tuple[tuple[str | re.Pattern[str], str], ...],
) -> str:
    command_lower = command.lower()
    for matcher, label in ordered_matchers:
        if isinstance(matcher, str):
            if matcher in command_lower:
                return label
        elif matcher.search(command):
            return label
    return "unknown"


def _detect_test_framework(cmd: str) -> str:
    return _first_match(cmd, _TEST_FRAMEWORK_MATCHERS)


def _detect_build_tool(cmd: str) -> str:
    return _first_match(cmd, _BUILD_TOOL_MATCHERS)


def _detect_install_tool(cmd: str) -> str:
    return _first_match(cmd, _INSTALL_TOOL_MATCHERS)


def _detect_lint_tool(cmd: str) -> str:
    return _first_match(cmd, _LINT_TOOL_MATCHERS)


# --- Output parsing ---


def parse_test_output(output: str, framework: str) -> TestRunOutcome:
    result = TestRunOutcome(framework=framework, command="", exit_code=None)

    if framework == "pytest":
        _parse_pytest(output, result)
    elif framework in ("jest", "vitest"):
        _parse_jest(output, result)
    elif framework == "cargo":
        _parse_cargo(output, result)
    elif framework == "go":
        _parse_go(output, result)

    return result


def _parse_pytest(output: str, r: TestRunOutcome):
    m_passed = re.search(r"(\d+)\s+passed", output)
    m_failed = re.search(r"(\d+)\s+failed", output)
    m_error = re.search(r"(\d+)\s+error", output)
    m_skipped = re.search(r"(\d+)\s+skipped", output)
    m_dur = re.search(r"in\s+([\d.]+)s", output)

    if not any([m_passed, m_failed, m_error, m_skipped]):
        return

    r.passed = int(m_passed.group(1)) if m_passed else 0
    r.failed = int(m_failed.group(1)) if m_failed else 0
    r.errors = int(m_error.group(1)) if m_error else 0
    r.skipped = int(m_skipped.group(1)) if m_skipped else 0
    r.duration_s = float(m_dur.group(1)) if m_dur else None
    r.total = (r.passed or 0) + (r.failed or 0) + (r.errors or 0) + (r.skipped or 0)


def _parse_jest(output: str, r: TestRunOutcome):
    m_passed = re.search(r"(\d+)\s+passed", output)
    m_failed = re.search(r"(\d+)\s+failed", output)
    m_total = re.search(r"(\d+)\s+total", output)

    r.passed = int(m_passed.group(1)) if m_passed else 0
    r.failed = int(m_failed.group(1)) if m_failed else 0
    r.total = int(m_total.group(1)) if m_total else None


def _parse_cargo(output: str, r: TestRunOutcome):
    m = re.search(r"test result:\s*(ok|FAILED)\.\s*(\d+)\s+passed;\s*(\d+)\s+failed", output)
    if m:
        r.passed = int(m.group(2))
        r.failed = int(m.group(3))
        r.total = r.passed + r.failed


def _parse_go(output: str, r: TestRunOutcome):
    ok_count = len(re.findall(r"^ok\s+", output, re.MULTILINE))
    fail_count = len(re.findall(r"^FAIL\s+", output, re.MULTILINE))
    r.passed = ok_count
    r.failed = fail_count
    r.total = ok_count + fail_count


# --- Build/lint output parsing ---


def parse_build_output(output: str, tool: str, exit_code: int | None = None) -> BuildOutcome:
    error_count = None
    m = re.search(r"Found\s+(\d+)\s+error", output)
    if m:
        error_count = int(m.group(1))

    warning_count = None
    m = re.search(r"(\d+)\s+warning", output)
    if m:
        warning_count = int(m.group(1))

    success = (
        (exit_code == 0) if exit_code is not None else (error_count is None or error_count == 0)
    )

    return BuildOutcome(
        tool=tool,
        command="",
        exit_code=exit_code,
        success=success,
        error_count=error_count,
        warning_count=warning_count,
    )


def parse_lint_output(output: str, tool: str, exit_code: int | None = None) -> LintOutcome:
    issue_count = None
    m = re.search(r"Found\s+(\d+)\s+error", output)
    if m:
        issue_count = int(m.group(1))

    clean = (exit_code == 0) if exit_code is not None else (issue_count is None or issue_count == 0)

    return LintOutcome(
        tool=tool,
        command="",
        exit_code=exit_code,
        clean=clean,
        issue_count=issue_count,
    )


_INSTALL_SUCCESS_PATTERNS = [
    re.compile(r"Successfully installed"),
    re.compile(r"added \d+ packages?"),
    re.compile(r"changed \d+ packages?"),
    re.compile(r"Installed \d+ packages?"),
    re.compile(r"Resolved \d+ packages?"),
    re.compile(r"audited \d+ packages?", re.IGNORECASE),
    re.compile(r"up to date", re.IGNORECASE),
    re.compile(r"Requirement already satisfied"),
    re.compile(r"packages? in \d+"),
]

# Hard failures — distinct from the diagnostic error_patterns below, which also
# match mere warnings that should not fail the install.
_INSTALL_FAILURE_PATTERNS = [
    re.compile(r"npm ERR!"),
    re.compile(r"ERESOLVE"),
    re.compile(r"ResolutionImpossible"),
    re.compile(r"Could not (?:find|resolve|install)\b"),
    re.compile(r"No matching (?:version|distribution)\b"),
    re.compile(r"^ERROR:", re.MULTILINE),
    re.compile(r"^error:", re.MULTILINE),
]


def parse_install_output(output: str, tool: str, exit_code: int | None = None) -> InstallOutcome:
    if exit_code is not None:
        success = exit_code == 0
    elif any(p.search(output) for p in _INSTALL_FAILURE_PATTERNS):
        success = False
    else:
        success = any(p.search(output) for p in _INSTALL_SUCCESS_PATTERNS)
    error_message = None

    error_patterns = [
        re.compile(r"(?:ERROR|error):?\s*(.+)", re.IGNORECASE),
        re.compile(r"(?:WARN|warning):?\s*(.+)", re.IGNORECASE),
        re.compile(r"Could not (?:find|resolve|install)\b.+"),
        re.compile(r"No matching (?:version|distribution)\b.+"),
        re.compile(r"ERESOLVE\b.+"),
        re.compile(r"ResolutionImpossible\b"),
    ]
    for pat in error_patterns:
        m = pat.search(output)
        if m:
            error_message = m.group(0)[:200]
            break

    return InstallOutcome(
        tool=tool,
        command="",
        exit_code=exit_code,
        success=success,
        error_message=error_message,
    )


_GIT_SUCCESS_PATTERNS = [
    re.compile(r"\[[\w/.-]+\s+[0-9a-f]+\]"),  # [main abc1234] commit message
    re.compile(r"\d+\s+files?\s+changed"),  # N files changed
    re.compile(r"Already up to date"),
    re.compile(r"Fast-forward"),
    re.compile(r"Everything up-to-date"),
    re.compile(r"branch .+ set up to track"),
    re.compile(r"Switched to"),
    re.compile(r"To [\w.:/]+"),  # To github.com:... (push)
    re.compile(r"create mode|delete mode"),
]

# Failure markers take precedence over success patterns: a rejected push still
# prints "To github.com:..." before the rejection, so pattern order alone is not
# enough to tell success from failure.
_GIT_FAILURE_PATTERNS = [
    re.compile(r"! \[rejected\]"),
    re.compile(r"\[remote rejected\]"),
    re.compile(r"failed to push"),
    re.compile(r"CONFLICT"),
    re.compile(r"Merge conflict"),
    re.compile(r"^error:", re.MULTILINE),
    re.compile(r"^fatal:", re.MULTILINE),
    re.compile(r"Automatic merge failed"),
]


def parse_git_output(output: str, operation: str, exit_code: int | None = None) -> GitOutcome:
    if exit_code is not None:
        success = exit_code == 0
    elif any(p.search(output) for p in _GIT_FAILURE_PATTERNS):
        success = False
    else:
        success = any(p.search(output) for p in _GIT_SUCCESS_PATTERNS)

    files_changed = None
    insertions = None
    deletions = None

    m = re.search(r"(\d+)\s+files?\s+changed", output)
    if m:
        files_changed = int(m.group(1))
    m = re.search(r"(\d+)\s+insertions?\(\+\)", output)
    if m:
        insertions = int(m.group(1))
    m = re.search(r"(\d+)\s+deletions?\(-\)", output)
    if m:
        deletions = int(m.group(1))

    return GitOutcome(
        operation=operation,
        command="",
        exit_code=exit_code,
        success=success,
        files_changed=files_changed,
        insertions=insertions,
        deletions=deletions,
    )


# --- Extraction from tool spans ---


def _extract_command(span: ToolSpan) -> str | None:
    try:
        args = json.loads(span.arguments)
        if not isinstance(args, dict):
            return None
        return args.get("command")
    except (json.JSONDecodeError, TypeError):
        return None


def _extract_exit_code(span: ToolSpan) -> int | None:
    try:
        result = json.loads(span.result)
    except (json.JSONDecodeError, TypeError):
        result = None
    # An explicit exit_code field is authoritative — including an explicit null,
    # which means "unknown" (let the parser infer from output) and must NOT fall
    # through to the coarser span status_code (which reports the tool call, not
    # the command). Only a missing field falls back to status_code.
    if isinstance(result, dict) and "exit_code" in result:
        ec = result["exit_code"]
        if ec is None:
            return None
        try:
            return int(ec)
        except (TypeError, ValueError):
            return None
    if span.status_code == "OK":
        return 0
    if span.status_code == "ERROR":
        return 1
    return None


def _extract_output_text(span: ToolSpan) -> str:
    """Extract human-readable output from a tool result.

    The result may be plain text or JSON with stdout/stderr fields.
    """
    raw = span.result or ""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if isinstance(parsed, dict):
        # A tool result with stdout/stderr keys is unwrapped even when both are
        # empty — returning the raw JSON blob would feed "{...}" to the parsers.
        if "stdout" in parsed or "stderr" in parsed:
            parts = [str(parsed.get(k) or "") for k in ("stdout", "stderr")]
            return "\n".join(p for p in parts if p)
        if "output" in parsed:
            return str(parsed.get("output") or "")
    return raw


def extract_outcomes(tool_calls: list[ToolSpan]) -> list:
    outcomes = []
    for tc in tool_calls:
        if tc.tool_name != "Bash":
            continue
        cmd = _extract_command(tc)
        if not cmd:
            continue

        kind = classify_command(cmd)
        if kind is None:
            continue

        exit_code = _extract_exit_code(tc)
        output = _extract_output_text(tc)

        if kind == "test":
            framework = _detect_test_framework(cmd)
            result = parse_test_output(output, framework)
            result.command = cmd
            result.exit_code = exit_code
            outcomes.append(result)
        elif kind == "build":
            tool = _detect_build_tool(cmd)
            result = parse_build_output(output, tool, exit_code)
            result.command = cmd
            outcomes.append(result)
        elif kind == "lint":
            tool = _detect_lint_tool(cmd)
            result = parse_lint_output(output, tool, exit_code)
            result.command = cmd
            outcomes.append(result)
        elif kind == "install":
            tool = _detect_install_tool(cmd)
            result = parse_install_output(output, tool, exit_code)
            result.command = cmd
            outcomes.append(result)
        elif kind == "git":
            operation = _mutating_git_op(cmd) or "unknown"
            result = parse_git_output(output, operation, exit_code)
            result.command = cmd
            outcomes.append(result)
        elif kind == "command":
            success = exit_code == 0 if exit_code is not None else None
            outcomes.append(
                CommandOutcome(
                    command=cmd,
                    exit_code=exit_code,
                    success=success,
                )
            )

    return outcomes


# --- Turn-level scoring ---


def score_turn_outcomes(turn: TurnSpan) -> list[Score]:
    all_tool_calls = turn.tool_calls + [tc for sub in turn.subagents for tc in sub.tool_calls]
    outcomes = extract_outcomes(all_tool_calls)
    if not outcomes:
        return []

    scores = []

    tests = [o for o in outcomes if isinstance(o, TestRunOutcome)]
    if tests:
        all_passed = all(t.success for t in tests)
        total_passed = sum(t.passed or 0 for t in tests)
        total_failed = sum(t.failed or 0 for t in tests)
        frameworks = ", ".join(sorted({t.framework for t in tests}))
        if all_passed:
            reason = f"{frameworks}: {total_passed} passed, {total_failed} failed"
        else:
            reason = f"{frameworks}: {total_failed} failed out of {total_passed + total_failed}"
        scores.append(
            Score(
                scorer="outcome.test",
                value=1.0 if all_passed else 0.0,
                tags=["test_pass"] if all_passed else ["test_failure"],
                confidence=1.0,
                metadata={
                    "outcomes": [t.to_dict() for t in tests],
                    "test_count": total_passed + total_failed,
                    "fail_count": total_failed,
                },
                granularity="turn",
                reason=reason,
            )
        )

    builds = [o for o in outcomes if isinstance(o, BuildOutcome)]
    if builds:
        all_ok = all(b.success for b in builds)
        tools = ", ".join(sorted({b.tool for b in builds}))
        reason = f"{tools}: {'all passed' if all_ok else 'build failure'}"
        scores.append(
            Score(
                scorer="outcome.build",
                value=1.0 if all_ok else 0.0,
                tags=["build_pass"] if all_ok else ["build_failure"],
                confidence=1.0,
                metadata={"outcomes": [b.to_dict() for b in builds]},
                granularity="turn",
                reason=reason,
            )
        )

    lints = [o for o in outcomes if isinstance(o, LintOutcome)]
    if lints:
        all_clean = all(lo.clean for lo in lints)
        tools = ", ".join(sorted({lo.tool for lo in lints}))
        total_issues = sum(lo.issue_count or 0 for lo in lints)
        reason = f"{tools}: {'clean' if all_clean else f'{total_issues} issues'}"
        scores.append(
            Score(
                scorer="outcome.lint",
                value=1.0 if all_clean else 0.0,
                tags=["lint_clean"] if all_clean else ["lint_issues"],
                confidence=1.0,
                metadata={"outcomes": [lo.to_dict() for lo in lints]},
                granularity="turn",
                reason=reason,
            )
        )

    installs = [o for o in outcomes if isinstance(o, InstallOutcome)]
    if installs:
        all_ok = all(i.success for i in installs)
        tools = ", ".join(sorted({i.tool for i in installs}))
        reason = f"{tools}: {'all succeeded' if all_ok else 'install failure'}"
        scores.append(
            Score(
                scorer="outcome.install",
                value=1.0 if all_ok else 0.0,
                tags=["install_success"] if all_ok else ["install_failure"],
                confidence=1.0,
                metadata={"outcomes": [i.to_dict() for i in installs]},
                granularity="turn",
                reason=reason,
            )
        )

    gits = [o for o in outcomes if isinstance(o, GitOutcome)]
    if gits:
        all_ok = all(g.success for g in gits)
        ops = ", ".join(sorted({g.operation for g in gits}))
        reason = f"git {ops}: {'all succeeded' if all_ok else 'failure'}"
        scores.append(
            Score(
                scorer="outcome.git",
                value=1.0 if all_ok else 0.0,
                tags=["git_success"] if all_ok else ["git_failure"],
                confidence=1.0,
                metadata={"outcomes": [g.to_dict() for g in gits]},
                granularity="turn",
                reason=reason,
            )
        )

    cmds = [o for o in outcomes if isinstance(o, CommandOutcome)]
    failed_cmds = [c for c in cmds if c.success is False]
    if failed_cmds:
        reasons = [(c.command.split()[0] if c.command.strip() else "?") for c in failed_cmds]
        reason = f"{len(failed_cmds)} command failure(s): {', '.join(reasons[:5])}"
        scores.append(
            Score(
                scorer="outcome.command",
                value=0.0,
                tags=["command_failure"],
                confidence=0.5,
                metadata={
                    "outcomes": [c.to_dict() for c in failed_cmds],
                    "total_commands": len(cmds),
                    "failed_commands": len(failed_cmds),
                },
                granularity="turn",
                reason=reason,
            )
        )

    return scores

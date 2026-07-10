from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

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
    raw_summary: str = ""

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
]


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
    return None


# --- Framework detection ---

def _detect_test_framework(cmd: str) -> str:
    cmd_lower = cmd.lower()
    if "pytest" in cmd_lower or "python" in cmd_lower and "pytest" in cmd_lower:
        return "pytest"
    if "jest" in cmd_lower or "vitest" in cmd_lower:
        return "jest"
    if "cargo test" in cmd_lower:
        return "cargo"
    if "go test" in cmd_lower:
        return "go"
    if "rspec" in cmd_lower:
        return "rspec"
    if "unittest" in cmd_lower:
        return "unittest"
    if "npm test" in cmd_lower:
        return "jest"
    return "unknown"


def _detect_build_tool(cmd: str) -> str:
    cmd_lower = cmd.lower()
    if "tsc" in cmd_lower:
        return "tsc"
    if "cargo build" in cmd_lower:
        return "cargo"
    if "go build" in cmd_lower:
        return "go"
    if "npm run build" in cmd_lower:
        return "npm"
    if "pyright" in cmd_lower:
        return "pyright"
    if re.search(r"\bg(?:cc|\+\+)\b", cmd):
        return "gcc"
    if "make" in cmd_lower:
        return "make"
    return "unknown"


def _detect_lint_tool(cmd: str) -> str:
    cmd_lower = cmd.lower()
    if "ruff" in cmd_lower:
        return "ruff"
    if "eslint" in cmd_lower:
        return "eslint"
    if "black" in cmd_lower:
        return "black"
    if "prettier" in cmd_lower:
        return "prettier"
    if "flake8" in cmd_lower:
        return "flake8"
    if "pylint" in cmd_lower:
        return "pylint"
    if "clippy" in cmd_lower:
        return "clippy"
    if "golangci-lint" in cmd_lower:
        return "golangci-lint"
    return "unknown"


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
    for line in output.splitlines():
        if "passed" in line or "failed" in line:
            r.raw_summary = line.strip()
            break


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
        r.raw_summary = m.group(0)


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

    success = (exit_code == 0) if exit_code is not None else (error_count is None or error_count == 0)

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


# --- Extraction from tool spans ---

def _extract_command(span: ToolSpan) -> str | None:
    try:
        args = json.loads(span.arguments)
        return args.get("command")
    except (json.JSONDecodeError, TypeError):
        return None


def _extract_exit_code(span: ToolSpan) -> int | None:
    if span.status_code == "OK":
        return 0
    if span.status_code == "ERROR":
        return 1
    try:
        result = json.loads(span.result)
        if isinstance(result, dict) and "exit_code" in result:
            return result["exit_code"]
    except (json.JSONDecodeError, TypeError):
        pass
    return None


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
        output = tc.result or ""

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

    return outcomes


# --- Turn-level scoring ---

def score_turn_outcomes(turn: TurnSpan) -> list[Score]:
    outcomes = extract_outcomes(turn.tool_calls)
    if not outcomes:
        return []

    scores = []

    tests = [o for o in outcomes if isinstance(o, TestRunOutcome)]
    if tests:
        all_passed = all(t.success for t in tests)
        total_passed = sum(t.passed or 0 for t in tests)
        total_failed = sum(t.failed or 0 for t in tests)
        scores.append(Score(
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
        ))

    builds = [o for o in outcomes if isinstance(o, BuildOutcome)]
    if builds:
        all_ok = all(b.success for b in builds)
        scores.append(Score(
            scorer="outcome.build",
            value=1.0 if all_ok else 0.0,
            tags=["build_pass"] if all_ok else ["build_failure"],
            confidence=1.0,
            metadata={"outcomes": [b.to_dict() for b in builds]},
            granularity="turn",
        ))

    lints = [o for o in outcomes if isinstance(o, LintOutcome)]
    if lints:
        all_clean = all(l.clean for l in lints)
        scores.append(Score(
            scorer="outcome.lint",
            value=1.0 if all_clean else 0.0,
            tags=["lint_clean"] if all_clean else ["lint_issues"],
            confidence=1.0,
            metadata={"outcomes": [l.to_dict() for l in lints]},
            granularity="turn",
        ))

    return scores

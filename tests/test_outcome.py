from __future__ import annotations

from datetime import datetime, timezone

from weave_agent_signals.models import ToolSpan, TurnSpan
from weave_agent_signals.scorers.outcome import (
    TestRunOutcome,
    BuildOutcome,
    LintOutcome,
    classify_command,
    parse_test_output,
    parse_build_output,
    parse_lint_output,
    extract_outcomes,
    score_turn_outcomes,
)


def _ts(h=12, m=0):
    return datetime(2026, 7, 9, h, m, tzinfo=timezone.utc)


def _bash(cmd, result="", status="OK"):
    return ToolSpan(
        span_id="sp-1",
        tool_name="Bash",
        arguments=f'{{"command": "{cmd}"}}',
        result=result,
        status_code=status,
        started_at=_ts(),
        ended_at=_ts(12, 1),
    )


def _turn(tool_calls):
    return TurnSpan(
        trace_id="t1",
        conversation_id="c1",
        started_at=_ts(),
        ended_at=_ts(12, 5),
        model="claude-opus-4",
        input_tokens=1000,
        output_tokens=500,
        cache_read_tokens=200,
        status_code="OK",
        config_version="abc",
        git_branch="main",
        effort_level="high",
        session_id="s1",
        steering_count=0,
        denial_count=0,
        tool_error_count=0,
        events=[],
        tool_calls=tool_calls,
        chat_spans=[],
        subagents=[],
    )


# --- Command classification ---

def test_classify_pytest():
    assert classify_command("pytest tests/") == "test"
    assert classify_command("python -m pytest -x") == "test"
    assert classify_command("python3 -m pytest tests/test_foo.py") == "test"


def test_classify_jest():
    assert classify_command("npm test") == "test"
    assert classify_command("npx jest --coverage") == "test"
    assert classify_command("npx vitest run") == "test"


def test_classify_cargo_test():
    assert classify_command("cargo test") == "test"


def test_classify_go_test():
    assert classify_command("go test ./...") == "test"


def test_classify_build():
    assert classify_command("cargo build") == "build"
    assert classify_command("npm run build") == "build"
    assert classify_command("tsc --noEmit") == "build"
    assert classify_command("go build ./cmd/server") == "build"


def test_classify_lint():
    assert classify_command("ruff check .") == "lint"
    assert classify_command("eslint src/") == "lint"
    assert classify_command("black --check .") == "lint"
    assert classify_command("pyright") == "lint"


def test_classify_cd_prefix():
    assert classify_command("cd /app && pytest") == "test"


def test_classify_env_prefix():
    assert classify_command("PYTHONPATH=. pytest tests/") == "test"


def test_classify_pipe():
    assert classify_command("pytest | tee log.txt") == "test"


def test_classify_unknown():
    assert classify_command("ls -la") is None
    assert classify_command("cat foo.py") is None


# --- Pytest parsing ---

def test_parse_pytest_all_passed():
    output = "===== 42 passed in 3.21s ====="
    result = parse_test_output(output, "pytest")
    assert result.passed == 42
    assert result.failed == 0
    assert result.framework == "pytest"


def test_parse_pytest_with_failures():
    output = "===== 10 passed, 3 failed, 1 error in 5.0s ====="
    result = parse_test_output(output, "pytest")
    assert result.passed == 10
    assert result.failed == 3
    assert result.errors == 1


def test_parse_pytest_with_skips():
    output = "===== 5 passed, 2 skipped in 1.0s ====="
    result = parse_test_output(output, "pytest")
    assert result.passed == 5
    assert result.skipped == 2
    assert result.failed == 0


# --- Jest parsing ---

def test_parse_jest_passed():
    output = """Tests:       12 passed, 12 total
Suites:      3 passed, 3 total"""
    result = parse_test_output(output, "jest")
    assert result.passed == 12
    assert result.failed == 0
    assert result.framework == "jest"


def test_parse_jest_with_failures():
    output = """Tests:       2 failed, 10 passed, 12 total"""
    result = parse_test_output(output, "jest")
    assert result.passed == 10
    assert result.failed == 2


# --- Cargo test parsing ---

def test_parse_cargo_test():
    output = "test result: ok. 15 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.30s"
    result = parse_test_output(output, "cargo")
    assert result.passed == 15
    assert result.failed == 0
    assert result.framework == "cargo"


def test_parse_cargo_test_failure():
    output = "test result: FAILED. 10 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out"
    result = parse_test_output(output, "cargo")
    assert result.passed == 10
    assert result.failed == 2


# --- Go test parsing ---

def test_parse_go_test_ok():
    output = """ok  	github.com/user/pkg	0.003s
ok  	github.com/user/pkg/sub	0.005s"""
    result = parse_test_output(output, "go")
    assert result.passed == 2
    assert result.failed == 0
    assert result.framework == "go"


def test_parse_go_test_failure():
    output = """--- FAIL: TestFoo (0.00s)
FAIL	github.com/user/pkg	0.003s
ok  	github.com/user/pkg/sub	0.005s"""
    result = parse_test_output(output, "go")
    assert result.failed == 1
    assert result.passed == 1


# --- Exit code fallback ---

def test_exit_code_fallback_pass():
    tc = _bash("./run_tests.sh", result='{"exit_code": 0}', status="OK")
    outcomes = extract_outcomes([tc])
    assert len(outcomes) == 0  # unknown command, no outcome


def test_exit_code_on_known_command():
    tc = _bash("pytest", result="no parseable output", status="ERROR")
    outcomes = extract_outcomes([tc])
    assert len(outcomes) == 1
    assert outcomes[0].framework == "pytest"
    assert outcomes[0].passed is None  # couldn't parse counts


# --- Build parsing ---

def test_parse_tsc_errors():
    output = "Found 3 errors in 2 files."
    result = parse_build_output(output, "tsc")
    assert result.success is False
    assert result.error_count == 3


def test_parse_tsc_clean():
    output = ""
    result = parse_build_output(output, "tsc", exit_code=0)
    assert result.success is True


# --- Lint parsing ---

def test_parse_ruff_clean():
    output = "All checks passed!"
    result = parse_lint_output(output, "ruff", exit_code=0)
    assert result.clean is True


def test_parse_ruff_issues():
    output = "Found 5 errors."
    result = parse_lint_output(output, "ruff", exit_code=1)
    assert result.clean is False
    assert result.issue_count == 5


# --- Turn-level scoring ---

def test_score_turn_all_pass():
    tc = _bash("pytest tests/", result="===== 10 passed in 1.0s =====")
    scores = score_turn_outcomes(_turn([tc]))
    test_scores = [s for s in scores if s.scorer == "outcome.test"]
    assert len(test_scores) == 1
    assert test_scores[0].value == 1.0
    assert "test_pass" in test_scores[0].tags


def test_score_turn_with_failure():
    tc = _bash("pytest", result="===== 8 passed, 2 failed in 3.0s =====")
    scores = score_turn_outcomes(_turn([tc]))
    test_scores = [s for s in scores if s.scorer == "outcome.test"]
    assert test_scores[0].value == 0.0
    assert "test_failure" in test_scores[0].tags


def test_score_turn_no_bash():
    tc = ToolSpan(
        span_id="sp-1", tool_name="Read", arguments="{}", result="file contents",
        status_code="OK", started_at=_ts(), ended_at=_ts(12, 1),
    )
    scores = score_turn_outcomes(_turn([tc]))
    assert len(scores) == 0


def test_score_turn_multiple_types():
    t1 = _bash("pytest", result="===== 5 passed in 1.0s =====")
    t2 = _bash("ruff check .", result="All checks passed!", status="OK")
    scores = score_turn_outcomes(_turn([t1, t2]))
    scorer_names = {s.scorer for s in scores}
    assert "outcome.test" in scorer_names
    assert "outcome.lint" in scorer_names

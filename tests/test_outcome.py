from __future__ import annotations

from datetime import datetime, timezone

from weave_agent_signals.models import ToolSpan, TurnSpan
from weave_agent_signals.scorers.outcome import (
    CommandOutcome,
    GitOutcome,
    InstallOutcome,
    _extract_command,
    _extract_exit_code,
    classify_command,
    extract_outcomes,
    parse_build_output,
    parse_git_output,
    parse_install_output,
    parse_lint_output,
    parse_test_output,
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


def test_classify_git():
    assert classify_command("git commit -m 'fix bug'") == "git"
    assert classify_command("git push origin main") == "git"
    assert classify_command("git merge feature-branch") == "git"
    assert classify_command("git rebase main") == "git"


def test_classify_git_non_mutating():
    assert classify_command("git status") is None
    assert classify_command("git log --oneline") is None
    assert classify_command("git diff") is None


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
    output = (
        "test result: ok. 15 passed; 0 failed; 0 ignored;"
        " 0 measured; 0 filtered out; finished in 2.30s"
    )
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
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], CommandOutcome)
    assert outcomes[0].success is True


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
        span_id="sp-1",
        tool_name="Read",
        arguments="{}",
        result="file contents",
        status_code="OK",
        started_at=_ts(),
        ended_at=_ts(12, 1),
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


# --- Git parsing ---


def test_parse_git_commit():
    output = " 3 files changed, 42 insertions(+), 10 deletions(-)"
    result = parse_git_output(output, "commit", exit_code=0)
    assert result.success is True
    assert result.files_changed == 3
    assert result.insertions == 42
    assert result.deletions == 10


def test_parse_git_commit_failure():
    result = parse_git_output("nothing to commit", "commit", exit_code=1)
    assert result.success is False


def test_parse_git_push_success():
    result = parse_git_output("Everything up-to-date", "push", exit_code=0)
    assert result.success is True
    assert result.operation == "push"


def test_parse_git_merge_conflict():
    output = "CONFLICT (content): Merge conflict in foo.py\nAutomatic merge failed"
    result = parse_git_output(output, "merge", exit_code=1)
    assert result.success is False


def test_classify_git_command():
    assert classify_command("git commit -m 'test'") == "git"


def test_extract_git_outcome():
    tc = _bash(
        "git commit -m 'fix'",
        result=" 2 files changed, 15 insertions(+), 3 deletions(-)",
    )
    outcomes = extract_outcomes([tc])
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], GitOutcome)
    assert outcomes[0].success is True


def test_score_turn_git():
    tc = _bash("git commit -m 'fix'", result=" 1 file changed, 5 insertions(+)")
    scores = score_turn_outcomes(_turn([tc]))
    git_scores = [s for s in scores if s.scorer == "outcome.git"]
    assert len(git_scores) == 1
    assert git_scores[0].value == 1.0
    assert "git_success" in git_scores[0].tags


# --- Edge cases ---


def test_mixed_test_and_git_in_one_turn():
    t1 = _bash("pytest tests/", result="===== 3 passed in 0.5s =====")
    t2 = _bash("git commit -m 'all green'", result=" 2 files changed, 10 insertions(+)")
    scores = score_turn_outcomes(_turn([t1, t2]))
    scorer_names = {s.scorer for s in scores}
    assert "outcome.test" in scorer_names
    assert "outcome.git" in scorer_names
    assert all(s.value == 1.0 for s in scores)


def test_empty_bash_output():
    tc = _bash("pytest", result="")
    outcomes = extract_outcomes([tc])
    assert len(outcomes) == 1
    assert outcomes[0].passed is None


def test_git_push_rejected():
    output = "! [rejected]        main -> main (non-fast-forward)\nerror: failed to push"
    result = parse_git_output(output, "push", exit_code=1)
    assert result.success is False
    assert result.operation == "push"


def test_git_rebase_success():
    result = parse_git_output(
        "Successfully rebased and updated refs/heads/main.", "rebase", exit_code=0
    )
    assert result.success is True


def test_git_commit_with_diffstat():
    output = """\
[main abc1234] fix: resolve import issue
 3 files changed, 42 insertions(+), 7 deletions(-)
 create mode 100644 src/new_file.py"""
    result = parse_git_output(output, "commit", exit_code=0)
    assert result.success is True
    assert result.files_changed == 3
    assert result.insertions == 42
    assert result.deletions == 7


def test_classify_git_cherry_pick():
    assert classify_command("git cherry-pick abc123") == "git"


def test_classify_git_stash():
    assert classify_command("git stash") == "git"


def test_classify_git_read_only_commands():
    assert classify_command("git branch -a") is None
    assert classify_command("git show HEAD") is None
    assert classify_command("git blame foo.py") is None
    assert classify_command("git fetch origin") is None


def test_classify_git_add_then_commit_compound():
    # the most common commit form: a non-mutating verb precedes the commit
    assert classify_command("git add -A && git commit -m 'fix'") == "git"
    assert classify_command("git add . && git commit -m 'x' && git push") == "git"


def test_classify_compound_still_none_for_read_only():
    assert classify_command("git add -A && git status") is None


def test_extract_git_operation_picks_mutating_verb():
    # operation should be the mutating verb (commit), not the leading `add`
    tc = _bash("git add -A && git commit -m 'fix'", result=" 1 file changed, 2 insertions(+)")
    outcomes = extract_outcomes([tc])
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], GitOutcome)
    assert outcomes[0].operation == "commit"
    assert outcomes[0].success is True


def test_non_bash_tool_ignored():
    tc = ToolSpan(
        span_id="sp-1",
        tool_name="Edit",
        arguments='{"file_path": "foo.py", "old_string": "a", "new_string": "b"}',
        result="ok",
        status_code="OK",
        started_at=_ts(),
        ended_at=_ts(12, 1),
    )
    outcomes = extract_outcomes([tc])
    assert len(outcomes) == 0


def test_score_turn_git_failure():
    tc = _bash("git push origin main", result="rejected", status="ERROR")
    scores = score_turn_outcomes(_turn([tc]))
    git_scores = [s for s in scores if s.scorer == "outcome.git"]
    assert len(git_scores) == 1
    assert git_scores[0].value == 0.0
    assert "git_failure" in git_scores[0].tags


def test_multiple_test_runs_one_fails():
    t1 = _bash("pytest tests/unit/", result="===== 20 passed in 2.0s =====")
    t2 = _bash("pytest tests/integration/", result="===== 3 passed, 1 failed in 5.0s =====")
    scores = score_turn_outcomes(_turn([t1, t2]))
    test_scores = [s for s in scores if s.scorer == "outcome.test"]
    assert len(test_scores) == 1
    assert test_scores[0].value == 0.0
    assert test_scores[0].metadata["fail_count"] == 1


def test_extract_exit_code_prefers_json_over_status():
    span = ToolSpan(
        span_id="sp-1",
        tool_name="Bash",
        arguments='{"command": "pytest"}',
        result='{"exit_code": 2}',
        status_code="ERROR",
        started_at=_ts(),
        ended_at=_ts(12, 1),
    )
    assert _extract_exit_code(span) == 2


def test_extract_exit_code_falls_back_to_status():
    span = ToolSpan(
        span_id="sp-1",
        tool_name="Bash",
        arguments='{"command": "pytest"}',
        result="plain text output",
        status_code="ERROR",
        started_at=_ts(),
        ended_at=_ts(12, 1),
    )
    assert _extract_exit_code(span) == 1


def test_extract_exit_code_ok_status():
    span = ToolSpan(
        span_id="sp-1",
        tool_name="Bash",
        arguments='{"command": "pytest"}',
        result="",
        status_code="OK",
        started_at=_ts(),
        ended_at=_ts(12, 1),
    )
    assert _extract_exit_code(span) == 0


def test_extract_exit_code_unset_no_result():
    span = ToolSpan(
        span_id="sp-1",
        tool_name="Bash",
        arguments='{"command": "pytest"}',
        result="",
        status_code="UNSET",
        started_at=_ts(),
        ended_at=_ts(12, 1),
    )
    assert _extract_exit_code(span) is None


# --- _extract_command hardening ---


def test_extract_command_non_dict_json():
    span = ToolSpan(
        span_id="sp-1",
        tool_name="Bash",
        arguments='"just a string"',
        result="",
        status_code="OK",
        started_at=_ts(),
        ended_at=_ts(12, 1),
    )
    assert _extract_command(span) is None


def test_extract_command_list_json():
    span = ToolSpan(
        span_id="sp-1",
        tool_name="Bash",
        arguments='["a", "b"]',
        result="",
        status_code="OK",
        started_at=_ts(),
        ended_at=_ts(12, 1),
    )
    assert _extract_command(span) is None


# --- parse_git_output unknown exit code ---


def test_parse_git_output_unknown_exit_defaults_false():
    result = parse_git_output("some output", "commit", exit_code=None)
    assert result.success is False


def test_parse_git_output_infers_success_from_content():
    output = "[main abc1234] fix: resolve issue\n 3 files changed, 42 insertions(+)"
    result = parse_git_output(output, "commit", exit_code=None)
    assert result.success is True
    assert result.files_changed == 3


def test_parse_git_output_infers_push_success():
    result = parse_git_output("Everything up-to-date", "push", exit_code=None)
    assert result.success is True


def test_parse_git_output_infers_failure_from_error():
    result = parse_git_output("fatal: not a git repository", "commit", exit_code=None)
    assert result.success is False


def test_parse_git_output_rejected_push_not_success():
    # A rejected push prints "To github.com:..." before the rejection. The
    # success heuristic must not treat that line as a successful push.
    output = (
        "To github.com:user/repo.git\n"
        " ! [rejected]        main -> main (non-fast-forward)\n"
        "error: failed to push some refs to 'github.com:user/repo.git'"
    )
    result = parse_git_output(output, "push", exit_code=None)
    assert result.success is False


def test_parse_git_output_merge_conflict_not_success():
    output = "Auto-merging file.py\nCONFLICT (content): Merge conflict in file.py"
    result = parse_git_output(output, "merge", exit_code=None)
    assert result.success is False


# --- Install classification ---


def test_classify_npm_install():
    assert classify_command("npm install") == "install"
    assert classify_command("npm ci") == "install"


def test_classify_pip_install():
    assert classify_command("pip install requests") == "install"
    assert classify_command("pip install -r requirements.txt") == "install"


def test_classify_uv_install():
    assert classify_command("uv pip install requests") == "install"
    assert classify_command("uv add httpx") == "install"
    assert classify_command("uv sync") == "install"


def test_classify_poetry_install():
    assert classify_command("poetry install") == "install"


def test_classify_cargo_add():
    assert classify_command("cargo add serde") == "install"


def test_classify_go_get():
    assert classify_command("go get github.com/pkg/errors") == "install"
    assert classify_command("go mod download") == "install"


def test_classify_yarn_install():
    assert classify_command("yarn install") == "install"
    assert classify_command("yarn add lodash") == "install"


def test_classify_bundle_install():
    assert classify_command("bundle install") == "install"


# --- Install output parsing ---


def test_parse_install_success():
    result = parse_install_output("Successfully installed requests-2.31.0", "pip", exit_code=0)
    assert result.success is True
    assert result.error_message is None


def test_parse_install_failure():
    output = "ERROR: Could not find a version that satisfies the requirement nonexistent-pkg"
    result = parse_install_output(output, "pip", exit_code=1)
    assert result.success is False
    assert result.error_message is not None
    assert "Could not find" in result.error_message


def test_parse_install_npm_eresolve():
    output = "npm ERR! ERESOLVE unable to resolve dependency tree"
    result = parse_install_output(output, "npm", exit_code=1)
    assert result.success is False
    assert result.error_message is not None


def test_parse_install_unknown_exit_defaults_false():
    result = parse_install_output("some output", "pip", exit_code=None)
    assert result.success is False


def test_parse_install_infers_success_when_no_exit_code():
    # Real Weave data usually has no exit code (status UNSET); a successful
    # install must be inferred from its output, not defaulted to failure.
    result = parse_install_output("added 142 packages in 3s", "npm", exit_code=None)
    assert result.success is True


def test_parse_install_infers_pip_success_when_no_exit_code():
    result = parse_install_output("Successfully installed requests-2.31.0", "pip", exit_code=None)
    assert result.success is True


def test_parse_install_infers_failure_over_success_when_no_exit_code():
    # Hard errors take precedence even if a success-ish word appears.
    output = "installing...\nnpm ERR! code ERESOLVE\nnpm ERR! unable to resolve"
    result = parse_install_output(output, "npm", exit_code=None)
    assert result.success is False


# --- Install scoring ---


def test_score_turn_install_success():
    tc = _bash(
        "pip install requests",
        result="Successfully installed requests-2.31.0",
        status="OK",
    )
    scores = score_turn_outcomes(_turn([tc]))
    install_scores = [s for s in scores if s.scorer == "outcome.install"]
    assert len(install_scores) == 1
    assert install_scores[0].value == 1.0
    assert "install_success" in install_scores[0].tags


def test_score_turn_install_failure():
    tc = _bash("npm install", result="npm ERR! ERESOLVE unable to resolve", status="ERROR")
    scores = score_turn_outcomes(_turn([tc]))
    install_scores = [s for s in scores if s.scorer == "outcome.install"]
    assert len(install_scores) == 1
    assert install_scores[0].value == 0.0
    assert "install_failure" in install_scores[0].tags


def test_extract_install_outcome():
    tc = _bash("pip install -r requirements.txt", result="Successfully installed 5 packages")
    outcomes = extract_outcomes([tc])
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], InstallOutcome)
    assert outcomes[0].tool == "pip"


# --- Mypy classification ---


def test_classify_mypy():
    assert classify_command("mypy src/") == "lint"
    assert classify_command("mypy --strict src/app.py") == "lint"


# --- Generic command fallback ---


def test_classify_exploration_commands_none():
    assert classify_command("ls -la") is None
    assert classify_command("cat foo.py") is None
    assert classify_command("grep -r pattern .") is None
    assert classify_command("find . -name '*.py'") is None
    assert classify_command("head -20 file.txt") is None
    assert classify_command("echo hello") is None
    assert classify_command("pwd") is None
    assert classify_command("which python") is None
    assert classify_command("wc -l file.txt") is None
    assert classify_command("tree src/") is None
    assert classify_command("diff a.py b.py") is None


def test_classify_read_only_git_none():
    assert classify_command("git status") is None
    assert classify_command("git log --oneline") is None
    assert classify_command("git diff") is None
    assert classify_command("git branch -a") is None


def test_classify_generic_command_fallback():
    assert classify_command("docker build .") == "command"
    assert classify_command("docker compose up -d") == "command"
    assert classify_command("python script.py") == "command"
    assert classify_command("node server.js") == "command"
    assert classify_command("terraform apply") == "command"
    assert classify_command("kubectl apply -f deploy.yaml") == "command"
    assert classify_command("curl -X POST http://localhost:8080") == "command"
    assert classify_command("./run.sh") == "command"


def test_extract_generic_command_outcome():
    tc = _bash("docker build .", result="Successfully built abc123", status="OK")
    outcomes = extract_outcomes([tc])
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], CommandOutcome)
    assert outcomes[0].success is True


def test_extract_generic_command_failure():
    tc = _bash("docker build .", result="Error: Dockerfile not found", status="ERROR")
    outcomes = extract_outcomes([tc])
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], CommandOutcome)
    assert outcomes[0].success is False


def test_extract_generic_command_unknown_exit():
    tc = _bash("python script.py", result="output", status="UNSET")
    outcomes = extract_outcomes([tc])
    assert len(outcomes) == 1
    assert outcomes[0].success is None


def test_score_turn_generic_command_failure():
    tc = _bash("docker build .", result="failed", status="ERROR")
    scores = score_turn_outcomes(_turn([tc]))
    cmd_scores = [s for s in scores if s.scorer == "outcome.command"]
    assert len(cmd_scores) == 1
    assert cmd_scores[0].value == 0.0
    assert cmd_scores[0].confidence == 0.5
    assert "command_failure" in cmd_scores[0].tags


def test_score_turn_generic_command_success_no_score():
    """Successful generic commands don't emit a score — only failures are interesting."""
    tc = _bash("docker build .", result="done", status="OK")
    scores = score_turn_outcomes(_turn([tc]))
    cmd_scores = [s for s in scores if s.scorer == "outcome.command"]
    assert len(cmd_scores) == 0


def test_score_turn_mixed_specific_and_generic():
    """Specific classifiers take priority; generic picks up the rest."""
    t1 = _bash("pytest tests/", result="===== 5 passed in 1.0s =====")
    t2 = _bash("docker build .", result="failed", status="ERROR")
    scores = score_turn_outcomes(_turn([t1, t2]))
    scorer_names = {s.scorer for s in scores}
    assert "outcome.test" in scorer_names
    assert "outcome.command" in scorer_names


def test_extract_output_from_json_result():
    """Real Weave data wraps output in {"stdout": ..., "stderr": ...}."""
    from weave_agent_signals.scorers.outcome import _extract_output_text

    span = _bash("git commit -m fix", result='{"stdout": "[main abc1234] 3 files changed"}')
    text = _extract_output_text(span)
    assert "[main abc1234]" in text


def test_git_outcome_from_json_wrapped_result():
    tc = _bash(
        "git commit -m 'fix'",
        result=(
            '{"stdout": "[main 12a51da] fix\\n 3 files changed, 42 insertions(+), 10 deletions(-)"}'
        ),
        status="UNSET",
    )
    outcomes = extract_outcomes([tc])
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], GitOutcome)
    assert outcomes[0].success is True
    assert outcomes[0].files_changed == 3


def test_extract_output_empty_stdout_stderr_returns_empty():
    """Both fields empty → empty string, not the raw JSON wrapper."""
    from weave_agent_signals.scorers.outcome import _extract_output_text

    span = _bash("git status", result='{"stdout": "", "stderr": ""}')
    assert _extract_output_text(span) == ""


def test_extract_exit_code_coerces_string():
    """A JSON exit_code serialized as a string must compare as an int."""
    from weave_agent_signals.scorers.outcome import _extract_exit_code

    span = _bash("pytest", result='{"stdout": "ok", "exit_code": "0"}', status="UNSET")
    assert _extract_exit_code(span) == 0


def test_extract_exit_code_explicit_null_is_unknown_not_status():
    """An explicit exit_code:null means unknown — must NOT fall through to
    status_code=OK and report success (which would mask a failed command)."""
    from weave_agent_signals.scorers.outcome import _extract_exit_code

    span = _bash("git push", result='{"exit_code": null, "stderr": "failed"}', status="OK")
    assert _extract_exit_code(span) is None


def test_exploration_not_scored():
    """Exploration commands should produce no outcomes at all."""
    t1 = _bash("ls -la", result="total 42\ndrwxr-xr-x ...")
    t2 = _bash("cat foo.py", result="import os")
    t3 = _bash("grep -r TODO .", result="file.py:# TODO fix")
    scores = score_turn_outcomes(_turn([t1, t2, t3]))
    assert len(scores) == 0

# Spec 02: Outcome extractor

Parses verifiable environment outcomes from `execute_tool` Bash spans. The strongest deterministic signal — when a test suite runs and reports results, that's ground truth no LLM judge can beat.

Ties to spec 01 (data flow — reads `ToolSpan`), spec 05 (score write-back).

---

## What we parse (M1)

Four evidence types from Bash tool calls, all deterministic (Tier 1 — regex detection + framework parsers):

### 1. Test runs

**Detection**: command pattern matching on the Bash `arguments.command` field.

| Pattern | Framework | Notes |
|---|---|---|
| `pytest`, `python -m pytest` | pytest | Most common in this user's sessions |
| `npm test`, `npx jest`, `vitest` | JS test runners | |
| `cargo test` | Rust | |
| `go test` | Go | |
| `make test`, `make check` | Make-based | |
| `python -m unittest` | unittest | |
| `rspec`, `bundle exec rspec` | Ruby | |

**Result parsing** (from `result` field — typically `stdout` + exit code):

```python
@dataclass
class TestOutcome:
    framework: str          # "pytest", "jest", etc.
    command: str            # the full command
    exit_code: int | None   # 0 = pass, non-zero = fail
    passed: int | None      # parsed from output
    failed: int | None
    errors: int | None
    skipped: int | None
    total: int | None
    duration_s: float | None
    raw_summary: str        # the summary line from output
```

Parsing is framework-specific:
- **pytest**: `"(\d+) passed"`, `"(\d+) failed"`, `"(\d+) error"` from the summary line
- **jest/vitest**: `"Tests:\s+(\d+) passed"`, `"(\d+) failed"`
- **cargo test**: `"test result: (ok|FAILED). (\d+) passed; (\d+) failed"`
- **go test**: `"(ok|FAIL)\s+"` per package + `"--- FAIL"` count
- **Fallback**: if framework-specific parsing fails, use exit code alone (0 = pass, else fail)

### 2. Build/compile

**Detection**: `make`, `gcc`, `g++`, `cargo build`, `npm run build`, `tsc`, `go build`, `python -m py_compile`.

```python
@dataclass
class BuildOutcome:
    tool: str               # "make", "cargo", "tsc", etc.
    command: str
    exit_code: int | None
    success: bool
    error_count: int | None   # parsed from compiler output
    warning_count: int | None
```

### 3. Lint/format

**Detection**: `ruff`, `eslint`, `flake8`, `pylint`, `black --check`, `prettier --check`, `clippy`, `golangci-lint`, `pyright`.

**Note**: `pyright` is classified as lint, not build — it's a type checker, not a compiler. Checked before build patterns to avoid misclassification.

```python
@dataclass
class LintOutcome:
    tool: str
    command: str
    exit_code: int | None
    clean: bool             # no issues found
    issue_count: int | None
    fix_count: int | None   # for auto-fix tools
```

### 4. Git operations

**Detection**: `git commit`, `git push`, `git merge`, `git rebase`.

```python
@dataclass
class GitOutcome:
    operation: str          # "commit", "push", "merge", "rebase"
    command: str
    exit_code: int | None
    success: bool
    files_changed: int | None  # from commit output
    insertions: int | None
    deletions: int | None
```

---

## Command parsing

Bash commands can be complex. Rules:

1. **Pipes** (`|`): classify the first command (the producer). `pytest | tee log` is a test run.
2. **cd prefixes** (`cd /path && pytest`): strip the `cd`, classify the rest.
3. **env prefixes** (`PYTHONPATH=. pytest`): strip env assignments, classify the rest.
4. **Timeout wrappers** (`timeout 30 pytest`): strip the wrapper, classify the rest.
5. **Subshells** (`$(...)`, backticks): skip — too ambiguous.

Not handled in M1: chained commands (`&&`, `||`, `;`) are not split — only the first segment before a pipe is classified. This is a known gap; false negatives (missed commands) are acceptable, false positives are not.

Implementation: regex-based command classifier, not a shell parser. Classification order: test → lint → build (lint checked before build to avoid misclassifying type checkers as compilers).

---

## Extraction tiers

M1 implements **Tier 1 only** — regex detection + framework parsers. `confidence=1.0`. Expected to cover the large majority of this user's sessions (pytest-dominant). Exit-code fallback is built into the framework parsers (if output doesn't parse, exit code alone determines pass/fail).

Backfill should report coverage (% of Bash spans classified, size of the unclassified residue) so coverage is measured, not assumed.

---

## Turn-level scoring

The outcome extractor produces a `Score` per turn:

```python
Score(
    scorer="outcome.test",
    value=1.0 if all_tests_passed else 0.0,
    tags=["test_pass"] or ["test_failure"],
    confidence=1.0,     # deterministic, always high confidence
    metadata={
        "outcomes": [outcome.to_dict() for outcome in outcomes],
        "test_count": total_tests,
        "fail_count": total_failures,
    },
    granularity="turn",
)
```

Multiple outcome types in one turn → multiple scores (one per type: `outcome.test`, `outcome.build`, `outcome.lint`, `outcome.git`).

**Attribution caveat**: a turn-level outcome score marks *where tests ran*, not whose work passed — tests passing in turn N reflect work from turns 1..N. Turn-level scores are anchors for navigation; the meaningful units are session-level aggregates and **arcs**: `session_ended_green` (last test outcome in session passed), `recovered` (fail → later pass), `ended_red` (fail never resolved). Arcs are computed at session level from the turn anchors.

---

## Edge cases

- **Truncated output**: Bash results are capped at 32KB by the adapter. If the summary line is truncated, fall back to exit code.
- **No exit code**: some tool results don't include explicit exit codes. If missing, infer from output patterns ("FAILED", "Error:", etc.).
- **Nested test runs**: `make test` might invoke `pytest` internally. Classify as the outer command; don't double-count from nested output.
- **Interactive commands**: `git rebase -i` — skip (not observable from output alone).
- **Dry runs**: `pytest --collect-only`, `black --check --diff` — classify as lint/check, not test/format.

## Non-goals

- Parsing non-Bash tool outputs (Read, Edit, Write have no verifiable outcomes in this sense)
- Semantic analysis of code changes (that's L2's job)
- Inferring test quality (coverage, assertion depth — not observable from output)

# Spec 02: Outcome extractor

Parses verifiable environment outcomes from `execute_tool` Bash spans. The strongest deterministic signal — when a test suite runs and reports results, that's ground truth no LLM judge can beat.

Ties to spec 01 (data flow — reads `ToolSpan`), spec 05 (score write-back), routing gates (FUTURE.md — `has_outcome` gate).

---

## What we parse

Three evidence types from Bash tool calls, plus git operations:

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

**Detection**: `make`, `gcc`, `g++`, `cargo build`, `npm run build`, `tsc`, `go build`, `python -m py_compile`, `pyright`.

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

**Detection**: `ruff`, `eslint`, `flake8`, `pylint`, `black --check`, `prettier --check`, `clippy`, `golangci-lint`.

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

1. **Chained commands** (`&&`, `||`, `;`): split and classify each segment independently. The overall exit code reflects the chain semantics (`&&` = all must pass).
2. **Pipes** (`|`): classify the first command (the producer). `pytest | tee log` is a test run.
3. **Subshells** (`$(...)`, backticks): skip — too ambiguous.
4. **cd prefixes** (`cd /path && pytest`): strip the `cd`, classify the rest.
5. **env prefixes** (`PYTHONPATH=. pytest`): strip env assignments, classify the rest.
6. **Timeout wrappers** (`timeout 30 pytest`): strip the wrapper, classify the rest.

Implementation: regex-based command classifier, not a shell parser. False negatives (missed commands) are fine — false positives (misclassified commands) are not.

---

## Tiered extraction (deterministic core + LLM fallback)

Pure regex has a long tail (`./run_tests.sh`, tox/nox, npm scripts aliasing pytest, missing exit codes, truncated summaries). The extractor is tiered so determinism covers the common case and an LLM plugs the tail:

1. **Tier 1 — regex detection + framework parsers** (the tables above): free, exact, `confidence=1.0`. Expected to cover the large majority of this user's sessions (pytest-dominant).
2. **Tier 2 — LLM extractor fallback**: for Bash spans that *look* outcome-ish (test-ish keywords, `FAILED`/`passed` patterns, nonzero exit) but don't parse cleanly, a small model reads the single command + output and emits structured JSON (`{kind, framework, success, passed, failed, ...}`). This is **LLM-as-parser, not LLM-as-judge** — single-output extraction is a task small models handle near-perfectly (TRAIL's ~11–22% figure is about issue-finding across whole raw traces, a different task). `confidence=0.9`, extractor model recorded in metadata. Costs ~$0.001/span, only on the unparsed residue.
3. **Tier 3 — exit-code only**: when neither parses, exit code alone (pass/fail, no counts).

Backfill reports coverage per tier (% of Bash spans classified, size of the unclassified residue) so coverage is measured, not assumed. Detection bias is acceptable for A/B purposes because it depends on repo tooling, not on the config surface being measured — but the coverage report makes this checkable.

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

## Verification-before-done check

A meta-scorer: did the agent run tests before claiming the task was done? Split into a deterministic fact and a micro-LLM fact — keyword-matching "done" in assistant text is exactly the never-flexible-enough trap (partial completion, "done with step 1", no explicit claim).

1. **Done-claim detection (Class 3, micro-LLM)**: classify each turn's final assistant message — `{claims_completion: bool, scope: "task" | "subtask" | "none"}`. Only `scope == "task"` counts. Short single-text classification; small model; only needs to run on late-session turns or turns followed by session end.
2. **Test-recency (Class 2, deterministic)**: from the claiming turn, look backwards for the most recent test/build outcome and whether it passed.
3. Score:
   - `verified_before_done = True` if a task-level claim exists and tests ran + passed within the last N turns before it
   - `verified_before_done = False` if a claim exists but no tests ran, or the last run failed
   - `not_applicable` if no completion claim, or micro-LLM judges the session had no testable work

```python
Score(
    scorer="outcome.verified_before_done",
    value=True/False,
    tags=["verified"] or ["unverified"],
    confidence=0.9,     # done-claim comes from the classifier, recency is exact
    metadata={"claim_turn": idx, "last_test_turn": idx, "turns_since_test": gap,
              "classifier_model": "..."},
    granularity="session",
)
```

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

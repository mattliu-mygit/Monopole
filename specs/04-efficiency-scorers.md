# Spec 04: Efficiency scorers

Detects waste patterns in agent behavior — not raw counts (those are anti-metrics), but normalized ratios and structural patterns. An efficient turn does its job with minimal repeated work and no error spirals.

Ties to spec 01 (data flow — reads tool spans), spec 03 (implicit feedback — related but measures different things: feedback measures *user* frustration, efficiency measures *agent* waste).

---

**Prior art to mirror**: anthropics/claude-code#42796 (6,852-session quality-regression analysis; replicated in `lucemia/claude-session-analyzer`, MIT) established a behavioral metric set — Read:Edit ratio, edits-without-prior-Read %, repeated edits, reasoning loops, premature stopping, self-admitted errors per 1K tool calls, interrupt/frustration indicators. Our scorers below cover the loop/repeat subset; adopt the remaining metrics from that list rather than inventing new ones.

## Scorers

### 1. Error loop detection

An error loop is when the agent calls the same tool repeatedly with similar arguments, each time failing. This is the clearest waste signal.

```python
@dataclass
class ErrorLoop:
    tool_name: str
    attempt_count: int
    first_args_hash: str   # hash of first attempt's arguments
    similarity: float      # argument similarity across attempts (0-1)
    all_failed: bool       # every attempt in the loop failed
    span_ids: list[str]
```

Detection algorithm:

```python
def detect_error_loops(tool_calls: list[ToolSpan], threshold: int = 3) -> list[ErrorLoop]:
    # Group consecutive calls to the same tool
    groups = group_consecutive_by_tool(tool_calls)

    loops = []
    for group in groups:
        if len(group) < threshold:
            continue

        # Check if arguments are similar (not identical — agent may tweak between retries)
        similarities = pairwise_arg_similarity(group)
        avg_sim = mean(similarities)
        if avg_sim < 0.5:  # too different to be a retry loop
            continue

        failed = [t for t in group if t.status_code != "OK"]
        if len(failed) / len(group) < 0.6:  # majority must fail
            continue

        loops.append(ErrorLoop(
            tool_name=group[0].tool_name,
            attempt_count=len(group),
            first_args_hash=hash_args(group[0].arguments),
            similarity=avg_sim,
            all_failed=len(failed) == len(group),
            span_ids=[t.span_id for t in group],
        ))
    return loops
```

**Argument similarity**: for Bash commands, compare the command strings (edit distance / max length). For other tools, compare JSON keys + values. Exact match = 1.0, completely different = 0.0.

### 2. Repeated reads

The agent reads the same file multiple times in one turn without modifying it in between. Common when the agent loses track of file contents (context limits, poor planning).

Two deterministic **excuses** before flagging (both visible in the trace):
- **Post-compaction re-reads**: the adapter emits compaction span events; a re-read after a compaction event is rational (context was just destroyed), not waste. Reset the read-tracking at each compaction.
- **Subagent contexts**: tool calls inside subagent `invoke_agent` spans run in separate contexts — two subagents each reading the same file is legitimate. Track repeats per agent scope (main turn vs each subagent), never across scopes.

```python
def detect_repeated_reads(tool_calls: list[ToolSpan]) -> list[RepeatedRead]:
    reads = {}  # file_path -> list of read spans
    writes = set()  # file_paths written/edited between reads

    for tc in tool_calls:
        if tc.tool_name == "Read":
            path = extract_path(tc.arguments)
            if path and path not in writes:
                reads.setdefault(path, []).append(tc)
            writes.discard(path)  # reset after re-read
        elif tc.tool_name in ("Edit", "Write"):
            path = extract_path(tc.arguments)
            if path:
                writes.add(path)
                reads.pop(path, None)  # legitimate re-read after edit

    return [
        RepeatedRead(path=path, count=len(spans), span_ids=[s.span_id for s in spans])
        for path, spans in reads.items()
        if len(spans) >= 2
    ]
```

### 3. Token efficiency ratio

High input tokens with low output tokens suggests the agent is reading more context than it needs. This is a soft signal — some tasks legitimately need heavy reading.

```python
def token_efficiency(turn: TurnSpan) -> float | None:
    if turn.input_tokens == 0:
        return None

    # Cache hit ratio — high is good (reusing context efficiently)
    cache_ratio = turn.cache_read_tokens / turn.input_tokens if turn.input_tokens > 0 else 0

    # Output-to-input ratio — very low suggests wasted reads
    output_ratio = turn.output_tokens / turn.input_tokens

    # Don't score turns with very few tokens (startup, simple queries)
    if turn.input_tokens + turn.output_tokens < 1000:
        return None

    return min(output_ratio * 2 + cache_ratio * 0.5, 1.0)  # normalized 0-1
```

**Caveat**: token efficiency is the weakest signal. It's a feature for L3 pattern analysis more than a standalone score. Include in metadata but use low weight.

---

## Turn-level score

```python
def efficiency_score(turn: TurnSpan) -> Score:
    loops = detect_error_loops(turn.tool_calls)
    repeats = detect_repeated_reads(turn.tool_calls)

    # Binary flags
    has_loops = len(loops) > 0
    has_repeats = len(repeats) > 0

    # Waste severity (0-1)
    loop_waste = sum(l.attempt_count - 1 for l in loops)  # wasted attempts
    repeat_waste = sum(r.count - 1 for r in repeats)      # wasted reads
    total_calls = len(turn.tool_calls) if turn.tool_calls else 1
    waste_ratio = min((loop_waste + repeat_waste) / total_calls, 1.0)

    tags = []
    if has_loops:
        tags.append("error_loop")
    if has_repeats:
        tags.append("repeated_reads")
    if not tags:
        tags.append("efficient")

    return Score(
        scorer="efficiency",
        value=1.0 - waste_ratio,  # 1.0 = fully efficient, 0.0 = all waste
        tags=tags,
        confidence=0.9,
        metadata={
            "error_loops": [l.to_dict() for l in loops],
            "repeated_reads": [r.to_dict() for r in repeats],
            "waste_ratio": waste_ratio,
            "token_efficiency": token_efficiency(turn),
            "tool_call_count": len(turn.tool_calls),
        },
        granularity="turn",
    )
```

---

## Session-level aggregation

```python
def session_efficiency(session: SessionView) -> Score:
    turn_scores = [efficiency_score(t) for t in session.turns]
    avg_value = mean([s.value for s in turn_scores]) if turn_scores else 1.0

    all_loops = sum(len(s.metadata["error_loops"]) for s in turn_scores)
    all_repeats = sum(len(s.metadata["repeated_reads"]) for s in turn_scores)

    tags = []
    if all_loops > 0:
        tags.append("has_error_loops")
    if all_repeats > 0:
        tags.append("has_repeated_reads")
    if not tags:
        tags.append("efficient_session")

    return Score(
        scorer="efficiency.session",
        value=avg_value,
        tags=tags,
        confidence=0.85,
        metadata={
            "total_error_loops": all_loops,
            "total_repeated_reads": all_repeats,
            "turn_count": len(session.turns),
            "avg_turn_efficiency": avg_value,
        },
        granularity="session",
    )
```

---

## Anti-metrics (denominators, NOT scores)

These are computed and stored in score metadata for L3 consumption, but never used as standalone quality signals:

| Metric | Why it's not a score |
|---|---|
| Total turns | Complex tasks need more turns |
| Total tokens | Expensive ≠ wasteful |
| Session duration | Long sessions aren't bad sessions |
| Tool call count | More tools ≠ less efficient |
| Model tier (Opus vs Sonnet) | User's decision, not quality |

They become useful in L3 as denominators (corrections *per turn*, waste *per tool call*) and as features for correlation analysis.

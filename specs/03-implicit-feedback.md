# Spec 03: Implicit feedback

Extracts human-feedback signals from observable user behavior — no explicit rating needed. The adapter already stamps three counters (`steering_count`, `denial_count`, `tool_error_count`) on turn roots; this layer adds semantic scoring and session-level aggregation.

Ties to spec 01 (data flow), spec 04 (efficiency — related but distinct), routing gates (FUTURE.md — `high_frustration` gate).

---

## Signal taxonomy

**Prior art to mirror**: Claude Code's own OTel telemetry enumerates the decision taxonomy — `tool_decision` with `accept|reject` and sources `config|hook|user_permanent|user_temporary|user_abort|user_reject`, plus `tool_result.error_type`. Keep our signal semantics aligned with those field meanings so scores stay comparable with anything built on CC's native telemetry.

### Turn-level signals (from adapter attrs + span events)

| Signal | Source | Weight | Notes |
|---|---|---|---|
| Steering count | `weave_agent_adapter.steering_count` attr | high | Direct correction — user interrupted the agent mid-turn |
| Denial count | `weave_agent_adapter.denial_count` attr | high | User explicitly rejected a tool call |
| Tool errors | `weave_agent_adapter.tool_error_count` attr | medium | Tool failures (may or may not be the agent's fault) |
| Corrective follow-up | Next turn's prompt (heuristic) | high | User's next message contradicts/corrects this turn's output |
| Quick re-prompt | Turn timing + prompt similarity | medium | User re-submits within seconds with modifications |

### Session-level signals (aggregated)

| Signal | Source | Notes |
|---|---|---|
| Session frustration | Weighted sum of turn signals | Overall session quality proxy |
| Abandonment | Session end pattern | Session ends without visible completion marker |
| Satisfaction proxy | Inverse of frustration + positive completion | Session completed with low friction |
| Correction density | Corrections / total turns | How often the user had to intervene |

---

## Turn-level scorers

### Event classification (Class 3, micro-LLM)

Raw counters can't distinguish "no! stop!" from "while you're at it, also add X" — both are one steering event, only one is frustration. So before weighting, classify each steering/denial *text* with a small model (single short message, structured output; fires only on turns that have steering/denials — a small fraction):

| Event | Classes |
|---|---|
| Steering | `correction` (agent was doing the wrong thing) · `addition` (new/expanded ask, no fault) · `redirect` (priorities changed, no fault) |
| Denial | `rejection` (wrong action proposed) · `changed_mind` / `permission_hygiene` (no fault) |

Raw counts stay in metadata as Class-1 facts; the index below consumes only the fault-implying classes.

### Frustration index

Weighted combination over *classified* events:

```python
def frustration_index(turn: TurnSpan) -> float:
    w_steer = 0.4   # corrections — user explicitly interrupted a wrong path
    w_deny  = 0.3   # rejections — user refused a proposed action as wrong
    w_error = 0.1   # errors matter less — often environmental, not agent fault
    w_correct = 0.2 # corrective follow-up — agent got it wrong

    score = 0.0
    score += min(turn.steering_corrections * w_steer, w_steer * 3)  # cap at 3
    score += min(turn.denial_rejections * w_deny, w_deny * 3)
    score += min(turn.tool_error_count * w_error, w_error * 5)
    if is_corrective_followup(turn, next_turn):
        score += w_correct

    return min(score, 1.0)
```

Output:

```python
Score(
    scorer="implicit.frustration",
    value=frustration_index,  # 0.0 = smooth, 1.0 = maximum frustration
    tags=tags,                # e.g. ["steering_heavy", "denial"]
    confidence=0.9,           # high for counter-based, lower for heuristics
    metadata={
        "steering_count": n,            # raw (Class 1)
        "steering_corrections": n,      # classified (Class 3)
        "denial_count": n,
        "denial_rejections": n,
        "tool_error_count": n,
        "corrective_followup": bool,
        "classifier_model": "...",
    },
    granularity="turn",
)
```

### Corrective follow-up detection

Heuristic: the user's *next* turn prompt references or contradicts this turn's output. Turns are already linked and ordered via `conversation_id` (SessionView), so finding the next turn is trivial. Prompt text is fully available on spans too — the adapter caps only tool *results* (32KB); `gen_ai.prompt.0.content` carries the full prompt, with only secret-pattern redaction applied. So text-similarity between consecutive prompts works from spans directly; local transcripts are only needed when tool-output fidelity matters. Signals, strongest first:

0. **Prompt-text similarity/contradiction**: next prompt references or negates this turn's action ("no, ...", "don't ...", re-statement of the same ask with modifications)

1. **Steering at turn start**: if the next turn has `steering_count > 0` in its first event, the user corrected course
2. **Quick re-prompt**: if `next_turn.started_at - turn.ended_at < 30s` and the turn had tool errors or denials, likely a correction
3. **Same-tool retry**: if the next turn re-invokes the same primary tool with different args, likely a correction

This is intentionally conservative — false negatives are better than false positives for a frustration metric.

---

## Session-level scorers

### Abandonment detection

A session is "abandoned" if it ends without a visible completion marker:

```python
def is_abandoned(session: SessionView) -> bool:
    if not session.turns:
        return True

    last_turn = session.turns[-1]

    # If the last turn has an outcome (test pass, git commit), not abandoned
    if has_positive_outcome(last_turn):
        return False

    # If the session has very few turns (1-2), likely just exploration — not abandonment
    if len(session.turns) <= 2:
        return False

    # If the last turn had high frustration, likely abandoned
    if frustration_index(last_turn) > 0.5:
        return True

    # If the session was marked incomplete by the adapter
    if last_turn.status_code == "ERROR":
        return True

    return False
```

### Session frustration (aggregate)

```python
def session_frustration(session: SessionView) -> float:
    if not session.turns:
        return 0.0

    turn_scores = [frustration_index(t) for t in session.turns]

    # Weighted: later turns matter more (frustration compounds)
    weights = [1.0 + i * 0.1 for i in range(len(turn_scores))]
    weighted = sum(s * w for s, w in zip(turn_scores, weights))
    total_weight = sum(weights)

    base = weighted / total_weight

    # Abandonment penalty
    if is_abandoned(session):
        base = min(base + 0.2, 1.0)

    return base
```

### Correction density

Simple ratio — how often did the user have to intervene?

```python
def correction_density(session: SessionView) -> float:
    if len(session.turns) <= 1:
        return 0.0
    corrections = sum(1 for t in session.turns if t.steering_count > 0 or t.denial_count > 0)
    return corrections / len(session.turns)
```

Output scores:

```python
Score(
    scorer="implicit.session_frustration",
    value=session_frustration,
    tags=tags,  # ["abandoned"] or ["smooth"] or ["high_friction"]
    confidence=0.85,
    metadata={
        "turn_count": len(turns),
        "correction_density": density,
        "abandoned": is_abandoned,
        "total_steerings": sum(t.steering_count for t in turns),
        "total_denials": sum(t.denial_count for t in turns),
    },
    granularity="session",
)
```

---

## Calibration

These weights and thresholds are initial guesses. Calibration path:
1. Backfill scores on ~50 recent sessions
2. Manually review ~10 highest/lowest-frustration sessions
3. Adjust weights until the ranking matches intuition
4. Lock weights as v1; version in score metadata for A/B

The L4 RSI loop can later propose weight adjustments based on judge-agreement data.

---

## Anti-signals (explicitly NOT scored here)

- **Turn count**: not a quality signal — complex tasks legitimately need many turns. Used as a denominator in efficiency (spec 04), not a score.
- **Token count**: same — expensive turns aren't bad turns. Feature for L3 correlation, not a score.
- **Session duration**: same.
- **Model choice**: Claude Opus vs Sonnet is a user decision, not a quality signal.

These are **features** for L3 pattern analysis, not scores.

# Spec 03: Implicit feedback

Extracts human-feedback signals from observable user behavior — no explicit rating needed. The adapter already stamps three counters (`steering_count`, `denial_count`, `tool_error_count`) on turn roots; this layer adds session-level aggregation over those facts.

Ties to spec 01 (data flow), spec 04 (efficiency — related but distinct: feedback measures *user* frustration, efficiency measures *agent* waste).

---

## M1 scope

M1 ships two factual session-level scorers: **correction density** and **abandonment detection**. Both rely only on adapter-stamped counters and observable session end-state — no LLM, no prompt-text analysis.

### Signal sources (M1)

| Signal | Source | Notes |
|---|---|---|
| Steering count | `weave_agent_adapter.steering_count` attr | Raw count per turn |
| Denial count | `weave_agent_adapter.denial_count` attr | Raw count per turn |
| Tool errors | `weave_agent_adapter.tool_error_count` attr | Raw count per turn |

Raw counts are available as span attrs for downstream consumers (L3 features, routing gates). M1 does not interpret *what kind* of steering/denial occurred — only whether it happened.

### Correction density

Simple ratio — how often did the user have to intervene?

```python
def correction_density(session: SessionView) -> float:
    if len(session.turns) <= 1:
        return 0.0
    corrections = sum(1 for t in session.turns if t.steering_count > 0 or t.denial_count > 0)
    return corrections / len(session.turns)
```

```python
Score(
    scorer="implicit.correction_density",
    value=density,
    tags=[],
    confidence=0.9,
    metadata={
        "turn_count": len(turns),
        "total_steerings": sum(t.steering_count for t in turns),
        "total_denials": sum(t.denial_count for t in turns),
    },
    granularity="session",
)
```

### Abandonment detection

A session is "abandoned" if it ends without a visible completion marker. M1 uses only observable end-state — no frustration scoring, no outcome checking.

```python
def is_abandoned(session: SessionView) -> bool:
    if not session.turns:
        return True

    # Very few turns (1-2) = likely exploration, not abandonment
    if len(session.turns) <= 2:
        return False

    # Last turn errored out
    if session.turns[-1].status_code == "ERROR":
        return True

    return False
```

```python
Score(
    scorer="implicit.abandonment",
    value=True/False,
    tags=["abandoned"] or ["completed"],
    confidence=0.8,
    metadata={},
    granularity="session",
)
```

---

## Anti-signals (explicitly NOT scored)

- **Turn count**: not a quality signal — complex tasks legitimately need many turns. Used as a denominator in efficiency (spec 04), not a score.
- **Token count**: same — expensive turns aren't bad turns. Feature for L3 correlation, not a score.
- **Session duration**: same.
- **Model choice**: Claude Opus vs Sonnet is a user decision, not a quality signal.

These are **features** for L3 pattern analysis, not scores.


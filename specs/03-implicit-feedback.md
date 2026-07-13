# Spec 03: Implicit feedback

Extracts human-feedback signals from observable user behavior — no explicit rating needed. The adapter already stamps three counters (`steering_count`, `denial_count`, `tool_error_count`) on turn roots; this layer aggregates those facts to the session level.

Ties to spec 01 (data flow), spec 04 (efficiency — related but distinct: this spec measures *user* frustration, efficiency measures *agent* waste).

## M1 scope

Two factual session-level scorers: **correction density** and **abandonment detection**. Both are deliberately limited to adapter-stamped counters and observable session end-state — no LLM, no prompt-text analysis. That interpretive layer (was this steering a correction, a neutral addition, or a redirect? how frustrated was the user, not just how often did they intervene?) is explicitly deferred to M2's micro-LLM classifiers — M1 only counts, it doesn't interpret.

Raw counters are also exposed as span attrs for downstream consumers (L3 features, M2 routing gates), independent of these two scorers.

### Correction density

Ratio of turns where the user had to intervene (steer or deny) to total turns — a frequency signal, not a judgment about whether the intervention was warranted.

### Abandonment detection

A session is "abandoned" if it ends without a visible completion marker (last turn errored out). Very short sessions (1-2 turns) are never flagged as abandoned — too likely to be quick exploration rather than a task given up on; abandonment as a concept only makes sense once there's a task actually in flight.

## Anti-signals (explicitly NOT scored)

Turn count, token count, session duration, and model choice are **not** quality signals here:

- Turn count / duration / tokens: complex tasks legitimately need more of all three — penalizing them would penalize hard work, not bad work.
- Model choice (Opus vs. Sonnet): a user decision, not a quality signal.

These remain useful as **features** for L3 pattern/correlation analysis (e.g. corrections per turn) and as denominators — just never as standalone scores.

# Spec 04: Efficiency scorers

Detects waste patterns in agent behavior — normalized ratios and structural patterns, not raw counts (those are anti-metrics, see below). An efficient turn does its job with minimal repeated work and no error spirals.

Ties to spec 01 (reads tool spans), spec 03 (implicit feedback — related but distinct: that spec measures *user* frustration, this measures *agent* waste).

**Prior art**: anthropics/claude-code#42796 (6,852-session quality-regression analysis, replicated in `lucemia/claude-session-analyzer`, MIT) established a behavioral metric set — Read:Edit ratio, edits-without-prior-Read %, repeated edits, reasoning loops, premature stopping, self-admitted errors per 1K tool calls, interrupt/frustration indicators. The scorers here cover the loop/repeat subset; remaining metrics from that list should be adopted as-is rather than reinvented.

## Error loop detection

The clearest waste signal: the agent calls the same tool repeatedly with similar (not necessarily identical — the agent may tweak between retries) arguments, and most attempts fail. Argument similarity is a fuzzy match (edit distance for Bash commands, key/value comparison for structured args) rather than exact-match, specifically to catch retry loops where the agent varies its approach slightly each time.

## Repeated reads

The agent re-reads the same file within a turn without modifying it in between — usually a sign of lost context. Two deterministic **excuses** are checked before flagging a repeat as waste, both visible in the trace:

- **Post-compaction re-reads**: the adapter emits compaction span events; re-reading right after a compaction is rational (context was just destroyed), not waste. Read-tracking resets at each compaction event.
- **Subagent contexts**: tool calls inside a subagent's `invoke_agent` span run in an isolated context, so two subagents each reading the same file independently is legitimate, not a repeat. Repeats are tracked per agent scope (main turn vs. each subagent), never across scopes.

## Token efficiency ratio

High input tokens with low output tokens can suggest over-reading, but it's the weakest signal here — some tasks legitimately need heavy context. Deliberately **not a standalone scorer**: it lives in the efficiency score's metadata for L3 pattern analysis, not as its own feedback entry or quality judgment.

## Scores

Turn-level efficiency score is `1.0 - waste_ratio`, where waste is loop attempts plus repeated reads normalized by total tool calls. Session-level efficiency averages the turn scores. Both carry loop/repeat detail in metadata for L3.

## Anti-metrics (denominators, NOT scores)

Total turns, total tokens, session duration, tool call count, and model tier are computed and stored in metadata for L3 — as denominators (corrections *per turn*, waste *per tool call*) and correlation features — but are never themselves quality signals. More tools, more tokens, or a longer session doesn't mean less efficient; it may just mean a harder task.

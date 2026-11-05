"""Scorers that need every turn of a Trajectory: a session's or a subagent's.

Each runs on a session Trajectory and on every subagent Trajectory below it. A session's own
turns include the tool calls of the subagents they launched, as turn outcomes do, so a test
that a subagent ran counts for its session too.
"""

from __future__ import annotations

from weave_agent_signals.models import Score, SessionView, ToolSpan
from weave_agent_signals.scorers.outcome import BuildOutcome, TestRunOutcome, extract_outcomes

SESSION_TESTS = "outcome.session_tests_passed"
SESSION_BUILDS = "outcome.session_no_failed_builds"
SESSION_OUTCOMES = (SESSION_TESTS, SESSION_BUILDS)
MEASURE_TURNS = "measure.turns"
MEASURE_TOKENS = "measure.tokens"
MEASURE_DURATION = "measure.duration_s"


def _tool_calls(session: SessionView) -> list[ToolSpan]:
    """Every tool call in turn order, each turn's own and its subagents' sorted by start time.
    A call without a start time sorts as the read time, so it lands last in its turn."""
    calls: list[ToolSpan] = []
    for turn in session.turns:
        found = turn.tool_calls + [tc for sub in turn.subagents for tc in sub.tool_calls]
        calls += sorted(found, key=lambda tc: tc.started_at)
    return calls


def score_session_outcomes(session: SessionView) -> list[Score]:
    """Whether the last test run passed, and whether no build failed. A Trajectory that ran no
    test, or no build, gets no Score for it, as a turn does."""
    outcomes = extract_outcomes(_tool_calls(session))
    scores = []
    tests = [o for o in outcomes if isinstance(o, TestRunOutcome)]
    if tests:
        last, failed = tests[-1], sum(not t.success for t in tests)
        state = "passed" if last.success else "failed"
        scores.append(
            Score(
                scorer=SESSION_TESTS,
                value=last.success,
                tags=["tests_passed" if last.success else "tests_failed"],
                confidence=1.0,
                metadata={"last_run": last.to_dict(), "runs": len(tests), "failed_runs": failed},
                granularity="session",
                reason=f"last of {len(tests)} test runs ({last.framework}) {state}",
            )
        )
    builds = [o for o in outcomes if isinstance(o, BuildOutcome)]
    if builds:
        failed_builds = [b for b in builds if not b.success]
        scores.append(
            Score(
                scorer=SESSION_BUILDS,
                value=not failed_builds,
                tags=["build_failure"] if failed_builds else ["builds_passed"],
                confidence=1.0,
                metadata={"runs": len(builds), "failed": [b.to_dict() for b in failed_builds]},
                granularity="session",
                reason=f"{len(failed_builds)} of {len(builds)} builds failed",
            )
        )
    return scores


def _measure(name: str, value: float, reason: str, **metadata) -> Score:
    return Score(name, float(value), [], metadata, "session", confidence=1.0, reason=reason)


def score_session_measures(session: SessionView) -> list[Score]:
    """How much the Trajectory took. These are measurements, not ratings: none is better high.

    - turns: the Trajectory's own turns.
    - tokens: input plus output tokens of its own turns and of every subagent below it, so a
      session's count is everything it spent. Cache reads aren't counted.
    - duration: seconds its own turns ran, each from its first Case's start to its last Case's
      end, summed, so the gaps between turns, when the user was reading or typing, don't
      count. Turns without timestamps are left out, and with none timed there's no Score."""
    own = session.total_tokens
    below = sum(sub.tokens for turn in session.turns for sub in turn.subagents)
    scores = [
        _measure(MEASURE_TURNS, len(session.turns), f"{len(session.turns)} turns"),
        _measure(
            MEASURE_TOKENS,
            own + below,
            f"{own + below} tokens ({below} in subagents)",
            own_tokens=own,
            subagent_tokens=below,
        ),
    ]
    timed = [t for t in session.turns if t.ended_at is not None]
    if timed:
        seconds = sum((t.ended_at - t.started_at).total_seconds() for t in timed)
        scores.append(
            _measure(
                MEASURE_DURATION,
                round(seconds, 3),
                f"{seconds:.0f} s across {len(timed)} timed turns",
                timed_turns=len(timed),
            )
        )
    return scores

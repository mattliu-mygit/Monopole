from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

import pytest

from weave_agent_signals.catalogs import build_rubric_catalog
from weave_agent_signals.judges.plan import build_judging_plan
from weave_agent_signals.judges.rubrics import RUBRICS
from weave_agent_signals.models import SessionView, ToolSpan, TurnSpan
from weave_agent_signals.run_config import RubricDescriptor

_CATALOG = build_rubric_catalog()
_DEFAULT_RUBRICS = _CATALOG.rubrics


def _ts(minute: int = 0) -> datetime:
    return datetime(2026, 7, 14, 12, minute, tzinfo=timezone.utc)


def _tool(
    name: str,
    arguments: str,
    result: str = "",
    *,
    status: str = "OK",
) -> ToolSpan:
    return ToolSpan(
        span_id=f"span-{name}-{arguments}",
        tool_name=name,
        arguments=arguments,
        result=result,
        status_code=status,
        started_at=_ts(),
        ended_at=_ts(1),
    )


def _turn(
    trace_id: str,
    minute: int,
    *,
    tools: list[ToolSpan] | None = None,
    errors: int = 0,
    steering: int = 0,
    user_input: str | None = None,
    assistant_output: str | None = "Completed.",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> TurnSpan:
    return TurnSpan(
        trace_id=trace_id,
        conversation_id="session-1",
        started_at=_ts(minute),
        ended_at=_ts(minute + 1),
        model="claude-opus-4",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=0,
        status_code="OK",
        config_version="cfg",
        git_branch="main",
        effort_level="high",
        session_id="session-1",
        steering_count=steering,
        denial_count=0,
        tool_error_count=errors,
        events=[],
        tool_calls=tools or [],
        chat_spans=[],
        subagents=[],
        user_input=user_input,
        assistant_output=assistant_output,
    )


def _session(turns: list[TurnSpan]) -> SessionView:
    return SessionView(
        conversation_id="session-1",
        turns=turns,
        config_version="cfg",
        git_branch="main",
    )


def _rubrics(*rubric_ids: str) -> tuple[RubricDescriptor, ...]:
    return tuple(_CATALOG.rubric(rubric_id) for rubric_id in rubric_ids)


def _plan(
    sessions: Sequence[SessionView],
    *,
    rubrics: Sequence[RubricDescriptor] = _DEFAULT_RUBRICS,
    review_depth: str = "primary",
    judge_count: int = 1,
    max_episodes_per_session: int = 8,
):
    return build_judging_plan(
        sessions,
        cohort_id="cohort-1",
        rubrics=rubrics,
        review_depth=review_depth,
        judge_count=judge_count,
        max_episodes_per_session=max_episodes_per_session,
    )


def _records(episode: dict) -> dict[str, dict]:
    return {record["id"]: record for record in episode["rubrics"]}


def test_active_turn_rubrics_are_the_nonoverlapping_dimensions() -> None:
    assert set(RUBRICS) == {
        "judge.verification",
        "judge.error_recovery",
        "judge.tool_choice",
        "judge.state_consistency",
    }


def test_plan_selects_salient_episodes_and_records_one_row_per_rubric() -> None:
    turns = [
        _turn("read", 0, tools=[_tool("Read", '{"path":"app.py"}')]),
        _turn(
            "failed-edit",
            2,
            tools=[_tool("Edit", '{"path":"app.py"}', "permission denied", status="ERROR")],
            errors=1,
        ),
        _turn(
            "recovery",
            4,
            tools=[_tool("Edit", '{"path":"app.py"}', "updated")],
        ),
        _turn(
            "verified",
            6,
            tools=[_tool("Bash", '{"command":"pytest -q"}', "12 passed")],
        ),
        _turn("terminal", 8, user_input="Anything else?"),
    ]

    plan = _plan([_session(turns)])
    session_plan = plan["sessions"][0]
    episodes = {episode["trace_id"]: episode for episode in session_plan["selected_episodes"]}

    assert set(episodes) == {"failed-edit", "recovery", "verified", "terminal"}
    assert "tool_error" in episodes["failed-edit"]["selection_reasons"]
    failed_edit = _records(episodes["failed-edit"])
    assert failed_edit["judge.error_recovery"]["applicability"] == "not_applicable"
    assert failed_edit["judge.error_recovery"]["minimum_reviewer_attempts"] == 0
    assert failed_edit["judge.error_recovery"]["maximum_reviewer_attempts"] == 0
    recovery = _records(episodes["recovery"])
    assert recovery["judge.error_recovery"]["applicability"] == "applicable"
    assert episodes["recovery"]["evidence_trace_ids"] == ["failed-edit", "recovery"]
    assert _records(episodes["verified"])["judge.verification"]["applicability"] == ("applicable")
    assert _records(episodes["terminal"])["judge.tool_choice"]["applicability"] == (
        "not_applicable"
    )
    assert "applicability" not in episodes["terminal"]
    assert "applicable_rubrics" not in episodes["terminal"]
    assert "not_applicable_rubrics" not in episodes["terminal"]
    assert session_plan["omitted_turn_count"] == 1


def test_plan_records_exact_pinned_descriptor_in_each_rubric_row() -> None:
    descriptor = _rubrics("judge.tool_choice")[0]
    turn = _turn("tool", 0, tools=[_tool("Read", '{"path":"app.py"}')])

    plan = _plan([_session([turn])], rubrics=(descriptor,))

    assert plan["requested_rubrics"] == [descriptor.model_dump(mode="json")]
    assert plan["sessions"][0]["selected_episodes"][0]["rubrics"] == [
        {
            **descriptor.model_dump(mode="json"),
            "applicability": "applicable",
            "minimum_reviewer_attempts": 1,
            "maximum_reviewer_attempts": 1,
        }
    ]


@pytest.mark.parametrize(
    (
        "review_depth",
        "judge_count",
        "minimum_attempts",
        "maximum_attempts",
    ),
    [
        ("primary", 1, 2, 2),
        ("selective", 2, 2, 4),
        ("selective", 3, 2, 6),
        ("full_panel", 3, 6, 6),
    ],
)
def test_plan_reports_exact_rubric_and_reviewer_attempt_totals(
    review_depth: str,
    judge_count: int,
    minimum_attempts: int,
    maximum_attempts: int,
) -> None:
    rubrics = _rubrics("judge.tool_choice", "judge.session_outcome")
    turn = _turn("tool", 0, tools=[_tool("Read", '{"path":"app.py"}')])

    plan = _plan(
        [_session([turn])],
        rubrics=rubrics,
        review_depth=review_depth,
        judge_count=judge_count,
    )

    assert plan["review_depth"] == review_depth
    assert plan["judge_count"] == judge_count
    assert plan["totals"] == {
        "turns_considered": 1,
        "episodes_selected": 1,
        "planned_episode_rubrics": 1,
        "planned_session_rubrics": 1,
        "planned_rubrics": 2,
        "minimum_episode_reviewer_attempts": minimum_attempts // 2,
        "maximum_episode_reviewer_attempts": maximum_attempts // 2,
        "minimum_session_reviewer_attempts": minimum_attempts // 2,
        "maximum_session_reviewer_attempts": maximum_attempts // 2,
        "minimum_reviewer_attempts": minimum_attempts,
        "maximum_reviewer_attempts": maximum_attempts,
    }


def test_not_applicable_episode_rubric_has_zero_attempt_bounds() -> None:
    rubrics = _rubrics("judge.tool_choice", "judge.session_outcome")
    turn = _turn("terminal", 0, user_input="Explain only")

    plan = _plan(
        [_session([turn])],
        rubrics=rubrics,
        review_depth="selective",
        judge_count=3,
    )
    episode_record = plan["sessions"][0]["selected_episodes"][0]["rubrics"][0]

    assert episode_record["applicability"] == "not_applicable"
    assert episode_record["minimum_reviewer_attempts"] == 0
    assert episode_record["maximum_reviewer_attempts"] == 0
    assert plan["totals"]["planned_episode_rubrics"] == 0
    assert plan["totals"]["planned_session_rubrics"] == 1
    assert plan["totals"]["minimum_reviewer_attempts"] == 1
    assert plan["totals"]["maximum_reviewer_attempts"] == 3


def test_plan_is_bounded_but_always_keeps_terminal_episode() -> None:
    turns = [
        _turn(
            f"error-{index}",
            index,
            tools=[_tool("Bash", f'{{"command":"bad-{index}"}}', "failed", status="ERROR")],
            errors=1,
        )
        for index in range(12)
    ]

    plan = _plan([_session(turns)], max_episodes_per_session=4)
    selected = plan["sessions"][0]["selected_episodes"]

    assert len(selected) == 4
    assert selected[-1]["trace_id"] == "error-11"
    assert "terminal" in selected[-1]["selection_reasons"]


def test_plan_is_content_stable_and_independent_of_session_order() -> None:
    first = _session([_turn("first-turn", 0, user_input="first")])
    first.conversation_id = "session-b"
    first.turns[0].conversation_id = "session-b"
    second = _session([_turn("second-turn", 0, user_input="second")])
    second.conversation_id = "session-a"
    second.turns[0].conversation_id = "session-a"

    forward = _plan([first, second])
    reverse = _plan([second, first])

    assert forward == reverse
    assert forward["plan_id"].startswith("sha256:")


def test_plan_is_independent_of_equal_timestamp_turn_order() -> None:
    first = _turn("turn-b", 0, user_input="second by trace ID")
    second = _turn("turn-a", 0, user_input="first by trace ID")

    forward = _plan([_session([first, second])])
    reverse = _plan([_session([second, first])])

    assert forward == reverse
    assert forward["sessions"][0]["selected_episodes"][-1]["trace_id"] == "turn-b"


def test_same_turn_error_recovery_requires_work_after_the_failure() -> None:
    rubric = _rubrics("judge.error_recovery")
    recovered = _turn(
        "recover-in-turn",
        0,
        tools=[
            _tool("Bash", '{"command":"pytest"}', "failed", status="ERROR"),
            _tool("Edit", '{"path":"app.py"}', "updated"),
            _tool("Bash", '{"command":"pytest"}', "passed"),
        ],
        errors=1,
    )
    failed_last = _turn(
        "failed-last",
        0,
        tools=[
            _tool("Read", '{"path":"app.py"}', "contents"),
            _tool("Bash", '{"command":"pytest"}', "failed", status="ERROR"),
        ],
        errors=1,
    )

    recovered_plan = _plan([_session([recovered])], rubrics=rubric)
    failed_plan = _plan([_session([failed_last])], rubrics=rubric)

    assert (
        recovered_plan["sessions"][0]["selected_episodes"][0]["rubrics"][0]["applicability"]
        == "applicable"
    )
    assert (
        failed_plan["sessions"][0]["selected_episodes"][0]["rubrics"][0]["applicability"]
        == "not_applicable"
    )


def test_verification_evidence_spans_latest_modification_through_verification() -> None:
    turns = [
        _turn("edit", 0, tools=[_tool("Edit", '{"path":"app.py"}', "updated")]),
        _turn("inspect", 2, tools=[_tool("Read", '{"path":"app.py"}', "contents")]),
        _turn("verify", 4, tools=[_tool("Bash", '{"command":"pytest"}', "passed")]),
    ]

    plan = _plan([_session(turns)], rubrics=_rubrics("judge.verification"))
    episodes = {
        episode["trace_id"]: episode for episode in plan["sessions"][0]["selected_episodes"]
    }

    assert episodes["verify"]["evidence_trace_ids"] == ["edit", "inspect", "verify"]


def test_verification_is_skipped_when_assistant_output_was_not_captured() -> None:
    turn = _turn(
        "bare-verification",
        0,
        tools=[
            _tool("Edit", '{"path":"app.py"}', "updated"),
            _tool("Bash", '{"command":"pytest"}', "passed"),
        ],
        assistant_output=None,
    )

    plan = _plan([_session([turn])], rubrics=_rubrics("judge.verification"))
    record = plan["sessions"][0]["selected_episodes"][0]["rubrics"][0]

    assert record["applicability"] == "not_applicable"
    assert record["minimum_reviewer_attempts"] == 0
    assert record["maximum_reviewer_attempts"] == 0
    assert record["skip_reason"] == (
        "Assistant output was not captured, so there was no completion or correctness claim "
        "to verify."
    )


def test_state_consistency_is_skipped_without_captured_prior_state() -> None:
    prior = _turn(
        "bare-prior-turn",
        0,
        assistant_output=None,
        input_tokens=0,
        output_tokens=0,
    )
    current = _turn("current-turn", 2, user_input="Continue")

    plan = _plan(
        [_session([prior, current])],
        rubrics=_rubrics("judge.state_consistency"),
    )
    episodes = {
        episode["trace_id"]: episode for episode in plan["sessions"][0]["selected_episodes"]
    }
    record = episodes["current-turn"]["rubrics"][0]

    assert record["applicability"] == "not_applicable"
    assert record["minimum_reviewer_attempts"] == 0
    assert record["maximum_reviewer_attempts"] == 0
    assert record["skip_reason"] == (
        "The prior turn contained no captured assistant output, tool activity, or model "
        "tokens to establish prior state."
    )


@pytest.mark.parametrize(
    ("review_depth", "judge_count"),
    [
        ("primary", 2),
        ("selective", 1),
        ("selective", 4),
        ("full_panel", 2),
    ],
)
def test_plan_rejects_inconsistent_review_shape(review_depth: str, judge_count: int) -> None:
    with pytest.raises(ValueError, match="judge_count"):
        _plan(
            [_session([_turn("terminal", 0, user_input="done")])],
            review_depth=review_depth,
            judge_count=judge_count,
        )

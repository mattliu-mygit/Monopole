import pytest

from weave_agent_signals.judges.inference import InferenceCancelled, JudgeResponse
from weave_agent_signals.judges.rubrics import VERIFICATION_DISCIPLINE
from weave_agent_signals.run_config import ModelDescriptor, PositionedJudge
from weave_agent_signals.runs.challenges.contracts import (
    ArmResult,
    TaskMaterialPlan,
    TaskPreflightError,
)
from weave_agent_signals.runs.challenges.environment import (
    RuntimeFile,
    capture_execution_environment,
)
from weave_agent_signals.runs.challenges.service import PreparedPair, run_challenge
from weave_agent_signals.runs.challenges.workspace import (
    ArmSnapshots,
    WorkspaceFile,
    WorkspaceSnapshot,
)


class _Client:
    backend = "cli"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        parsed, model = next(self.responses)
        return parsed, JudgeResponse(
            content="{}",
            model=model,
            usage={"input_tokens": 10, "output_tokens": 5},
        )


class _Runner:
    def __init__(self, results, *, preflight_error: Exception | None = None):
        self.results = results
        self.preflight_error = preflight_error
        self.preflight_calls = []
        self.calls = []

    def preflight(self, **kwargs):
        self.preflight_calls.append(kwargs)
        if self.preflight_error is not None:
            raise self.preflight_error

    def run_pair(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


class _CancelledClient:
    backend = "cli"

    def chat_json(self, **_kwargs):
        raise InferenceCancelled("cancelled")


def _descriptor(model: str, *, position: int | None = None):
    values = dict(
        id=model,
        label=model,
        provider="codex" if model.startswith("gpt") else "claude",
        provider_model=model,
        family="openai" if model.startswith("gpt") else "anthropic",
        supported_roles=("judge",) if position else ("proposal_evaluator",),
    )
    return PositionedJudge(**values, position=position) if position else ModelDescriptor(**values)


def _environment():
    return capture_execution_environment(
        cohort={
            "turns": [
                {
                    "model": "gpt-5.6-sol",
                    "model_family": "openai",
                    "effort_level": "high",
                }
            ]
        },
        workspace_digest="sha256:workspace",
        image="runtime",
        image_digest=f"sha256:{'a' * 64}",
        timeout_seconds=60,
        runtime_env={},
        version_reader=lambda _argv: "codex-cli 0.144.1",
    )


def _arm(arm: str, *, infrastructure: bool = False) -> ArmResult:
    if infrastructure:
        return ArmResult(
            arm=arm,
            status="infrastructure_failed",
            exit_code=None,
            duration_seconds=1,
            initial_workspace_digest=f"sha256:{arm}-initial",
            final_workspace_digest=None,
            transcript="boot failed",
            transcript_digest=f"sha256:{arm}-transcript",
            artifact_digests={},
            infrastructure_error="boot failed",
        )
    return ArmResult(
        arm=arm,
        status="exited",
        exit_code=0,
        duration_seconds=10,
        initial_workspace_digest=f"sha256:{arm}-initial",
        final_workspace_digest=f"sha256:{arm}-final",
        transcript=f"{arm} completed",
        transcript_digest=f"sha256:{arm}-transcript",
        artifact_digests={},
    )


def _material_response():
    return (
        {
            "setup_mode": "agent_bootstrap",
            "materials": [],
        },
        "claude-sonnet-5",
    )


def _author_response(*, required_files=("README.md",)):
    return (
        {
            "prompt": "Fix the parser.",
            "goal": "Parser tests pass.",
            "judging_criteria": ["Parser tests pass."],
            "start_checks": ["The workspace is available."],
            "required_files": list(required_files),
            "required_executables": ["python"],
            "requires_git_metadata": False,
        },
        "claude-sonnet-5",
    )


def _author_responses():
    return [_material_response(), _author_response()]


def _judge_response(winner: str, model: str):
    scores = {
        "candidate": (1.0, 0.5),
        "baseline": (0.5, 1.0),
        "tie": (0.75, 0.75),
    }[winner]
    presented_winner = {"candidate": "arm-1", "baseline": "arm-2", "tie": "tie"}[winner]
    return (
        {
            "task_valid": True,
            "task_invalid_reason": None,
            "winner": presented_winner,
            "rationale": f"{winner} verdict",
            "rubrics": [
                {
                    "rubric_id": "judge.verification",
                    "arm_1_score": scores[0],
                    "arm_2_score": scores[1],
                    "winner": presented_winner,
                    "rationale": "comparison",
                }
            ],
        },
        model,
    )


def test_challenge_runs_ordered_user_panel_and_requires_strict_majority() -> None:
    author_client = _Client(_author_responses())
    first_judge_client = _Client([_judge_response("candidate", "gpt-5.6-sol")])
    second_judge_client = _Client([_judge_response("baseline", "claude-sonnet-5")])
    runner = _Runner((_arm("baseline"), _arm("candidate")))
    snapshot = WorkspaceSnapshot((WorkspaceFile("README.md", b"same"),))

    result = run_challenge(
        candidate_id="candidate-1",
        coaching_digest="verification was weak",
        authoring_snapshot=snapshot,
        baseline_snapshot=snapshot,
        candidate_snapshot=snapshot,
        environment=_environment(),
        author=_descriptor("claude-sonnet-5"),
        judges=(
            _descriptor("gpt-5.6-sol", position=1),
            _descriptor("claude-sonnet-5", position=2),
        ),
        rubrics=(VERIFICATION_DISCIPLINE,),
        author_client=author_client,
        judge_clients={
            "gpt-5.6-sol": first_judge_client,
            "claude-sonnet-5": second_judge_client,
        },
        runner=runner,
        baseline_runtime_files=(RuntimeFile("/root/.codex/AGENTS.md", b"baseline instructions"),),
        candidate_runtime_files=(RuntimeFile("/root/.codex/AGENTS.md", b"candidate instructions"),),
    )

    assert result.status == "complete"
    assert result.winner == "tie"
    assert result.recommendation() is None
    assert [verdict.position for verdict in result.judges] == [1, 2]
    assert len(first_judge_client.calls) == 1
    assert len(second_judge_client.calls) == 1
    assert len(runner.calls) == 1
    assert runner.calls[0]["baseline_runtime_files"][0].content == b"baseline instructions"
    assert runner.calls[0]["candidate_runtime_files"][0].content == b"candidate instructions"


def test_challenge_propagates_author_inference_cancellation() -> None:
    snapshot = WorkspaceSnapshot((WorkspaceFile("README.md", b"same"),))

    with pytest.raises(InferenceCancelled, match="cancelled"):
        run_challenge(
            candidate_id="candidate-1",
            coaching_digest="verification was weak",
            authoring_snapshot=snapshot,
            baseline_snapshot=snapshot,
            candidate_snapshot=snapshot,
            environment=_environment(),
            author=_descriptor("claude-sonnet-5"),
            judges=(_descriptor("gpt-5.6-sol", position=1),),
            rubrics=(VERIFICATION_DISCIPLINE,),
            author_client=_CancelledClient(),
            judge_clients={},
            runner=_Runner((_arm("baseline"), _arm("candidate"))),
        )


def test_challenge_checks_cancellation_before_task_authoring() -> None:
    snapshot = WorkspaceSnapshot((WorkspaceFile("README.md", b"same"),))
    author_client = _Client(_author_responses())

    with pytest.raises(InferenceCancelled, match="cancelled"):
        run_challenge(
            candidate_id="candidate-1",
            coaching_digest="verification was weak",
            authoring_snapshot=snapshot,
            baseline_snapshot=snapshot,
            candidate_snapshot=snapshot,
            environment=_environment(),
            author=_descriptor("claude-sonnet-5"),
            judges=(_descriptor("gpt-5.6-sol", position=1),),
            rubrics=(VERIFICATION_DISCIPLINE,),
            author_client=author_client,
            judge_clients={},
            runner=_Runner((_arm("baseline"), _arm("candidate"))),
            cancel_requested=lambda: True,
        )

    assert author_client.calls == []


def test_challenge_checks_cancellation_after_task_preparation() -> None:
    snapshot = WorkspaceSnapshot((WorkspaceFile("README.md", b"same"),))
    runner = _Runner((_arm("baseline"), _arm("candidate")))
    cancelled = False

    def prepare(_plan: TaskMaterialPlan) -> PreparedPair:
        nonlocal cancelled
        cancelled = True
        return PreparedPair(
            authoring=snapshot,
            arms=ArmSnapshots(snapshot, snapshot, snapshot),
            environment=_environment(),
        )

    with pytest.raises(InferenceCancelled, match="cancelled"):
        run_challenge(
            candidate_id="candidate-1",
            coaching_digest="verification was weak",
            authoring_snapshot=snapshot,
            baseline_snapshot=snapshot,
            candidate_snapshot=snapshot,
            environment=_environment(),
            author=_descriptor("claude-sonnet-5"),
            judges=(_descriptor("gpt-5.6-sol", position=1),),
            rubrics=(VERIFICATION_DISCIPLINE,),
            author_client=_Client(_author_responses()),
            judge_clients={},
            runner=runner,
            prepare_pair=prepare,
            cancel_requested=lambda: cancelled,
        )

    assert runner.calls == []


def test_challenge_propagates_judge_inference_cancellation() -> None:
    snapshot = WorkspaceSnapshot((WorkspaceFile("README.md", b"same"),))

    with pytest.raises(InferenceCancelled, match="cancelled"):
        run_challenge(
            candidate_id="candidate-1",
            coaching_digest="verification was weak",
            authoring_snapshot=snapshot,
            baseline_snapshot=snapshot,
            candidate_snapshot=snapshot,
            environment=_environment(),
            author=_descriptor("claude-sonnet-5"),
            judges=(_descriptor("gpt-5.6-sol", position=1),),
            rubrics=(VERIFICATION_DISCIPLINE,),
            author_client=_Client(_author_responses()),
            judge_clients={"gpt-5.6-sol": _CancelledClient()},
            runner=_Runner((_arm("baseline"), _arm("candidate"))),
        )


def test_challenge_prepares_one_common_workspace_after_authoring() -> None:
    author_client = _Client(_author_responses())
    judge_client = _Client([_judge_response("tie", "gpt-5.6-sol")])
    runner = _Runner((_arm("baseline"), _arm("candidate")))
    seed = WorkspaceSnapshot((WorkspaceFile("README.md", b"seed"),))
    prepared = WorkspaceSnapshot(
        (
            WorkspaceFile("README.md", b"seed"),
            WorkspaceFile("task/app.py", b"prepared"),
        )
    )
    calls: list[TaskMaterialPlan] = []

    def prepare(plan: TaskMaterialPlan) -> PreparedPair:
        calls.append(plan)
        return PreparedPair(
            authoring=prepared,
            arms=ArmSnapshots(prepared, prepared, prepared),
            environment=capture_execution_environment(
                cohort={
                    "turns": [
                        {
                            "model": "gpt-5.6-sol",
                            "model_family": "openai",
                            "effort_level": "high",
                        }
                    ]
                },
                workspace_digest=prepared.digest,
                image="runtime",
                image_digest=f"sha256:{'a' * 64}",
                timeout_seconds=60,
                runtime_env={},
                version_reader=lambda _argv: "codex-cli 0.144.1",
            ),
        )

    result = run_challenge(
        candidate_id="candidate-1",
        coaching_digest="verification was weak",
        authoring_snapshot=seed,
        baseline_snapshot=seed,
        candidate_snapshot=seed,
        environment=_environment(),
        author=_descriptor("claude-sonnet-5"),
        judges=(_descriptor("gpt-5.6-sol", position=1),),
        rubrics=(VERIFICATION_DISCIPLINE,),
        author_client=author_client,
        judge_clients={"gpt-5.6-sol": judge_client},
        runner=runner,
        prepare_pair=prepare,
    )

    assert len(calls) == 1
    assert calls[0].setup_mode == "agent_bootstrap"
    assert result.task.workspace_digest == prepared.digest
    assert result.execution.workspace_digest == prepared.digest
    assert runner.calls[0]["baseline"] is prepared
    assert runner.calls[0]["candidate"] is prepared


def test_challenge_capability_preflight_failure_stops_before_both_arms() -> None:
    snapshot = WorkspaceSnapshot((WorkspaceFile("README.md", b"same"),))
    runner = _Runner(
        (_arm("baseline"), _arm("candidate")),
        preflight_error=TaskPreflightError("required sandbox executable is unavailable: python"),
    )

    result = run_challenge(
        candidate_id="candidate-1",
        coaching_digest="verification was weak",
        authoring_snapshot=snapshot,
        baseline_snapshot=snapshot,
        candidate_snapshot=snapshot,
        environment=_environment(),
        author=_descriptor("claude-sonnet-5"),
        judges=(_descriptor("gpt-5.6-sol", position=1),),
        rubrics=(VERIFICATION_DISCIPLINE,),
        author_client=_Client(_author_responses()),
        judge_clients={},
        runner=runner,
    )

    assert result.status == "invalid_task"
    assert "python" in result.reason
    assert result.task is not None
    assert len(runner.preflight_calls) == 1
    assert runner.calls == []


def test_challenge_missing_required_file_stops_before_sandbox_preflight() -> None:
    snapshot = WorkspaceSnapshot((WorkspaceFile("README.md", b"same"),))
    runner = _Runner((_arm("baseline"), _arm("candidate")))

    result = run_challenge(
        candidate_id="candidate-1",
        coaching_digest="verification was weak",
        authoring_snapshot=snapshot,
        baseline_snapshot=snapshot,
        candidate_snapshot=snapshot,
        environment=_environment(),
        author=_descriptor("claude-sonnet-5"),
        judges=(_descriptor("gpt-5.6-sol", position=1),),
        rubrics=(VERIFICATION_DISCIPLINE,),
        author_client=_Client(
            [_material_response(), _author_response(required_files=("lodash.js",))]
        ),
        judge_clients={},
        runner=runner,
    )

    assert result.status == "invalid_task"
    assert "lodash.js" in result.reason
    assert result.task is not None
    assert runner.preflight_calls == []
    assert runner.calls == []


def test_challenge_preflight_infrastructure_failure_is_not_an_invalid_task() -> None:
    snapshot = WorkspaceSnapshot((WorkspaceFile("README.md", b"same"),))
    runner = _Runner(
        (_arm("baseline"), _arm("candidate")),
        preflight_error=RuntimeError("Smol machine creation failed"),
    )

    result = run_challenge(
        candidate_id="candidate-1",
        coaching_digest="verification was weak",
        authoring_snapshot=snapshot,
        baseline_snapshot=snapshot,
        candidate_snapshot=snapshot,
        environment=_environment(),
        author=_descriptor("claude-sonnet-5"),
        judges=(_descriptor("gpt-5.6-sol", position=1),),
        rubrics=(VERIFICATION_DISCIPLINE,),
        author_client=_Client(_author_responses()),
        judge_clients={},
        runner=runner,
    )

    assert result.status == "incomplete"
    assert "infrastructure" in result.reason.lower()
    assert result.task is not None
    assert runner.calls == []


def test_challenge_candidate_majority_recommends_c() -> None:
    client = _Client(_author_responses())
    judge_client = _Client(
        [
            _judge_response("candidate", "gpt-5.6-sol"),
            _judge_response("candidate", "claude-sonnet-5"),
            _judge_response("baseline", "claude-haiku-4-5"),
        ]
    )
    runner = _Runner((_arm("baseline"), _arm("candidate")))
    snapshot = WorkspaceSnapshot((WorkspaceFile("README.md", b"same"),))
    judges = (
        _descriptor("gpt-5.6-sol", position=1),
        _descriptor("claude-sonnet-5", position=2),
        _descriptor("claude-haiku-4-5", position=3),
    )

    result = run_challenge(
        candidate_id="candidate-1",
        coaching_digest="verification was weak",
        authoring_snapshot=snapshot,
        baseline_snapshot=snapshot,
        candidate_snapshot=snapshot,
        environment=_environment(),
        author=_descriptor("claude-sonnet-5"),
        judges=judges,
        rubrics=(VERIFICATION_DISCIPLINE,),
        author_client=client,
        judge_clients={judge.id: judge_client for judge in judges},
        runner=runner,
    )

    assert result.winner == "candidate"
    assert result.recommendation() == "candidate-1"


def test_infrastructure_failure_stops_before_judging() -> None:
    author_client = _Client(_author_responses())
    judge_client = _Client([])
    runner = _Runner((_arm("baseline"), _arm("candidate", infrastructure=True)))
    snapshot = WorkspaceSnapshot((WorkspaceFile("README.md", b"same"),))

    result = run_challenge(
        candidate_id="candidate-1",
        coaching_digest="verification was weak",
        authoring_snapshot=snapshot,
        baseline_snapshot=snapshot,
        candidate_snapshot=snapshot,
        environment=_environment(),
        author=_descriptor("claude-sonnet-5"),
        judges=(_descriptor("gpt-5.6-sol", position=1),),
        rubrics=(VERIFICATION_DISCIPLINE,),
        author_client=author_client,
        judge_clients={"gpt-5.6-sol": judge_client},
        runner=runner,
    )

    assert result.status == "incomplete"
    assert result.winner == "tie"
    assert result.recommendation() is None
    assert judge_client.calls == []


def test_semantically_invalid_task_is_rejected_after_blinded_judging() -> None:
    author_client = _Client(_author_responses())
    judge_client = _Client(
        [
            (
                {
                    "task_valid": False,
                    "task_invalid_reason": "The requested parser does not exist in this workspace.",
                    "winner": "tie",
                    "rationale": "Neither arm could perform a workspace-inapplicable task.",
                    "rubrics": [
                        {
                            "rubric_id": "judge.verification",
                            "arm_1_score": 0.0,
                            "arm_2_score": 0.0,
                            "winner": "tie",
                            "rationale": "The task itself was invalid.",
                        }
                    ],
                },
                "gpt-5.6-sol",
            )
        ]
    )
    runner = _Runner((_arm("baseline"), _arm("candidate")))
    snapshot = WorkspaceSnapshot((WorkspaceFile("README.md", b"same"),))

    result = run_challenge(
        candidate_id="candidate-1",
        coaching_digest="verification was weak",
        authoring_snapshot=snapshot,
        baseline_snapshot=snapshot,
        candidate_snapshot=snapshot,
        environment=_environment(),
        author=_descriptor("claude-sonnet-5"),
        judges=(_descriptor("gpt-5.6-sol", position=1),),
        rubrics=(VERIFICATION_DISCIPLINE,),
        author_client=author_client,
        judge_clients={"gpt-5.6-sol": judge_client},
        runner=runner,
    )

    assert result.status == "invalid_task"
    assert result.winner == "tie"
    assert result.task is not None
    assert result.judges[0].task_valid is False
    assert result.recommendation() is None

import pytest
from pydantic import ValidationError

from weave_agent_signals.runs import reflection_records
from weave_agent_signals.runs.challenges.contracts import (
    ArmResult,
    ArtifactChange,
    AuthoredTask,
    ChallengeResult,
    ExecutionIdentity,
    JudgeVerdict,
    NamedDigest,
    RubricVerdict,
    TaskMaterial,
    TaskMaterialPlan,
)


def _execution() -> ExecutionIdentity:
    return ExecutionIdentity(
        model="gpt-5.6-sol",
        model_family="openai",
        harness="codex",
        harness_version="codex-cli 0.144.1",
        effort="high",
        image="agent-runtime:codex-0.144.1",
        image_digest=f"sha256:{'a' * 64}",
        command=("codex", "exec", "--model", "gpt-5.6-sol", "-"),
        timeout_seconds=1800,
        network_enabled=True,
        environment=(NamedDigest(name="PATH", digest="sha256:path"),),
        runtime_files=(NamedDigest(name="/root/.codex/config.toml", digest="sha256:config"),),
        workspace_digest="sha256:workspace",
    )


def _task() -> AuthoredTask:
    return AuthoredTask(
        prompt="Fix the failing parser behavior and verify the result.",
        goal="All parser tests pass and the regression is covered.",
        setup_mode="prepared_workspace",
        materials=(
            TaskMaterial(
                url="https://github.com/example/parser.git",
                revision="a" * 40,
                destination="task",
            ),
        ),
        judging_criteria=("The parser regression is fixed.", "Relevant tests pass."),
        start_checks=("The task repository is present under task/.",),
        required_files=("task/src/parser.py",),
        required_executables=("python",),
        requires_git_metadata=False,
        workspace_digest="sha256:workspace",
        author_model="claude-sonnet-5",
        author_backend="cli",
    )


def test_material_plan_round_trips_with_content_authenticated_identity() -> None:
    plan = TaskMaterialPlan(
        setup_mode="prepared_workspace",
        materials=_task().materials,
    )

    restored = TaskMaterialPlan.from_dict(plan.to_dict())

    assert restored == plan
    assert plan.plan_id.startswith("sha256:")


def test_prepared_workspace_task_cannot_require_git_metadata() -> None:
    with pytest.raises(ValidationError, match="Git metadata"):
        AuthoredTask.model_validate(
            {
                **_task().model_dump(mode="json"),
                "task_id": "pending",
                "requires_git_metadata": True,
            }
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("required_files", ("/absolute.py",), "safe relative"),
        ("required_files", ("task/.git/config",), "Git metadata"),
        ("required_executables", ("../python",), "executable name"),
    ],
)
def test_task_preflight_requirements_are_closed(
    field: str,
    value: tuple[str, ...],
    message: str,
) -> None:
    payload = _task().model_dump(mode="json", exclude={"task_id"})
    payload[field] = value

    with pytest.raises(ValidationError, match=message):
        AuthoredTask.model_validate(payload)


def _arm(arm: str) -> ArmResult:
    return ArmResult(
        arm=arm,
        status="exited",
        exit_code=0,
        duration_seconds=12.5,
        initial_workspace_digest=f"sha256:{arm}-initial",
        final_workspace_digest=f"sha256:{arm}-final",
        transcript="Agent completed the task and ran tests.",
        transcript_digest=f"sha256:{arm}-transcript",
        artifact_digests={"src/parser.py": f"sha256:{arm}-parser"},
        artifact_changes=(
            ArtifactChange(
                path="src/parser.py",
                action="modified",
                before_digest="sha256:before",
                after_digest=f"sha256:{arm}-parser",
                diff="--- a/src/parser.py\n+++ b/src/parser.py\n@@\n-old\n+new\n",
            ),
        ),
    )


def _judge(winner: str = "candidate", position: int = 1) -> JudgeVerdict:
    return JudgeVerdict(
        position=position,
        requested_model="claude-sonnet-5",
        resolved_model="claude-sonnet-5",
        family="anthropic",
        backend="cli",
        baseline_label="arm-2",
        candidate_label="arm-1",
        winner=winner,
        rationale="The candidate achieved the goal with stronger verification.",
        rubrics=(
            RubricVerdict(
                rubric_id="judge.verification",
                baseline_score=0.5,
                candidate_score=1.0,
                winner="candidate",
                rationale="Only the candidate ran the full regression test.",
            ),
        ),
        usage={"input_tokens": 100, "output_tokens": 20},
    )


def _result(*, winner: str = "candidate", status: str = "complete") -> ChallengeResult:
    effective_winner = winner if status == "complete" else "tie"
    return ChallengeResult(
        candidate_id="candidate-1",
        status=status,
        winner=effective_winner,
        reason=None if status == "complete" else "machine setup failed",
        task=_task(),
        execution=_execution(),
        baseline=_arm("baseline"),
        candidate=_arm("candidate"),
        judges=(_judge(winner=winner),) if status == "complete" else (),
    )


def test_challenge_contract_round_trips_with_content_authenticated_ids() -> None:
    result = _result()

    restored = ChallengeResult.from_dict(result.to_dict())

    assert restored == result
    assert result.challenge_id.startswith("sha256:")
    assert result.task.task_id.startswith("sha256:")
    assert result.execution.execution_id.startswith("sha256:")
    assert result.judges[0].rubrics[0].delta == pytest.approx(0.5)
    assert result.to_dict()["judges"][0]["rubrics"][0]["delta"] == pytest.approx(0.5)
    assert result.to_dict()["baseline"]["artifact_changes"][0]["path"] == "src/parser.py"
    assert result.to_dict()["task"]["schema_version"] == "3"


def test_epoch9_challenge_task_v2_upgrades_to_current_authenticated_contract() -> None:
    value = _result().to_dict()
    value["challenge_id"] = "sha256:legacy-challenge"
    value["task"]["schema_version"] = "2"
    value["task"]["task_id"] = "sha256:legacy-task"
    value["task"].pop("required_files")
    value["task"].pop("required_executables")
    value["task"].pop("requires_git_metadata")

    upgraded = reflection_records._challenge_from_epoch9(value)

    assert isinstance(upgraded, ChallengeResult)
    assert upgraded.task is not None
    assert upgraded.task.schema_version == "3"


@pytest.mark.parametrize("winner", ["baseline", "tie"])
def test_only_a_complete_candidate_win_recommends_b(winner: str) -> None:
    assert _result(winner=winner).recommendation() is None


def test_complete_candidate_win_recommends_exact_candidate() -> None:
    assert _result().recommendation() == "candidate-1"


def test_incomplete_comparison_never_recommends_candidate() -> None:
    result = _result(status="incomplete")
    assert result.recommendation() is None


def test_arm_infrastructure_failure_cannot_have_agent_exit_evidence() -> None:
    with pytest.raises(ValidationError, match="infrastructure"):
        ArmResult(
            arm="baseline",
            status="infrastructure_failed",
            exit_code=1,
            duration_seconds=0.2,
            initial_workspace_digest="sha256:initial",
            final_workspace_digest=None,
            transcript="",
            transcript_digest="sha256:empty",
            artifact_digests={},
            infrastructure_error="microVM failed to boot",
        )


def test_timed_out_arm_is_behavioral_evidence_not_infrastructure_failure() -> None:
    arm = ArmResult(
        arm="candidate",
        status="timed_out",
        exit_code=None,
        duration_seconds=1800.0,
        initial_workspace_digest="sha256:initial",
        final_workspace_digest="sha256:final",
        transcript="Agent was still working when the budget expired.",
        transcript_digest="sha256:transcript",
        artifact_digests={},
    )

    assert arm.infrastructure_error is None
    assert arm.status == "timed_out"


def test_judges_must_be_ordered_and_contiguous() -> None:
    with pytest.raises(ValidationError, match="positions"):
        ChallengeResult(
            candidate_id="candidate-1",
            status="complete",
            winner="candidate",
            reason=None,
            task=_task(),
            execution=_execution(),
            baseline=_arm("baseline"),
            candidate=_arm("candidate"),
            judges=(_judge(position=2),),
        )


def test_complete_winner_must_match_strict_panel_majority() -> None:
    with pytest.raises(ValidationError, match="majority"):
        ChallengeResult(
            candidate_id="candidate-1",
            status="complete",
            winner="candidate",
            reason=None,
            task=_task(),
            execution=_execution(),
            baseline=_arm("baseline"),
            candidate=_arm("candidate"),
            judges=(
                _judge(winner="candidate", position=1),
                _judge(winner="baseline", position=2),
            ),
        )


def test_task_material_requires_pinned_https_git_source_and_safe_destination() -> None:
    with pytest.raises(ValidationError, match="https"):
        TaskMaterial(url="git@github.com:example/repo.git", revision="a" * 40, destination="task")

    with pytest.raises(ValidationError, match="revision"):
        TaskMaterial(url="https://github.com/example/repo.git", revision="main", destination="task")

    with pytest.raises(ValidationError, match="destination"):
        TaskMaterial(
            url="https://github.com/example/repo.git",
            revision="a" * 40,
            destination="../task",
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/repo.git",
        "https://2130706433/repo.git",
        "https://127.1/repo.git",
        "https://0x7f000001/repo.git",
        "https://example.com/repo.git",
    ],
)
def test_task_material_rejects_non_public_repository_hosts(url: str) -> None:
    with pytest.raises(ValidationError, match="supported public Git host"):
        TaskMaterial(url=url, revision="a" * 40, destination="task")


def test_task_contract_requires_complete_nonblank_task_package() -> None:
    with pytest.raises(ValidationError, match="judging_criteria"):
        _task().model_copy(update={"judging_criteria": ()}).model_validate(
            _task().model_copy(update={"judging_criteria": ()}).model_dump()
        )

    with pytest.raises(ValidationError, match="Extra inputs"):
        AuthoredTask.model_validate(
            {
                **_task().model_dump(),
                "capability_profile": "coding",
            }
        )


@pytest.mark.parametrize(
    ("field", "values"),
    [
        (
            "materials",
            tuple(
                _task().materials[0].model_copy(update={"destination": f"task-{index}"})
                for index in range(4)
            ),
        ),
        ("judging_criteria", tuple(f"criterion {index}" for index in range(9))),
        ("start_checks", tuple(f"check {index}" for index in range(9))),
    ],
)
def test_task_package_bounds_model_authored_collections(
    field: str, values: tuple[object, ...]
) -> None:
    payload = _task().model_dump(exclude={"task_id"})
    payload[field] = values

    with pytest.raises(ValidationError, match="at most"):
        AuthoredTask.model_validate(payload)

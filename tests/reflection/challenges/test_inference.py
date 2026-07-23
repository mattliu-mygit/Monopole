import json

import pytest

from weave_agent_signals.judges.inference import (
    JudgeResponse,
    json_output_contract_messages,
)
from weave_agent_signals.judges.rubrics import TOOL_CHOICE, VERIFICATION_DISCIPLINE
from weave_agent_signals.judges.tokens import count_tokens
from weave_agent_signals.run_config import ModelDescriptor, PositionedJudge
from weave_agent_signals.runs.challenges.contracts import (
    ArmResult,
    ArtifactChange,
    AuthoredTask,
    TaskMaterialPlan,
)
from weave_agent_signals.runs.challenges.inference import (
    _MATERIAL_SYSTEM,
    MATERIAL_PLAN_SCHEMA,
    TASK_SCHEMA,
    _input_tokens,
    _judge_schema,
    author_task,
    judge_pair,
    plan_task_materials,
)
from weave_agent_signals.runs.challenges.workspace import WorkspaceFile, WorkspaceSnapshot


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
            usage={"input_tokens": 20, "output_tokens": 10},
        )


def _author(*, max_input_tokens: int = 128_000) -> ModelDescriptor:
    return ModelDescriptor(
        id="claude-sonnet-5",
        label="Claude Sonnet 5",
        provider="claude",
        provider_model="claude-sonnet-5",
        family="anthropic",
        supported_roles=("proposal_evaluator",),
        max_input_tokens=max_input_tokens,
    )


def _judge(*, max_input_tokens: int = 128_000) -> PositionedJudge:
    return PositionedJudge(
        id="gpt-5.6-sol",
        label="GPT 5.6 Sol",
        provider="codex",
        provider_model="gpt-5.6-sol",
        family="openai",
        supported_roles=("judge",),
        position=1,
        max_input_tokens=max_input_tokens,
    )


def _arm(arm: str, transcript: str) -> ArmResult:
    return ArmResult(
        arm=arm,
        status="exited",
        exit_code=0,
        duration_seconds=10,
        initial_workspace_digest=f"sha256:{arm}-initial",
        final_workspace_digest=f"sha256:{arm}-final",
        transcript=transcript,
        transcript_digest=f"sha256:{arm}-transcript",
        artifact_digests={"result.txt": f"sha256:{arm}-result"},
    )


def test_challenge_schemas_include_canonical_examples() -> None:
    assert [example["setup_mode"] for example in MATERIAL_PLAN_SCHEMA.examples] == [
        "prepared_workspace",
        "agent_bootstrap",
    ]
    assert TASK_SCHEMA.examples[0]["judging_criteria"]

    schema = _judge_schema((VERIFICATION_DISCIPLINE, TOOL_CHOICE))
    assert [example["task_valid"] for example in schema.examples] == [True, False]
    assert [item["rubric_id"] for item in schema.examples[0]["rubrics"]] == [
        "judge.verification",
        "judge.tool_choice",
    ]
    assert schema.examples[1]["winner"] == "tie"
    material_properties = MATERIAL_PLAN_SCHEMA.schema["properties"]["materials"]["items"][
        "properties"
    ]
    assert "revision" not in material_properties
    material_prompt = " ".join(_MATERIAL_SYSTEM.split())
    assert "real source and tests" in material_prompt
    assert "toy, demo, example, or placeholder repositories" in material_prompt


def test_challenge_input_tokens_count_prompt_contract_and_provider_schema() -> None:
    messages = [{"role": "user", "content": "author a task"}]
    contract_messages = json_output_contract_messages(messages, TASK_SCHEMA)
    rendered = "".join(message["content"] for message in contract_messages)
    rendered += json.dumps(TASK_SCHEMA.schema, sort_keys=True)

    assert _input_tokens(messages, TASK_SCHEMA, token_counter="utf8_bytes_div_3") == count_tokens(
        rendered,
        "utf8_bytes_div_3",
    )


def test_task_author_plans_materials_then_uses_the_fetched_manifest() -> None:
    client = _Client(
        [
            (
                {
                    "setup_mode": "prepared_workspace",
                    "materials": [
                        {
                            "kind": "git_repository",
                            "url": "https://github.com/example/parser.git",
                            "destination": "task",
                        }
                    ],
                },
                "claude-sonnet-5",
            ),
            (
                {
                    "prompt": "Repair the parser regression in task/src/parser.py.",
                    "goal": "The parser regression test passes.",
                    "judging_criteria": ["The regression is fixed.", "Tests pass."],
                    "start_checks": ["The fetched parser sources are available under task/."],
                    "required_files": ["task/src/parser.py", "task/tests/test_parser.py"],
                    "required_executables": ["python"],
                    "requires_git_metadata": False,
                },
                "claude-sonnet-5",
            ),
        ]
    )

    resolved_urls: list[str] = []

    def resolve_revision(url: str) -> str:
        resolved_urls.append(url)
        return "a" * 40

    plan = plan_task_materials(
        client,
        author=_author(),
        coaching_digest="Agents often failed to verify parser fixes.",
        workspace_digest="sha256:seed",
        workspace=WorkspaceSnapshot((WorkspaceFile("README.md", b"seed workspace\n"),)),
        resolve_revision=resolve_revision,
    )
    task = author_task(
        client,
        author=_author(),
        coaching_digest="Agents often failed to verify parser fixes.",
        material_plan=plan,
        workspace_digest="sha256:prepared",
        workspace=WorkspaceSnapshot(
            (
                WorkspaceFile("task/src/parser.py", b"def parse(value):\n    return value\n"),
                WorkspaceFile("task/tests/test_parser.py", b"def test_parse():\n    assert True\n"),
            )
        ),
        initial_workspace=WorkspaceSnapshot(
            (
                WorkspaceFile("task/src/parser.py", b"def parse(value):\n    return value\n"),
                WorkspaceFile("task/tests/test_parser.py", b"def test_parse():\n    assert True\n"),
            )
        ),
    )

    assert task.prompt == "Repair the parser regression in task/src/parser.py."
    assert task.goal == "The parser regression test passes."
    assert task.setup_mode == "prepared_workspace"
    assert task.materials[0].revision == "a" * 40
    assert resolved_urls == ["https://github.com/example/parser.git"]
    assert task.judging_criteria == ("The regression is fixed.", "Tests pass.")
    assert task.required_files == ("task/src/parser.py", "task/tests/test_parser.py")
    assert task.required_executables == ("python",)
    assert task.requires_git_metadata is False
    assert len(client.calls) == 2
    serialized_messages = str([call["messages"] for call in client.calls])
    assert "baseline agents" not in serialized_messages
    assert "candidate agents" not in serialized_messages
    payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert payload["workspace_summary"]["file_count"] == 1
    authored_payload = json.loads(client.calls[1]["messages"][1]["content"])
    assert authored_payload["workspace_manifest"] == [
        "task/src/parser.py",
        "task/tests/test_parser.py",
    ]
    assert authored_payload["initial_workspace_manifest"] == authored_payload["workspace_manifest"]
    assert authored_payload["material_plan"]["plan_id"] == plan.plan_id
    assert "def parse(value)" not in serialized_messages
    assert set(client.calls[0]["response_schema"].schema["properties"]) == {
        "setup_mode",
        "materials",
    }
    assert set(client.calls[1]["response_schema"].schema["properties"]) == {
        "prompt",
        "goal",
        "judging_criteria",
        "start_checks",
        "required_files",
        "required_executables",
        "requires_git_metadata",
    }


def test_material_planner_replaces_an_inaccessible_repository() -> None:
    client = _Client(
        [
            (
                {
                    "setup_mode": "prepared_workspace",
                    "materials": [
                        {
                            "kind": "git_repository",
                            "url": "https://github.com/wandb/private-project.git",
                            "destination": "task",
                        }
                    ],
                },
                "claude-sonnet-5",
            ),
            (
                {
                    "setup_mode": "prepared_workspace",
                    "materials": [
                        {
                            "kind": "git_repository",
                            "url": "https://github.com/psf/requests.git",
                            "destination": "task",
                        }
                    ],
                },
                "claude-sonnet-5",
            ),
        ]
    )

    def resolve_revision(url: str) -> str:
        if "private-project" in url:
            raise RuntimeError("anonymous access failed")
        return "b" * 40

    plan = plan_task_materials(
        client,
        author=_author(),
        coaching_digest="Agents need a fresh analogous verification task.",
        workspace_digest="sha256:seed",
        workspace=WorkspaceSnapshot((WorkspaceFile("README.md", b"seed\n"),)),
        resolve_revision=resolve_revision,
    )

    assert len(client.calls) == 2
    assert plan.materials[0].url == "https://github.com/psf/requests.git"
    assert plan.materials[0].revision == "b" * 40
    correction_messages = client.calls[1]["messages"]
    assert "https://github.com/wandb/private-project.git" in str(correction_messages)
    assert "accessed anonymously" in str(correction_messages)


def test_material_planner_reports_rejected_urls_after_three_attempts() -> None:
    urls = [
        "https://github.com/wandb/private-one.git",
        "https://github.com/wandb/private-two.git",
        "https://github.com/wandb/private-three.git",
    ]
    client = _Client(
        [
            (
                {
                    "setup_mode": "prepared_workspace",
                    "materials": [{"kind": "git_repository", "url": url, "destination": "task"}],
                },
                "claude-sonnet-5",
            )
            for url in urls
        ]
    )

    with pytest.raises(RuntimeError, match="anonymously accessible public material") as error:
        plan_task_materials(
            client,
            author=_author(),
            coaching_digest="Agents need a fresh analogous verification task.",
            workspace_digest="sha256:seed",
            workspace=WorkspaceSnapshot((WorkspaceFile("README.md", b"seed\n"),)),
            resolve_revision=lambda _url: (_ for _ in ()).throw(
                RuntimeError("anonymous access failed")
            ),
        )

    assert len(client.calls) == 3
    assert all(url in str(error.value) for url in urls)


def test_pairwise_judge_uses_exact_configured_rubrics_and_blinded_order() -> None:
    client = _Client(
        [
            (
                {
                    "task_valid": True,
                    "task_invalid_reason": None,
                    "winner": "arm-1",
                    "rationale": "Arm 1 achieved the goal more completely.",
                    "rubrics": [
                        {
                            "rubric_id": "judge.verification",
                            "arm_1_score": 1.0,
                            "arm_2_score": 0.5,
                            "winner": "arm-1",
                            "rationale": "Arm 1 ran the relevant tests.",
                        },
                        {
                            "rubric_id": "judge.tool_choice",
                            "arm_1_score": 0.75,
                            "arm_2_score": 0.75,
                            "winner": "tie",
                            "rationale": "Both used appropriate tools.",
                        },
                    ],
                },
                "gpt-5.6-sol",
            )
        ]
    )
    task = AuthoredTask(
        prompt="Repair the parser.",
        goal="Tests pass.",
        workspace_digest="sha256:workspace",
        author_model="claude-sonnet-5",
        author_backend="cli",
    )

    verdict = judge_pair(
        client,
        judge=_judge(),
        task=task,
        baseline=_arm("baseline", "Baseline output"),
        candidate=_arm("candidate", "Candidate output"),
        rubrics=(VERIFICATION_DISCIPLINE, TOOL_CHOICE),
        presentation_order=("candidate", "baseline"),
    )

    assert verdict.winner == "candidate"
    assert verdict.task_valid is True
    assert verdict.baseline_label == "arm-2"
    assert verdict.candidate_label == "arm-1"
    assert [item.rubric_id for item in verdict.rubrics] == [
        "judge.verification",
        "judge.tool_choice",
    ]
    assert verdict.rubrics[0].baseline_score == 0.5
    assert verdict.rubrics[0].candidate_score == 1.0
    payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert payload["rubrics"][0]["instructions"] == VERIFICATION_DISCIPLINE.system_prompt
    assert payload["rubrics"][1]["instructions"] == TOOL_CHOICE.system_prompt
    assert payload["task_judging_criteria"] == ["The stated goal is achieved."]
    assert payload["task_start_checks"] == ["The supplied workspace is available."]
    assert payload["arm-1"]["transcript"] == "Candidate output"
    assert payload["arm-2"]["transcript"] == "Baseline output"
    assert "arm" not in payload["arm-1"]
    assert "arm" not in payload["arm-2"]


def test_task_author_uses_compact_workspace_summary_for_large_workspaces() -> None:
    client = _Client(
        [
            (
                {
                    "prompt": "Repair the parser.",
                    "goal": "Parser tests pass.",
                    "judging_criteria": ["Parser tests pass."],
                    "start_checks": ["The workspace is writable."],
                    "required_files": ["README.md"],
                    "required_executables": ["python"],
                    "requires_git_metadata": False,
                },
                "claude-sonnet-5",
            ),
        ]
    )
    workspace = WorkspaceSnapshot(
        (
            WorkspaceFile("README.md", b"workspace overview"),
            *(
                WorkspaceFile(f"src/package_{index:04d}/module.py", b"value = 1\n")
                for index in range(2_000)
            ),
        )
    )
    author = _author(max_input_tokens=5_000)

    author_task(
        client,
        author=author,
        coaching_digest="verification was weak",
        material_plan=TaskMaterialPlan(setup_mode="agent_bootstrap", materials=()),
        workspace_digest="sha256:workspace",
        workspace=workspace,
    )

    call = client.calls[0]
    rendered = "".join(message["content"] for message in call["messages"])
    rendered += json.dumps(call["response_schema"].schema, sort_keys=True)
    assert count_tokens(rendered, author.token_counter) <= author.max_input_tokens - 1_536
    summary = json.loads(call["messages"][1]["content"])["workspace_summary"]
    assert summary["file_count"] == len(workspace.files)
    payload = json.loads(call["messages"][1]["content"])
    assert payload["workspace_manifest_truncated"] is True
    assert len(payload["workspace_manifest"]) == 64
    assert "workspace overview" not in rendered
    assert len(rendered) < 20_000


def test_pairwise_judge_redacts_instruction_variants_and_fits_model_capacity() -> None:
    client = _Client(
        [
            (
                {
                    "task_valid": True,
                    "task_invalid_reason": None,
                    "winner": "tie",
                    "rationale": "Both arms were equivalent.",
                    "rubrics": [
                        {
                            "rubric_id": "judge.verification",
                            "arm_1_score": 0.5,
                            "arm_2_score": 0.5,
                            "winner": "tie",
                            "rationale": "Equivalent evidence.",
                        }
                    ],
                },
                "gpt-5.6-sol",
            )
        ]
    )
    secret = 'CANDIDATE "ONLY"\nINSTRUCTION CONTENT'
    escaped_secret = json.dumps(secret)[1:-1]
    baseline = _arm(
        "baseline",
        ("baseline output\n" * 20_000) + ".codex/AGENTS.md " + escaped_secret,
    )
    candidate = _arm(
        "candidate",
        ("candidate output\n" * 20_000) + ".codex/AGENTS.md " + escaped_secret,
    )
    changes = (
        ArtifactChange(
            path="/root/.codex/AGENTS.md",
            action="modified",
            before_digest="sha256:before",
            after_digest="sha256:after",
            diff=f"-{secret}\n+replacement",
        ),
    )
    baseline = baseline.model_copy(
        update={
            "artifact_changes": changes,
            "artifact_digests": {
                **baseline.artifact_digests,
                "/root/.codex/AGENTS.md": "sha256:after",
            },
        }
    )
    candidate = candidate.model_copy(
        update={
            "artifact_changes": changes,
            "artifact_digests": {
                **candidate.artifact_digests,
                "/root/.codex/AGENTS.md": "sha256:after",
            },
        }
    )
    judge = _judge(max_input_tokens=10_000)

    judge_pair(
        client,
        judge=judge,
        task=AuthoredTask(
            prompt="Repair the parser.",
            goal="Tests pass.",
            workspace_digest="sha256:workspace",
            author_model="claude-sonnet-5",
            author_backend="cli",
        ),
        baseline=baseline,
        candidate=candidate,
        rubrics=(VERIFICATION_DISCIPLINE,),
        presentation_order=("baseline", "candidate"),
        blinded_instruction_paths=("/root/.codex/AGENTS.md",),
        blinded_instruction_contents=(secret,),
    )

    call = client.calls[0]
    rendered = "".join(message["content"] for message in call["messages"])
    rendered += json.dumps(call["response_schema"].schema, sort_keys=True)
    assert secret not in rendered
    assert count_tokens(rendered, judge.token_counter) <= judge.max_input_tokens - 3_072
    payload = json.loads(call["messages"][1]["content"])
    assert "artifact_digests" not in payload["arm-1"]
    assert payload["arm-1"]["artifact_count"] == 1
    assert payload["arm-1"]["artifact_change_count"] == 0
    assert payload["arm-1"]["artifact_changes"] == []
    assert "AGENTS.md" not in rendered
    assert ".codex" not in rendered

import json

from weave_agent_signals.judges.inference import JudgeResponse
from weave_agent_signals.judges.rubrics import TOOL_CHOICE, VERIFICATION_DISCIPLINE
from weave_agent_signals.judges.tokens import count_tokens
from weave_agent_signals.run_config import ModelDescriptor, PositionedJudge
from weave_agent_signals.runs.challenges.contracts import ArmResult, ArtifactChange, AuthoredTask
from weave_agent_signals.runs.challenges.inference import author_task, judge_pair
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


def test_task_author_makes_one_complete_task_package_call_without_b_or_c_content() -> None:
    client = _Client(
        [
            (
                {
                    "prompt": "Repair the parser regression in task/.",
                    "goal": "The parser regression test passes.",
                    "setup_mode": "prepared_workspace",
                    "materials": [
                        {
                            "kind": "git_repository",
                            "url": "https://github.com/example/parser.git",
                            "revision": "a" * 40,
                            "destination": "task",
                        }
                    ],
                    "judging_criteria": ["The regression is fixed.", "Tests pass."],
                    "start_checks": ["task/ contains the pinned repository."],
                },
                "claude-sonnet-5",
            ),
        ]
    )

    task = author_task(
        client,
        author=_author(),
        coaching_digest="Agents often failed to verify parser fixes.",
        workspace_digest="sha256:workspace",
        workspace=WorkspaceSnapshot(
            (
                WorkspaceFile("src/parser.py", b"def parse(value):\n    return value\n"),
                WorkspaceFile("tests/test_parser.py", b"def test_parse():\n    assert True\n"),
            )
        ),
    )

    assert task.prompt == "Repair the parser regression in task/."
    assert task.goal == "The parser regression test passes."
    assert task.setup_mode == "prepared_workspace"
    assert task.materials[0].revision == "a" * 40
    assert task.judging_criteria == ("The regression is fixed.", "Tests pass.")
    assert task.start_checks == ("task/ contains the pinned repository.",)
    assert len(client.calls) == 1
    serialized_messages = str([call["messages"] for call in client.calls])
    assert "baseline agents" not in serialized_messages
    assert "candidate agents" not in serialized_messages
    payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert payload["workspace_summary"]["file_count"] == 2
    assert payload["workspace_summary"]["top_level_entries"] == ["src", "tests"]
    assert "def parse(value)" not in serialized_messages
    assert set(client.calls[0]["response_schema"].schema["properties"]) == {
        "prompt",
        "goal",
        "setup_mode",
        "materials",
        "judging_criteria",
        "start_checks",
    }


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
                    "setup_mode": "agent_bootstrap",
                    "materials": [],
                    "judging_criteria": ["Parser tests pass."],
                    "start_checks": ["The workspace is writable."],
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
        workspace_digest="sha256:workspace",
        workspace=workspace,
    )

    call = client.calls[0]
    rendered = "".join(message["content"] for message in call["messages"])
    rendered += json.dumps(call["response_schema"].schema, sort_keys=True)
    assert count_tokens(rendered, author.token_counter) <= author.max_input_tokens - 1_536
    summary = json.loads(call["messages"][1]["content"])["workspace_summary"]
    assert summary["file_count"] == len(workspace.files)
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

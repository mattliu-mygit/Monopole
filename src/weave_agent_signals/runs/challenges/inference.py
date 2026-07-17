"""Blinded task-author and paired-judge inference boundaries."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from weave_agent_signals.judges.families import model_family
from weave_agent_signals.judges.inference import ChatClient, JsonSchemaSpec
from weave_agent_signals.judges.rubrics import Rubric
from weave_agent_signals.judges.tokens import count_tokens
from weave_agent_signals.run_config import ModelDescriptor, PositionedJudge
from weave_agent_signals.runs.challenges.contracts import (
    ArmName,
    ArmResult,
    AuthoredTask,
    JudgeVerdict,
    RubricVerdict,
    TaskMaterial,
    Winner,
)
from weave_agent_signals.runs.challenges.workspace import WorkspaceSnapshot

_INFERENCE_SAFETY_TOKENS = 1_024
_JUDGE_MAX_TRANSCRIPT_CHARACTERS = 100_000
_JUDGE_MAX_ARTIFACT_DIFF_CHARACTERS = 50_000
_JUDGE_MAX_ARTIFACT_CHANGES = 200
_REDACTED_INSTRUCTION_EVIDENCE = "managed instruction evidence withheld from blinded judge"

TASK_SCHEMA = JsonSchemaSpec(
    name="paired_challenge_task",
    schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "prompt": {"type": "string", "minLength": 1},
            "goal": {"type": "string", "minLength": 1},
            "setup_mode": {
                "type": "string",
                "enum": ["prepared_workspace", "agent_bootstrap"],
            },
            "materials": {
                "type": "array",
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "kind": {"type": "string", "enum": ["git_repository"]},
                        "url": {
                            "type": "string",
                            "pattern": (
                                "^https://(?:github\\.com|gitlab\\.com|"
                                "bitbucket\\.org|codeberg\\.org)/"
                            ),
                        },
                        "revision": {"type": "string", "pattern": "^[0-9a-fA-F]{40}$"},
                        "destination": {"type": "string", "minLength": 1},
                    },
                    "required": ["kind", "url", "revision", "destination"],
                },
            },
            "judging_criteria": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "start_checks": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
        },
        "required": [
            "prompt",
            "goal",
            "setup_mode",
            "materials",
            "judging_criteria",
            "start_checks",
        ],
    },
)

_TASK_SYSTEM = """\
Design one fresh, realistic verification task that exercises the behavioral weaknesses in the
evaluation digest. Return a complete task package: prompt, measurable goal, setup mode, pinned
public materials, judging criteria, and starting-state checks. You do not know which instruction
variant or evaluated model will run it and must not infer or mention them.

Use prepared_workspace when public source files should be fetched once and copied identically
into both environments. Use agent_bootstrap only when obtaining or updating the sources is itself
part of the task; then state the exact fetch requirement in the prompt. Git materials must use a
public HTTPS URL, a full 40-character commit SHA, and a safe relative destination. Do not request
credentials, private sources, moving branches, arbitrary setup scripts, commits, tags, or pushes.
Supported repository hosts are github.com, gitlab.com, bitbucket.org, and codeberg.org.
"""

_PAIR_SYSTEM = """\
Compare two blinded agent runs against the same prompt and goal. Judge only the supplied traces,
exit outcomes, and bounded artifact diffs. First decide whether the task was achievable, specific,
workspace-applicable, and fair to both arms. If it was invalid, mark it invalid and select a tie.
Otherwise score every configured rubric for each arm, select the better arm for that rubric, then
return one overall arm winner or tie. Do not infer the identity of either arm or reward verbosity.
"""


def _strict_object(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} has an invalid shape")
    return value


def _input_tokens(
    messages: Sequence[Mapping[str, str]],
    schema: JsonSchemaSpec,
    *,
    token_counter: str,
) -> int:
    rendered = "".join(message["content"] for message in messages)
    rendered += json.dumps(schema.schema, sort_keys=True)
    return count_tokens(rendered, token_counter)


def _input_limit(model: ModelDescriptor, *, max_output_tokens: int) -> int:
    limit = model.max_input_tokens - max_output_tokens - _INFERENCE_SAFETY_TOKENS
    if limit <= 0:
        raise ValueError("model context is too small for paired inference")
    return limit


def _workspace_summary(workspace: WorkspaceSnapshot) -> dict[str, object]:
    extensions = Counter(PurePosixPath(item.path).suffix or "[none]" for item in workspace.files)
    return {
        "file_count": len(workspace.files),
        "total_bytes": sum(len(item.content) for item in workspace.files),
        "top_level_entries": sorted(
            {PurePosixPath(item.path).parts[0] for item in workspace.files}
        ),
        "extensions": dict(sorted(extensions.items(), key=lambda item: (-item[1], item[0]))[:20]),
    }


def author_task(
    client: ChatClient,
    *,
    author: ModelDescriptor,
    coaching_digest: str,
    workspace_digest: str,
    workspace: WorkspaceSnapshot,
) -> AuthoredTask:
    """Author one complete task package without exposing either instruction arm."""

    if "proposal_evaluator" not in author.supported_roles:
        raise ValueError("task author must support proposal evaluation")
    if not isinstance(coaching_digest, str) or not coaching_digest.strip():
        raise ValueError("task author requires a nonblank coaching digest")
    task_messages = [
        {"role": "system", "content": _TASK_SYSTEM},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "evaluation_digest": coaching_digest,
                    "seed_workspace_digest": workspace_digest,
                    "workspace_summary": _workspace_summary(workspace),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
    ]
    if _input_tokens(task_messages, TASK_SCHEMA, token_counter=author.token_counter) > _input_limit(
        author, max_output_tokens=1_024
    ):
        raise ValueError("task author input cannot fit the configured model context")

    parsed, response = client.chat_json(
        model=author.provider_model,
        messages=task_messages,
        temperature=0.0,
        max_tokens=1_024,
        response_schema=TASK_SCHEMA,
    )
    expected = {
        "prompt",
        "goal",
        "setup_mode",
        "materials",
        "judging_criteria",
        "start_checks",
    }
    value = _strict_object(parsed, expected, "task author response")
    return AuthoredTask(
        prompt=value["prompt"],
        goal=value["goal"],
        setup_mode=value["setup_mode"],
        materials=tuple(TaskMaterial.model_validate(item) for item in value["materials"]),
        judging_criteria=tuple(value["judging_criteria"]),
        start_checks=tuple(value["start_checks"]),
        workspace_digest=workspace_digest,
        author_model=response.model,
        author_backend=client.backend,
    )


def _judge_schema(rubrics: Sequence[Rubric]) -> JsonSchemaSpec:
    ids = [rubric.scorer_name for rubric in rubrics]
    return JsonSchemaSpec(
        name="paired_challenge_verdict",
        schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "task_valid": {"type": "boolean"},
                "task_invalid_reason": {"type": ["string", "null"]},
                "winner": {"type": "string", "enum": ["arm-1", "arm-2", "tie"]},
                "rationale": {"type": "string"},
                "rubrics": {
                    "type": "array",
                    "minItems": len(ids),
                    "maxItems": len(ids),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "rubric_id": {"type": "string", "enum": ids},
                            "arm_1_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "arm_2_score": {"type": "number", "minimum": 0, "maximum": 1},
                            "winner": {
                                "type": "string",
                                "enum": ["arm-1", "arm-2", "tie"],
                            },
                            "rationale": {"type": "string"},
                        },
                        "required": [
                            "rubric_id",
                            "arm_1_score",
                            "arm_2_score",
                            "winner",
                            "rationale",
                        ],
                    },
                },
            },
            "required": [
                "task_valid",
                "task_invalid_reason",
                "winner",
                "rationale",
                "rubrics",
            ],
        },
    )


def _redact_instruction_content(text: str, contents: Sequence[str]) -> str:
    values: set[str] = set()
    for content in contents:
        raw_values = [content] if content else []
        raw_values.extend(line for line in content.splitlines() if len(line.strip()) >= 8)
        for value in raw_values:
            values.add(value)
            values.add(json.dumps(value)[1:-1])
            values.add(json.dumps(value, ensure_ascii=False)[1:-1])
    for value in sorted(values, key=len, reverse=True):
        text = text.replace(value, f"[{_REDACTED_INSTRUCTION_EVIDENCE}]")
    return text


def _instruction_path_variants(paths: Sequence[str]) -> tuple[str, ...]:
    variants: set[str] = set()
    for path in paths:
        parts = tuple(part for part in PurePosixPath(path).parts if part != "/")
        variants.update("/".join(parts[index:]) for index in range(len(parts)))
    return tuple(sorted(variants))


def _blinded_arm(
    arm: ArmResult,
    *,
    transcript_characters: int,
    artifact_diff_characters: int,
    artifact_change_limit: int,
    instruction_paths: frozenset[str],
    instruction_contents: Sequence[str],
) -> dict[str, object]:
    transcript = _redact_instruction_content(
        arm.transcript,
        (*_instruction_path_variants(tuple(instruction_paths)), *instruction_contents),
    )
    remaining = artifact_diff_characters
    changes: list[dict[str, object]] = []
    public_changes = tuple(
        item for item in arm.artifact_changes if item.path not in instruction_paths
    )
    for item in public_changes[:artifact_change_limit]:
        diff = item.diff[:remaining]
        remaining -= len(diff)
        changes.append(
            {
                "path": item.path,
                "action": item.action,
                "before_digest": item.before_digest,
                "after_digest": item.after_digest,
                "diff": diff or "artifact diff budget exhausted",
            }
        )
    return {
        "status": arm.status,
        "exit_code": arm.exit_code,
        "duration_seconds": arm.duration_seconds,
        "initial_workspace_digest": arm.initial_workspace_digest,
        "final_workspace_digest": arm.final_workspace_digest,
        "transcript": transcript[-transcript_characters:],
        "transcript_digest": arm.transcript_digest,
        "artifact_count": sum(path not in instruction_paths for path in arm.artifact_digests),
        "artifact_change_count": len(public_changes),
        "artifact_changes_truncated": len(public_changes) > len(changes),
        "artifact_changes": changes,
    }


def _winner(value: object, order: tuple[ArmName, ArmName]) -> Winner:
    if value == "tie":
        return "tie"
    if value == "arm-1":
        return order[0]
    if value == "arm-2":
        return order[1]
    raise ValueError("paired judge returned an invalid winner")


def _score(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    score = float(value)
    if not 0 <= score <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return score


def judge_pair(
    client: ChatClient,
    *,
    judge: PositionedJudge,
    task: AuthoredTask,
    baseline: ArmResult,
    candidate: ArmResult,
    rubrics: Sequence[Rubric],
    presentation_order: tuple[ArmName, ArmName],
    blinded_instruction_paths: Sequence[str] = (),
    blinded_instruction_contents: Sequence[str] = (),
) -> JudgeVerdict:
    """Run one configured judge over blinded paired traces and artifacts."""

    if set(presentation_order) != {"baseline", "candidate"}:
        raise ValueError("presentation order must contain both arms exactly once")
    if not rubrics:
        raise ValueError("paired judging requires at least one rubric")
    by_arm = {"baseline": baseline, "candidate": candidate}
    rubric_payload = [
        {
            "rubric_id": rubric.scorer_name,
            "description": rubric.description,
            "instructions": rubric.system_prompt,
        }
        for rubric in rubrics
    ]
    schema = _judge_schema(rubrics)
    input_limit = _input_limit(judge, max_output_tokens=2048)
    instruction_paths = frozenset(blinded_instruction_paths)
    transcript_characters = _JUDGE_MAX_TRANSCRIPT_CHARACTERS
    artifact_diff_characters = _JUDGE_MAX_ARTIFACT_DIFF_CHARACTERS
    artifact_change_limit = _JUDGE_MAX_ARTIFACT_CHANGES
    messages: list[dict[str, str]] = []
    while True:
        payload = {
            "prompt": task.prompt,
            "goal": task.goal,
            "setup_mode": task.setup_mode,
            "materials": [item.to_dict() for item in task.materials],
            "task_judging_criteria": list(task.judging_criteria),
            "task_start_checks": list(task.start_checks),
            "rubrics": rubric_payload,
            "arm-1": _blinded_arm(
                by_arm[presentation_order[0]],
                transcript_characters=transcript_characters,
                artifact_diff_characters=artifact_diff_characters,
                artifact_change_limit=artifact_change_limit,
                instruction_paths=instruction_paths,
                instruction_contents=blinded_instruction_contents,
            ),
            "arm-2": _blinded_arm(
                by_arm[presentation_order[1]],
                transcript_characters=transcript_characters,
                artifact_diff_characters=artifact_diff_characters,
                artifact_change_limit=artifact_change_limit,
                instruction_paths=instruction_paths,
                instruction_contents=blinded_instruction_contents,
            ),
        }
        messages = [
            {"role": "system", "content": _PAIR_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(payload, sort_keys=True, separators=(",", ":")),
            },
        ]
        if _input_tokens(messages, schema, token_counter=judge.token_counter) <= input_limit:
            break
        if (
            transcript_characters <= 1_000
            and artifact_diff_characters <= 1_000
            and artifact_change_limit <= 10
        ):
            raise ValueError("paired judge evidence cannot fit the configured model context")
        transcript_characters = max(1_000, transcript_characters // 2)
        artifact_diff_characters = max(1_000, artifact_diff_characters // 2)
        artifact_change_limit = max(10, artifact_change_limit // 2)

    parsed, response = client.chat_json(
        model=judge.provider_model,
        messages=messages,
        temperature=0.0,
        max_tokens=2048,
        response_schema=schema,
    )
    value = _strict_object(
        parsed,
        {"task_valid", "task_invalid_reason", "winner", "rationale", "rubrics"},
        "paired verdict",
    )
    raw_rubrics = value["rubrics"]
    if not isinstance(raw_rubrics, list):
        raise ValueError("paired verdict rubrics must be a list")
    expected_ids = [rubric.scorer_name for rubric in rubrics]
    returned_ids = [
        item.get("rubric_id") if isinstance(item, Mapping) else None for item in raw_rubrics
    ]
    if returned_ids != expected_ids:
        raise ValueError("paired verdict rubric order does not match configured rubrics")
    labels = {
        presentation_order[0]: "arm-1",
        presentation_order[1]: "arm-2",
    }
    verdicts: list[RubricVerdict] = []
    for item in raw_rubrics:
        raw = _strict_object(
            item,
            {"rubric_id", "arm_1_score", "arm_2_score", "winner", "rationale"},
            "paired rubric verdict",
        )
        scores = {
            presentation_order[0]: _score(raw["arm_1_score"], "arm_1_score"),
            presentation_order[1]: _score(raw["arm_2_score"], "arm_2_score"),
        }
        verdicts.append(
            RubricVerdict(
                rubric_id=raw["rubric_id"],
                baseline_score=scores["baseline"],
                candidate_score=scores["candidate"],
                winner=_winner(raw["winner"], presentation_order),
                rationale=raw["rationale"],
            )
        )
    usage = {
        key: int(item)
        for key, item in response.usage.items()
        if isinstance(key, str) and type(item) is int and item >= 0
    }
    return JudgeVerdict(
        position=judge.position,
        requested_model=judge.id,
        resolved_model=response.model,
        family=model_family(response.model),
        backend=client.backend,
        baseline_label=labels["baseline"],
        candidate_label=labels["candidate"],
        task_valid=value["task_valid"],
        task_invalid_reason=value["task_invalid_reason"],
        winner=_winner(value["winner"], presentation_order),
        rationale=value["rationale"],
        rubrics=tuple(verdicts),
        usage=usage,
    )

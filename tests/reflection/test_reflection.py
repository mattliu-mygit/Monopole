from __future__ import annotations

import hashlib
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import pytest

import weave_agent_signals.runs.reflection as reflection_module
from weave_agent_signals.judges.inference import JudgeResponse
from weave_agent_signals.run_config import ModelDescriptor
from weave_agent_signals.runs.bundles import (
    BundleSnapshot,
    BundleValidationError,
    ScopeDescriptor,
    bundle_from_content_map,
    compare_bundles,
)
from weave_agent_signals.runs.challenges.contracts import (
    ArmResult,
    AuthoredTask,
    ChallengeResult,
    ExecutionIdentity,
    JudgeVerdict,
    NamedDigest,
    RubricVerdict,
)
from weave_agent_signals.runs.proposals import (
    REFLECTION_PROPOSAL_SCHEMA,
    CandidateProposal,
    parse_candidate_proposal,
    serialize_candidate_proposal,
)
from weave_agent_signals.runs.reflection import (
    PREDICTED_EVALUATOR_SCORE_BASIS,
    ReflectionCancelled,
    ReflectionEvaluationError,
    ReflectionResult,
    run_reflection,
)

SCOPE = ScopeDescriptor("file", "/project", ("CLAUDE.md", ".claude/**/*.md"))
WRITER = ModelDescriptor(
    id="writer-requested",
    label="Writer",
    provider="claude",
    provider_model="writer-requested",
    family="anthropic",
    supported_roles=("proposal_writer",),
)
EVALUATOR = ModelDescriptor(
    id="evaluator-requested",
    label="Evaluator",
    provider="wandb",
    provider_model="evaluator-requested",
    family="meta",
    supported_roles=("proposal_evaluator",),
)
SCOPE_POLICY = {
    "version": "test-policy-v1",
    "markdown_policy": {
        "suffix": ".md",
        "excluded_directories": [".git"],
    },
}


def test_usage_keeps_token_counters_and_ignores_optional_provider_metadata() -> None:
    usage = reflection_module._usage(
        {
            "prompt_tokens": 39,
            "completion_tokens": 3,
            "total_tokens": 42,
            "prompt_tokens_details": None,
        }
    )

    assert dict(usage) == {
        "prompt_tokens": 39,
        "completion_tokens": 3,
        "total_tokens": 42,
    }


def _bundle(**contents: str) -> BundleSnapshot:
    return bundle_from_content_map(contents, scope=SCOPE)


def _challenge(candidate_id: str, winner: str) -> ChallengeResult:
    task = AuthoredTask(
        prompt="Fix the regression.",
        goal="Regression tests pass.",
        workspace_digest="sha256:workspace",
        author_model="author-model",
        author_backend="cli",
    )
    execution = ExecutionIdentity(
        model="gpt-5.6-sol",
        model_family="openai",
        harness="codex",
        harness_version="codex-cli 0.144.1",
        effort="high",
        image="runtime",
        image_digest=f"sha256:{'a' * 64}",
        command=("codex", "exec", "{task}"),
        timeout_seconds=60,
        network_enabled=True,
        environment=(NamedDigest(name="PATH", digest="sha256:path"),),
        runtime_files=(),
        workspace_digest="sha256:workspace",
    )
    arms = {
        arm: ArmResult(
            arm=arm,
            status="exited",
            exit_code=0,
            duration_seconds=1,
            initial_workspace_digest=f"sha256:{arm}-initial",
            final_workspace_digest=f"sha256:{arm}-final",
            transcript=f"{arm} result",
            transcript_digest=f"sha256:{arm}-transcript",
            artifact_digests={},
        )
        for arm in ("baseline", "candidate")
    }
    scores = {
        "candidate": (0.5, 1.0),
        "baseline": (1.0, 0.5),
        "tie": (0.75, 0.75),
    }[winner]
    return ChallengeResult(
        candidate_id=candidate_id,
        status="complete",
        winner=winner,
        reason=None,
        task=task,
        execution=execution,
        baseline=arms["baseline"],
        candidate=arms["candidate"],
        judges=(
            JudgeVerdict(
                position=1,
                requested_model="judge",
                resolved_model="judge",
                family="unknown",
                backend="cli",
                baseline_label="arm-2",
                candidate_label="arm-1",
                winner=winner,
                rationale="paired result",
                rubrics=(
                    RubricVerdict(
                        rubric_id="judge.verification",
                        baseline_score=scores[0],
                        candidate_score=scores[1],
                        winner=winner,
                        rationale="paired rubric result",
                    ),
                ),
                usage={},
            ),
        ),
    )


class SequenceClient:
    def __init__(
        self,
        backend: str,
        responses: list[tuple[dict[str, Any], JudgeResponse]],
    ) -> None:
        self.backend = backend
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _writer_client(*, count: int = 1) -> SequenceClient:
    return SequenceClient(
        "cli",
        [
            (
                {},
                JudgeResponse(
                    content=f"writer output {number}",
                    model="writer-resolved",
                    usage={"input_tokens": 10 + number, "output_tokens": 4},
                ),
            )
            for number in range(1, count + 1)
        ],
    )


def _writer_outputs(*contents: str) -> SequenceClient:
    return SequenceClient(
        "cli",
        [
            (
                {},
                JudgeResponse(
                    content=content,
                    model="writer-resolved",
                    usage={"input_tokens": 10 + number, "output_tokens": 4},
                ),
            )
            for number, content in enumerate(contents, start=1)
        ],
    )


def _evaluator_client(*scores: float) -> SequenceClient:
    return SequenceClient(
        "wandb-resolved",
        [
            (
                {"score": score, "rationale": f"rationale {score}"},
                JudgeResponse(
                    content="json",
                    model="Llama-3.1-8B",
                    usage={
                        "input_tokens": 20,
                        "output_tokens": 5,
                        "input_token_details": {"cached_tokens": 3},
                    },
                ),
            )
            for score in scores
        ],
    )


def _feedback() -> list[dict[str, Any]]:
    return [
        {
            "feedback_type": "weave_agent_signals.outcome.test",
            "payload": {
                "rating": 0.25,
                "tags": ["missing_verification"],
                "details": {"rationale": "Tests were not rerun."},
            },
        }
    ]


def _build_candidate(contents: Mapping[str, str]) -> BundleSnapshot:
    return bundle_from_content_map(contents, scope=SCOPE)


def _resolve_locator(locator: str, *, require_absent_for_create: bool = False) -> object:
    del require_absent_for_create
    if locator.startswith("../"):
        raise BundleValidationError(
            "outside managed scope",
            changed_locators={locator},
        )
    return object()


def _proposal(changes: list[dict[str, Any]]) -> str:
    return serialize_candidate_proposal(
        parse_candidate_proposal({"schema_version": 3, "changes": changes})
        if changes
        else CandidateProposal(3, ())
    )


def _baseline_only_result(seed_candidate: str, score: float = 0.3) -> SimpleNamespace:
    return SimpleNamespace(
        candidates=[{"current_candidate": seed_candidate}],
        val_aggregate_scores=[score],
        best_idx=0,
        _str_candidate_key="current_candidate",
    )


def test_invalid_writer_output_is_rejected_before_evaluator_and_retried():
    baseline = _bundle(**{"CLAUDE.md": "old"})
    valid = _proposal([{"action": "update", "locator": "CLAUDE.md", "content": "new"}])
    writer_client = _writer_outputs("not JSON", valid)
    evaluator_client = _evaluator_client(0.3, 0.8)
    prompts: list[list[dict[str, str]]] = []

    def optimize(*, seed_candidate, evaluator, config, **_kwargs):
        evaluator(seed_candidate)
        with pytest.raises(ReflectionEvaluationError, match="proposal"):
            config.reflection.reflection_lm("first proposal")
        assert len(evaluator_client.calls) == 1
        candidate = config.reflection.reflection_lm("second proposal")
        prompts.extend(call["messages"] for call in writer_client.calls)
        evaluator(candidate)
        return SimpleNamespace()

    result = run_reflection(
        baseline=baseline,
        feedback=_feedback(),
        coaching_text="Improve.",
        scope_policy=SCOPE_POLICY,
        requested_writer=WRITER,
        requested_evaluator=EVALUATOR,
        writer_client=writer_client,
        evaluator_client=evaluator_client,
        resolve_locator=_resolve_locator,
        candidate_budget=2,
        optimizer=optimize,
    )

    assert len(writer_client.calls) == 2
    assert len(evaluator_client.calls) == 2
    assert [attempt.status for attempt in result.generation_attempts] == ["failed", "succeeded"]
    assert [candidate.bundle.revision for candidate in result.candidates] == [
        _bundle(**{"CLAUDE.md": "new"}).revision
    ]
    assert "previous proposal" in str(prompts[1]).lower()
    assert "json" in str(prompts[1]).lower()


def test_validation_correction_is_used_only_for_the_next_writer_attempt():
    first_valid = _proposal(
        [{"action": "update", "locator": "CLAUDE.md", "content": "first valid"}]
    )
    second_valid = _proposal(
        [{"action": "update", "locator": "CLAUDE.md", "content": "second valid"}]
    )
    writer_client = _writer_outputs("not JSON", first_valid, second_valid)

    def optimize(*, seed_candidate, evaluator, config, **_kwargs):
        evaluator(seed_candidate)
        with pytest.raises(ReflectionEvaluationError, match="proposal"):
            config.reflection.reflection_lm("invalid")
        evaluator(config.reflection.reflection_lm("first valid"))
        evaluator(config.reflection.reflection_lm("second valid"))
        return SimpleNamespace()

    run_reflection(
        baseline=_bundle(**{"CLAUDE.md": "old"}),
        feedback=_feedback(),
        coaching_text="Improve.",
        scope_policy=SCOPE_POLICY,
        requested_writer=WRITER,
        requested_evaluator=EVALUATOR,
        writer_client=writer_client,
        evaluator_client=_evaluator_client(0.3, 0.7, 0.8),
        resolve_locator=_resolve_locator,
        candidate_budget=3,
        optimizer=optimize,
    )

    correction = "Correct the previous proposal rejection"
    assert correction in str(writer_client.calls[1]["messages"])
    assert correction not in str(writer_client.calls[2]["messages"])


def test_real_gepa_continues_after_invalid_first_writer_output():
    baseline = _bundle(**{"CLAUDE.md": "old"})
    valid = _proposal([{"action": "update", "locator": "CLAUDE.md", "content": "new"}])
    writer_client = _writer_outputs("not JSON", valid)
    evaluator_client = _evaluator_client(0.3, 0.8)

    result = run_reflection(
        baseline=baseline,
        feedback=_feedback(),
        coaching_text="Improve.",
        scope_policy=SCOPE_POLICY,
        requested_writer=WRITER,
        requested_evaluator=EVALUATOR,
        writer_client=writer_client,
        evaluator_client=evaluator_client,
        resolve_locator=_resolve_locator,
        candidate_budget=2,
    )

    assert len(writer_client.calls) == 2
    assert len(evaluator_client.calls) == 2
    assert [attempt.status for attempt in result.generation_attempts] == ["failed", "succeeded"]


def test_real_gepa_preserves_valid_proposal_with_markdown_fences():
    baseline = _bundle(**{"CLAUDE.md": "old"})
    valid = _proposal(
        [
            {
                "action": "update",
                "locator": "CLAUDE.md",
                "content": "Run:\n```bash\npytest -q\n```",
            }
        ]
    )
    writer_client = _writer_outputs(valid)
    evaluator_client = _evaluator_client(0.3, 0.8)

    result = run_reflection(
        baseline=baseline,
        feedback=_feedback(),
        coaching_text="Improve.",
        scope_policy=SCOPE_POLICY,
        requested_writer=WRITER,
        requested_evaluator=EVALUATOR,
        writer_client=writer_client,
        evaluator_client=evaluator_client,
        resolve_locator=_resolve_locator,
        candidate_budget=1,
    )

    assert len(evaluator_client.calls) == 2
    assert result.generation_attempts[0].status == "succeeded"
    assert [candidate.bundle.revision for candidate in result.candidates] == [
        _bundle(**{"CLAUDE.md": "Run:\n```bash\npytest -q\n```"}).revision
    ]


def test_all_invalid_outputs_finish_with_audited_baseline():
    baseline = _bundle(**{"CLAUDE.md": "old"})
    writer_client = _writer_outputs("bad one", "bad two", "bad three")
    evaluator_client = _evaluator_client(0.3)

    result = run_reflection(
        baseline=baseline,
        feedback=_feedback(),
        coaching_text="Improve.",
        scope_policy=SCOPE_POLICY,
        requested_writer=WRITER,
        requested_evaluator=EVALUATOR,
        writer_client=writer_client,
        evaluator_client=evaluator_client,
        resolve_locator=_resolve_locator,
        candidate_budget=3,
    )

    assert result.baseline_evaluation.target_revision == baseline.revision
    assert result.candidates == ()
    assert result.recommended_candidate_id is None
    assert result.baseline_won is False
    assert result.reason == "No valid proposal generated"
    assert len(result.generation_attempts) == 3
    assert all(attempt.status == "failed" for attempt in result.generation_attempts)


def test_writer_transport_failure_remains_terminal():
    baseline = _bundle(**{"CLAUDE.md": "old"})
    writer_client = SequenceClient("cli", [])

    def fail(**_kwargs):
        raise RuntimeError("writer transport unavailable")

    writer_client.chat_json = fail

    with pytest.raises(ReflectionEvaluationError, match="writer transport unavailable"):
        run_reflection(
            baseline=baseline,
            feedback=_feedback(),
            coaching_text="Improve.",
            scope_policy=SCOPE_POLICY,
            requested_writer=WRITER,
            requested_evaluator=EVALUATOR,
            writer_client=writer_client,
            evaluator_client=_evaluator_client(0.3),
            resolve_locator=_resolve_locator,
            candidate_budget=2,
        )


def test_candidate_budget_counts_writer_calls():
    writer_client = _writer_outputs("bad one", "bad two", "bad three")

    result = run_reflection(
        baseline=_bundle(**{"CLAUDE.md": "old"}),
        feedback=_feedback(),
        coaching_text="Improve.",
        scope_policy=SCOPE_POLICY,
        requested_writer=WRITER,
        requested_evaluator=EVALUATOR,
        writer_client=writer_client,
        evaluator_client=_evaluator_client(0.3),
        resolve_locator=_resolve_locator,
        candidate_budget=3,
    )

    assert len(writer_client.calls) == 3
    assert len(result.generation_attempts) == 3


def test_duplicate_revision_is_rejected_without_rescoring():
    baseline = _bundle(**{"CLAUDE.md": "old"})
    proposal = _proposal([{"action": "update", "locator": "CLAUDE.md", "content": "same"}])
    writer_client = _writer_outputs(proposal, proposal)
    evaluator_client = _evaluator_client(0.3, 0.7)

    def optimize(*, seed_candidate, evaluator, config, **_kwargs):
        evaluator(seed_candidate)
        first = config.reflection.reflection_lm("first")
        evaluator(first)
        with pytest.raises(ReflectionEvaluationError, match="duplicate"):
            config.reflection.reflection_lm("second")
        return SimpleNamespace()

    result = run_reflection(
        baseline=baseline,
        feedback=_feedback(),
        coaching_text="Improve.",
        scope_policy=SCOPE_POLICY,
        requested_writer=WRITER,
        requested_evaluator=EVALUATOR,
        writer_client=writer_client,
        evaluator_client=evaluator_client,
        resolve_locator=_resolve_locator,
        candidate_budget=2,
        optimizer=optimize,
    )

    assert len(evaluator_client.calls) == 2
    assert len(result.candidates) == 1
    assert [attempt.status for attempt in result.generation_attempts] == ["succeeded", "failed"]
    assert result.generation_attempts[1].candidate_revision == result.candidates[0].bundle.revision
    assert "duplicate" in (result.generation_attempts[1].error or "").lower()


def test_each_bundle_revision_is_evaluated_exactly_once():
    baseline = _bundle(**{"CLAUDE.md": "old"})
    proposal = _proposal([{"action": "update", "locator": "CLAUDE.md", "content": "new"}])
    evaluator_client = _evaluator_client(0.3, 0.8)
    side_info: list[dict[str, Any]] = []

    def optimize(*, seed_candidate, evaluator, config, **_kwargs):
        side_info.append(evaluator(seed_candidate)[1])
        side_info.append(evaluator(seed_candidate)[1])
        candidate = config.reflection.reflection_lm("candidate")
        side_info.append(evaluator(candidate)[1])
        side_info.append(evaluator(candidate)[1])
        return SimpleNamespace()

    result = run_reflection(
        baseline=baseline,
        feedback=_feedback(),
        coaching_text="Improve.",
        scope_policy=SCOPE_POLICY,
        requested_writer=WRITER,
        requested_evaluator=EVALUATOR,
        writer_client=_writer_outputs(proposal),
        evaluator_client=evaluator_client,
        resolve_locator=_resolve_locator,
        candidate_budget=1,
        optimizer=optimize,
    )

    assert len(evaluator_client.calls) == 2
    assert side_info[0] is side_info[1]
    assert side_info[2] is side_info[3]
    assert result.baseline_evaluation.evaluation_id == "evaluation-1"
    assert result.candidates[0].evaluation.evaluation_id == "evaluation-2"


def test_rejected_attempt_keeps_paths_digest_and_bounded_excerpt():
    baseline = _bundle(**{"CLAUDE.md": "old"})
    secret = "must-not-leak"
    invalid = _proposal(
        [
            {
                "action": "create",
                "locator": "../secret.md",
                "content": f'"OPENAI_API_KEY":"{secret}"\n' + ("x" * 2_000),
            }
        ]
    )

    def reject_unmanaged(locator: str, **_kwargs: object) -> object:
        raise BundleValidationError(
            "outside managed scope",
            changed_locators={locator},
        )

    def optimize(*, seed_candidate, evaluator, config, **_kwargs):
        evaluator(seed_candidate)
        with pytest.raises(ReflectionEvaluationError, match="outside managed scope"):
            config.reflection.reflection_lm("candidate")
        return _baseline_only_result(seed_candidate)

    result = run_reflection(
        baseline=baseline,
        feedback=_feedback(),
        coaching_text="Improve.",
        scope_policy=SCOPE_POLICY,
        requested_writer=WRITER,
        requested_evaluator=EVALUATOR,
        writer_client=_writer_outputs(invalid),
        evaluator_client=_evaluator_client(0.3),
        resolve_locator=reject_unmanaged,
        candidate_budget=1,
        optimizer=optimize,
    )

    attempt = result.generation_attempts[0]
    assert attempt.changed_paths == ("../secret.md",)
    assert attempt.response_digest == f"sha256:{hashlib.sha256(invalid.encode()).hexdigest()}"
    assert attempt.response_excerpt is not None
    assert len(attempt.response_excerpt) <= 1_000
    assert secret not in attempt.response_excerpt
    assert attempt.error_type == "BundleValidationError"
    assert attempt.error == "outside managed scope"


def test_rejected_parseable_proposal_redacts_escaped_content_key():
    secret = "escaped-key-secret-must-not-leak"
    invalid = (
        '{"schema_version":3,"changes":[{"action":"create",'
        '"locator":"../secret.md","cont\\u0065nt":"'
        f'{secret}"}}]}}'
    )

    def reject_unmanaged(locator: str, **_kwargs: object) -> object:
        raise BundleValidationError(
            "outside managed scope",
            changed_locators={locator},
        )

    def optimize(*, seed_candidate, evaluator, config, **_kwargs):
        evaluator(seed_candidate)
        with pytest.raises(ReflectionEvaluationError, match="outside managed scope"):
            config.reflection.reflection_lm("candidate")
        return _baseline_only_result(seed_candidate)

    result = run_reflection(
        baseline=_bundle(**{"CLAUDE.md": "old"}),
        feedback=_feedback(),
        coaching_text="Improve.",
        scope_policy=SCOPE_POLICY,
        requested_writer=WRITER,
        requested_evaluator=EVALUATOR,
        writer_client=_writer_outputs(invalid),
        evaluator_client=_evaluator_client(0.3),
        resolve_locator=reject_unmanaged,
        candidate_budget=1,
        optimizer=optimize,
    )

    excerpt = result.generation_attempts[0].response_excerpt
    assert excerpt is not None
    assert secret not in excerpt
    assert "../secret.md" in excerpt


def test_rejected_malformed_single_quoted_proposal_never_persists_body():
    secret = "single-quoted-secret-must-not-leak"
    invalid = (
        "{'schema_version': 3, 'changes': [{'action': 'update', "
        f"'locator': 'CLAUDE.md', 'content': '{secret}" + ("x" * 2_000) + "'}]}"
    )

    def optimize(*, seed_candidate, evaluator, config, **_kwargs):
        evaluator(seed_candidate)
        with pytest.raises(ReflectionEvaluationError, match="proposal"):
            config.reflection.reflection_lm("candidate")
        return _baseline_only_result(seed_candidate)

    result = run_reflection(
        baseline=_bundle(**{"CLAUDE.md": "old"}),
        feedback=_feedback(),
        coaching_text="Improve.",
        scope_policy=SCOPE_POLICY,
        requested_writer=WRITER,
        requested_evaluator=EVALUATOR,
        writer_client=_writer_outputs(invalid),
        evaluator_client=_evaluator_client(0.3),
        resolve_locator=_resolve_locator,
        candidate_budget=1,
        optimizer=optimize,
    )

    attempt = result.generation_attempts[0]
    assert attempt.response_digest == f"sha256:{hashlib.sha256(invalid.encode()).hexdigest()}"
    assert attempt.response_excerpt is not None
    assert len(attempt.response_excerpt) <= 1_000
    assert secret not in attempt.response_excerpt
    assert "malformed" in attempt.response_excerpt.lower()


def test_writer_process_failure_is_terminal_without_persisting_raw_output():
    writer_client = SequenceClient("cli", [])
    raw_secret = "must-not-leak"

    def fail(**_kwargs):
        raise RuntimeError(f"codex judge exited 1: raw output {raw_secret}")

    writer_client.chat_json = fail
    events: list[dict[str, Any]] = []

    with pytest.raises(ReflectionEvaluationError, match="proposal writer process exited 1"):
        run_reflection(
            baseline=_bundle(**{"CLAUDE.md": "old"}),
            feedback=_feedback(),
            coaching_text="Improve.",
            scope_policy=SCOPE_POLICY,
            requested_writer=WRITER,
            requested_evaluator=EVALUATOR,
            writer_client=writer_client,
            evaluator_client=_evaluator_client(0.3),
            resolve_locator=_resolve_locator,
            candidate_budget=1,
            progress_callback=events.append,
        )

    assert raw_secret not in repr(events)
    assert events[-1]["message"] == "Proposal writer failed: proposal writer process exited 1"


def test_evaluator_failure_remains_terminal_and_actionable():
    evaluator_client = SequenceClient("wandb", [])

    def fail(**_kwargs):
        raise RuntimeError("evaluator unavailable; check W&B credits")

    evaluator_client.chat_json = fail
    events: list[dict[str, Any]] = []

    with pytest.raises(ReflectionEvaluationError, match="evaluator unavailable; check W&B credits"):
        run_reflection(
            baseline=_bundle(**{"CLAUDE.md": "old"}),
            feedback=_feedback(),
            coaching_text="Improve.",
            scope_policy=SCOPE_POLICY,
            requested_writer=WRITER,
            requested_evaluator=EVALUATOR,
            writer_client=_writer_outputs("unused"),
            evaluator_client=evaluator_client,
            resolve_locator=_resolve_locator,
            candidate_budget=1,
            progress_callback=events.append,
        )

    assert events[-1]["phase"] == "evaluation_failed"
    assert events[-1]["message"] == (
        "Proposal evaluator failed: evaluator unavailable; check W&B credits"
    )


def test_writer_empty_or_noop_proposal_is_invalid():
    baseline = _bundle(**{"AGENTS.md": "old"})

    def optimize(*, seed_candidate, evaluator, config, **_kwargs):
        evaluator(seed_candidate)
        with pytest.raises(ReflectionEvaluationError, match="effective change"):
            config.reflection.reflection_lm("empty proposal")
        with pytest.raises(ReflectionEvaluationError, match="must change"):
            config.reflection.reflection_lm("no-op proposal")
        return SimpleNamespace()

    result = run_reflection(
        baseline=baseline,
        feedback=_feedback(),
        coaching_text="Improve.",
        scope_policy=SCOPE_POLICY,
        requested_writer=WRITER,
        requested_evaluator=EVALUATOR,
        writer_client=_writer_outputs(
            _proposal([]),
            _proposal(
                [
                    {
                        "action": "update",
                        "locator": "AGENTS.md",
                        "content": "old",
                    }
                ]
            ),
        ),
        evaluator_client=_evaluator_client(0.4),
        resolve_locator=_resolve_locator,
        candidate_budget=2,
        optimizer=optimize,
    )

    assert [attempt.status for attempt in result.generation_attempts] == ["failed", "failed"]


def test_internal_baseline_may_use_empty_proposal():
    baseline = _bundle(**{"AGENTS.md": "old"})

    def optimize(*, seed_candidate, evaluator, **_kwargs):
        assert seed_candidate == _proposal([])
        evaluator(seed_candidate)
        return SimpleNamespace(
            candidates=[{"current_candidate": seed_candidate}],
            val_aggregate_scores=[0.4],
            best_idx=0,
            _str_candidate_key="current_candidate",
        )

    result = run_reflection(
        baseline=baseline,
        feedback=_feedback(),
        coaching_text="Improve.",
        scope_policy=SCOPE_POLICY,
        requested_writer=WRITER,
        requested_evaluator=EVALUATOR,
        writer_client=_writer_client(),
        evaluator_client=_evaluator_client(0.4),
        resolve_locator=_resolve_locator,
        candidate_budget=1,
        optimizer=optimize,
    )

    assert result.baseline is baseline
    assert result.candidates == ()


def test_writer_prompt_lists_exact_contract_scope_and_inventory():
    baseline = _bundle(**{"AGENTS.md": "alpha", "docs/check.md": "beta"})
    scope_policy = {
        **SCOPE_POLICY,
        "targets": [
            {
                "kind": "markdown_root",
                "id": "project",
                "description": "Repository-wide agent instructions.",
                "allowed_actions": ["update", "create"],
                "locator_format": "markdown:project/<relative-path.md>",
            }
        ],
    }

    def optimize(*, seed_candidate, evaluator, objective, background, **_kwargs):
        prompt = f"{objective}\n{background}"
        assert '"AGENTS.md": "alpha"' in prompt
        assert '"docs/check.md": "beta"' in prompt
        assert '"suffix": ".md"' in prompt
        assert '"description": "Repository-wide agent instructions."' in prompt
        assert '"allowed_actions": [\n        "update",\n        "create"' in prompt
        assert '"locator_format": "markdown:project/<relative-path.md>"' in prompt
        assert "create requires a locator admitted by the pinned target registry" in prompt
        assert "update requires a locator present in baseline A" in prompt
        assert "delete requires" not in prompt
        assert "one atomic multi-file proposal" in prompt
        assert "JSON only" in prompt
        assert "unrelated documentation" in prompt
        assert "evaluator score" in prompt
        evaluator(seed_candidate)
        return SimpleNamespace(
            candidates=[{"current_candidate": seed_candidate}],
            val_aggregate_scores=[0.4],
            best_idx=0,
            _str_candidate_key="current_candidate",
        )

    run_reflection(
        baseline=baseline,
        feedback=_feedback(),
        coaching_text="Improve verification.",
        scope_policy=scope_policy,
        requested_writer=WRITER,
        requested_evaluator=EVALUATOR,
        writer_client=_writer_client(),
        evaluator_client=_evaluator_client(0.4),
        resolve_locator=_resolve_locator,
        candidate_budget=1,
        optimizer=optimize,
    )


def test_writer_passes_proposal_schema_to_chat_client():
    baseline = _bundle(**{"AGENTS.md": "old"})
    proposal = _proposal([{"action": "update", "locator": "AGENTS.md", "content": "new"}])
    writer_client = _writer_outputs(proposal)
    evaluator_client = _evaluator_client(0.4, 0.5)

    def optimize(*, seed_candidate, evaluator, config, **_kwargs):
        evaluator(seed_candidate)
        candidate = config.reflection.reflection_lm("propose")
        evaluator(candidate)
        return SimpleNamespace()

    run_reflection(
        baseline=baseline,
        feedback=_feedback(),
        coaching_text="Improve.",
        scope_policy=SCOPE_POLICY,
        requested_writer=WRITER,
        requested_evaluator=EVALUATOR,
        writer_client=writer_client,
        evaluator_client=evaluator_client,
        resolve_locator=_resolve_locator,
        candidate_budget=1,
        optimizer=optimize,
    )

    assert writer_client.calls[0]["response_schema"] is REFLECTION_PROPOSAL_SCHEMA
    evaluator_schema = evaluator_client.calls[0]["response_schema"]
    assert evaluator_schema.name == "reflection_bundle_evaluation"
    assert evaluator_schema.schema["properties"]["score"] == {
        "type": "number",
        "minimum": 0,
        "maximum": 1,
    }


def test_gepa_applies_one_proposal_to_update_and_create_targets():
    baseline = _bundle(**{"CLAUDE.md": "old", ".claude/skills/keep.md": "keep"})
    candidate = _proposal(
        [
            {"action": "update", "locator": "CLAUDE.md", "content": "new"},
            {
                "action": "create",
                "locator": ".claude/commands/add.md",
                "content": "created",
            },
        ]
    )

    def optimize(*, seed_candidate, evaluator, config, **_kwargs):
        assert seed_candidate == _proposal([])
        evaluator(seed_candidate)
        generated = config.reflection.reflection_lm("propose complete manifest")
        assert generated == candidate
        evaluator(generated)
        return SimpleNamespace()

    result = run_reflection(
        baseline=baseline,
        feedback=_feedback(),
        coaching_text="Improve.",
        scope_policy=SCOPE_POLICY,
        requested_writer=WRITER,
        requested_evaluator=EVALUATOR,
        writer_client=_writer_outputs(candidate),
        evaluator_client=_evaluator_client(0.3, 0.8),
        resolve_locator=_resolve_locator,
        candidate_budget=1,
        optimizer=optimize,
    )

    assert [
        (action.action, action.locator)
        for action in compare_bundles(baseline, result.candidates[0].bundle).actions
    ] == [
        ("create", ".claude/commands/add.md"),
        ("update", "CLAUDE.md"),
    ]


def test_proposal_candidates_reject_invalid_json_and_unmanaged_locators():
    baseline = _bundle(**{"CLAUDE.md": "old"})
    unmanaged = _proposal(
        [
            {"action": "update", "locator": "CLAUDE.md", "content": "new"},
            {"action": "create", "locator": "../secret.md", "content": "steal"},
        ]
    )

    def optimize(*, seed_candidate, evaluator, config, **_kwargs):
        evaluator(seed_candidate)
        with pytest.raises(ReflectionEvaluationError, match="proposal"):
            config.reflection.reflection_lm("invalid JSON")
        with pytest.raises(ReflectionEvaluationError, match="outside managed scope"):
            config.reflection.reflection_lm("unmanaged path")
        return SimpleNamespace()

    events = []
    result = run_reflection(
        baseline=baseline,
        feedback=_feedback(),
        coaching_text="Improve.",
        scope_policy=SCOPE_POLICY,
        requested_writer=WRITER,
        requested_evaluator=EVALUATOR,
        writer_client=_writer_outputs("not JSON", unmanaged),
        evaluator_client=_evaluator_client(0.4),
        resolve_locator=_resolve_locator,
        candidate_budget=2,
        optimizer=optimize,
        progress_callback=events.append,
    )

    assert [attempt.status for attempt in result.generation_attempts] == ["failed", "failed"]
    assert [attempt.error_type for attempt in result.generation_attempts] == [
        "JSONDecodeError",
        "BundleValidationError",
    ]
    assert events[-1]["phase"] == "selection_complete"
    assert events[-1]["attempted"] == 2
    assert events[-1]["valid"] == 0
    assert events[-1]["rejected"] == 2


def test_reflection_requires_at_least_one_finite_rated_signal():
    feedback = [
        {"feedback_type": "bad.bool", "payload": {"rating": True}},
        {"feedback_type": "bad.nan", "payload": {"rating": float("nan")}},
        {"feedback_type": "bad.range", "payload": {"rating": 1.01}},
        {
            "feedback_type": "weave_agent_signals.judge.verification",
            "payload": {"rating": 0.5, "details": {"evaluation_unit": "episode"}},
        },
    ]

    with pytest.raises(ReflectionEvaluationError, match="valid rated signal"):
        run_reflection(
            baseline=_bundle(**{"CLAUDE.md": "old"}),
            feedback=feedback,
            coaching_text="Improve.",
            scope_policy=SCOPE_POLICY,
            requested_writer=WRITER,
            requested_evaluator=EVALUATOR,
            writer_client=_writer_client(),
            evaluator_client=_evaluator_client(),
            resolve_locator=_resolve_locator,
        )


def test_reflection_records_exact_writer_evaluator_and_bundle_provenance():
    baseline = _bundle(**{"CLAUDE.md": "old", ".claude/skills/check.md": "check"})
    candidate_map = {
        "CLAUDE.md": "new",
        ".claude/skills/check.md": "check more",
    }
    proposal = _proposal(
        [
            {"action": "update", "locator": "CLAUDE.md", "content": "new"},
            {
                "action": "update",
                "locator": ".claude/skills/check.md",
                "content": "check more",
            },
        ]
    )
    writer_client = _writer_outputs(proposal)
    evaluator_client = _evaluator_client(0.30, 0.82)

    def optimize(*, seed_candidate, evaluator, config, **_kwargs):
        assert seed_candidate == _proposal([])
        _, baseline_side_info = evaluator(seed_candidate)
        assert baseline_side_info["complete_bundle"] == {
            "CLAUDE.md": "old",
            ".claude/skills/check.md": "check",
        }
        generated = config.reflection.reflection_lm("propose")
        assert generated == proposal
        first = evaluator(generated)
        second = evaluator(generated)  # GEPA full evaluation after acceptance.
        assert first[1] is second[1]
        return SimpleNamespace()

    result = run_reflection(
        baseline=baseline,
        feedback=_feedback(),
        coaching_text="Improve verification.",
        scope_policy=SCOPE_POLICY,
        requested_writer=WRITER,
        requested_evaluator=EVALUATOR,
        writer_client=writer_client,
        evaluator_client=evaluator_client,
        resolve_locator=_resolve_locator,
        candidate_budget=3,
        optimizer=optimize,
    )

    assert isinstance(result, ReflectionResult)
    assert result.baseline is baseline
    assert result.baseline_score == 0.30
    assert result.score_basis == PREDICTED_EVALUATOR_SCORE_BASIS
    assert result.baseline_evaluation.target_revision == baseline.revision
    assert result.baseline_evaluation.requested_model == EVALUATOR.id
    assert result.baseline_evaluation.requested_family == EVALUATOR.family
    assert result.baseline_evaluation.requested_backend == EVALUATOR.provider
    assert result.baseline_evaluation.resolved_model == "Llama-3.1-8B"
    assert result.baseline_evaluation.resolved_family == "meta"
    assert result.baseline_evaluation.resolved_backend == "wandb-resolved"
    assert dict(result.baseline_evaluation.usage) == {
        "input_tokens": 20,
        "output_tokens": 5,
    }

    attempt = result.generation_attempts[0]
    assert attempt.attempt_id == "attempt-1"
    assert attempt.number == 1
    assert attempt.status == "succeeded"
    assert attempt.requested_writer is WRITER
    assert attempt.resolved_model == "writer-resolved"
    assert attempt.resolved_family == "unknown"
    assert attempt.resolved_backend == "cli"
    assert dict(attempt.usage) == {"input_tokens": 11, "output_tokens": 4}

    candidate = result.candidates[0]
    assert candidate.bundle == _build_candidate(candidate_map)
    assert candidate.generation_attempt_id == attempt.attempt_id
    assert candidate.requested_writer is WRITER
    assert candidate.resolved_writer_model == "writer-resolved"
    assert candidate.resolved_writer_family == "unknown"
    assert candidate.resolved_writer_backend == "cli"
    assert candidate.score == 0.82
    assert candidate.score_delta == pytest.approx(0.52)
    assert candidate.evaluation.target_revision == candidate.bundle.revision
    assert result.recommended_candidate_id == candidate.candidate_id
    assert result.baseline_won is False

    assert ReflectionResult.from_dict(result.to_dict()) == result

    mismatched = result.to_dict()
    mismatched["candidates"][0]["score"] = 0.81
    with pytest.raises(ValueError, match="matching evaluator record"):
        ReflectionResult.from_dict(mismatched)

    for field in ("resolved_writer_model", "resolved_writer_family"):
        mismatched = result.to_dict()
        mismatched["candidates"][0][field] = "contradictory"
        with pytest.raises(ValueError, match="writer provenance"):
            ReflectionResult.from_dict(mismatched)

    mismatched = result.to_dict()
    mismatched["candidates"][0]["rationale"] = "contradictory"
    with pytest.raises(ValueError, match="matching evaluator rationale"):
        ReflectionResult.from_dict(mismatched)

    for field in (
        "requested_model",
        "requested_family",
        "requested_backend",
        "resolved_model",
        "resolved_family",
        "resolved_backend",
    ):
        mismatched = result.to_dict()
        mismatched["candidates"][0]["evaluation"][field] = "contradictory"
        with pytest.raises(ValueError, match="authoritative baseline evaluator identity"):
            ReflectionResult.from_dict(mismatched)


def test_reflection_recomputes_the_winner_from_evaluated_scores():
    baseline = _bundle(**{"CLAUDE.md": "old"})
    candidate = _proposal([{"action": "update", "locator": "CLAUDE.md", "content": "worse"}])

    def optimize(*, seed_candidate, evaluator, config, **_kwargs):
        evaluator(seed_candidate)
        generated = config.reflection.reflection_lm("propose")
        assert generated == candidate
        evaluator(generated)
        return SimpleNamespace()

    result = run_reflection(
        baseline=baseline,
        feedback=_feedback(),
        coaching_text="Improve.",
        scope_policy=SCOPE_POLICY,
        requested_writer=WRITER,
        requested_evaluator=EVALUATOR,
        writer_client=_writer_outputs(candidate),
        evaluator_client=_evaluator_client(0.9, 0.1),
        resolve_locator=_resolve_locator,
        candidate_budget=1,
        optimizer=optimize,
    )

    assert result.baseline_won is True
    assert result.recommended_candidate_id is None


@pytest.mark.parametrize(
    ("winner", "recommended", "baseline_won"),
    [
        ("candidate", True, False),
        ("baseline", False, True),
        ("tie", False, True),
    ],
)
def test_paired_challenge_replaces_provisional_prediction_for_recommendation(
    winner: str,
    recommended: bool,
    baseline_won: bool,
):
    baseline = _bundle(**{"CLAUDE.md": "old"})
    proposal = _proposal([{"action": "update", "locator": "CLAUDE.md", "content": "new"}])

    def optimize(*, seed_candidate, evaluator, config, **_kwargs):
        evaluator(seed_candidate)
        generated = config.reflection.reflection_lm("propose")
        assert generated == proposal
        evaluator(generated)
        return SimpleNamespace()

    provisional = run_reflection(
        baseline=baseline,
        feedback=_feedback(),
        coaching_text="Improve.",
        scope_policy=SCOPE_POLICY,
        requested_writer=WRITER,
        requested_evaluator=EVALUATOR,
        writer_client=_writer_outputs(proposal),
        evaluator_client=_evaluator_client(0.3, 0.8),
        resolve_locator=_resolve_locator,
        candidate_budget=1,
        optimizer=optimize,
    )
    candidate_id = provisional.recommended_candidate_id
    assert candidate_id is not None

    verified = provisional.with_challenge(_challenge(candidate_id, winner))

    assert verified.provisional_candidate_id == candidate_id
    assert verified.recommended_candidate_id == (candidate_id if recommended else None)
    assert verified.baseline_won is baseline_won
    assert ReflectionResult.from_dict(verified.to_dict()) == verified


def test_result_rejects_candidate_revision_equal_to_baseline():
    baseline = _bundle(**{"CLAUDE.md": "old"})
    proposal = _proposal([{"action": "update", "locator": "CLAUDE.md", "content": "new"}])

    def optimize(*, seed_candidate, evaluator, config, **_kwargs):
        evaluator(seed_candidate)
        candidate = config.reflection.reflection_lm("propose")
        evaluator(candidate)
        return SimpleNamespace()

    result = run_reflection(
        baseline=baseline,
        feedback=_feedback(),
        coaching_text="Improve.",
        scope_policy=SCOPE_POLICY,
        requested_writer=WRITER,
        requested_evaluator=EVALUATOR,
        writer_client=_writer_outputs(proposal),
        evaluator_client=_evaluator_client(0.3, 0.8),
        resolve_locator=_resolve_locator,
        candidate_budget=1,
        optimizer=optimize,
    )
    tampered = result.to_dict()
    tampered["candidates"][0]["bundle"] = baseline.to_dict()
    tampered["candidates"][0]["evaluation"]["target_revision"] = baseline.revision
    tampered["generation_attempts"][0]["candidate_revision"] = baseline.revision

    with pytest.raises(ValueError, match="baseline"):
        ReflectionResult.from_dict(tampered)


def test_evaluator_failure_is_retained_in_progress_before_raising():
    baseline = _bundle(**{"CLAUDE.md": "old"})
    evaluator_client = SequenceClient("wandb", [])

    def fail(**_kwargs):
        raise RuntimeError("evaluator unavailable")

    evaluator_client.chat_json = fail
    events: list[dict[str, Any]] = []

    def optimize(*, seed_candidate, evaluator, **_kwargs):
        evaluator(seed_candidate)

    with pytest.raises(ReflectionEvaluationError, match="evaluator unavailable"):
        run_reflection(
            baseline=baseline,
            feedback=_feedback(),
            coaching_text="Improve.",
            scope_policy=SCOPE_POLICY,
            requested_writer=WRITER,
            requested_evaluator=EVALUATOR,
            writer_client=_writer_client(),
            evaluator_client=evaluator_client,
            resolve_locator=_resolve_locator,
            candidate_budget=1,
            optimizer=optimize,
            progress_callback=events.append,
        )

    assert events[-1]["phase"] == "evaluation_failed"
    assert events[-1]["model"] == EVALUATOR.id
    assert events[-1]["acting_role"] == "proposal_evaluator"
    assert events[-1]["error_type"] == "RuntimeError"


def test_reflection_passes_candidate_budget_and_patience_two_to_gepa():
    baseline = _bundle(**{"CLAUDE.md": "old"})
    seen: dict[str, Any] = {}

    def optimize(*, seed_candidate, evaluator, config, **_kwargs):
        assert "dataset" not in _kwargs
        seen["budget"] = config.engine.max_candidate_proposals
        seen["patience"] = next(
            callback.max_iterations_without_improvement
            for callback in config.stop_callbacks
            if hasattr(callback, "max_iterations_without_improvement")
        )
        evaluator(seed_candidate)
        return SimpleNamespace(
            candidates=[{"current_candidate": seed_candidate}],
            val_aggregate_scores=[0.4],
            best_idx=0,
            _str_candidate_key="current_candidate",
        )

    result = run_reflection(
        baseline=baseline,
        feedback=_feedback(),
        coaching_text="Improve.",
        scope_policy=SCOPE_POLICY,
        requested_writer=WRITER,
        requested_evaluator=EVALUATOR,
        writer_client=_writer_client(),
        evaluator_client=_evaluator_client(0.4),
        resolve_locator=_resolve_locator,
        candidate_budget=3,
        optimizer=optimize,
    )

    assert seen == {"budget": 3, "patience": 2}
    assert result.generation_attempts == ()


def test_reflection_preserves_explicit_missing_targets_in_exact_baseline():
    baseline = bundle_from_content_map(
        {"CLAUDE.md": "old"},
        scope=SCOPE,
        include_missing=(".claude/skills/future.md",),
    )

    def optimize(*, seed_candidate, evaluator, **_kwargs):
        evaluator(seed_candidate)
        return SimpleNamespace(
            candidates=[{"current_candidate": seed_candidate}],
            val_aggregate_scores=[0.4],
            best_idx=0,
            _str_candidate_key="current_candidate",
        )

    result = run_reflection(
        baseline=baseline,
        feedback=_feedback(),
        coaching_text="Improve.",
        scope_policy=SCOPE_POLICY,
        requested_writer=WRITER,
        requested_evaluator=EVALUATOR,
        writer_client=_writer_client(),
        evaluator_client=_evaluator_client(0.4),
        resolve_locator=_resolve_locator,
        candidate_budget=1,
        optimizer=optimize,
    )

    assert result.baseline is baseline
    assert result.baseline.locators == (
        ".claude/skills/future.md",
        "CLAUDE.md",
    )


def test_reflection_cancellation_preserves_cancel_signal():
    baseline = _bundle(**{"CLAUDE.md": "old"})

    def optimize(*, seed_candidate, evaluator, config, **_kwargs):
        evaluator(seed_candidate)
        config.reflection.reflection_lm("proposal")
        return SimpleNamespace(
            candidates=[{"current_candidate": seed_candidate}],
            val_aggregate_scores=[0.4],
            best_idx=0,
            _str_candidate_key="current_candidate",
        )

    cancel_checks = iter((False, False, True))

    with pytest.raises(ReflectionCancelled, match="cancelled"):
        run_reflection(
            baseline=baseline,
            feedback=_feedback(),
            coaching_text="Improve.",
            scope_policy=SCOPE_POLICY,
            requested_writer=WRITER,
            requested_evaluator=EVALUATOR,
            writer_client=_writer_client(),
            evaluator_client=_evaluator_client(0.4),
            resolve_locator=_resolve_locator,
            candidate_budget=1,
            optimizer=optimize,
            cancel_requested=lambda: next(cancel_checks, True),
        )


def test_reflection_rejects_candidate_budget_above_ten_before_inference():
    with pytest.raises(ValueError, match="between 1 and 10"):
        run_reflection(
            baseline=_bundle(**{"CLAUDE.md": "old"}),
            feedback=_feedback(),
            coaching_text="Improve.",
            scope_policy=SCOPE_POLICY,
            requested_writer=WRITER,
            requested_evaluator=EVALUATOR,
            writer_client=_writer_client(),
            evaluator_client=_evaluator_client(0.4),
            resolve_locator=_resolve_locator,
            candidate_budget=11,
            optimizer=lambda **_kwargs: pytest.fail("optimizer must not run"),
        )

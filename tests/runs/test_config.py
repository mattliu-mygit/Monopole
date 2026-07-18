import pytest
from pydantic import ValidationError

from weave_agent_signals.catalogs import build_model_catalog, build_rubric_catalog
from weave_agent_signals.run_config import (
    EFFECTIVE_RUN_CONFIG_SCHEMA_VERSION,
    MODEL_CATALOG_SCHEMA_VERSION,
    PIPELINE_VERSION,
    RUBRIC_CATALOG_SCHEMA_VERSION,
    EffectiveRunConfig,
    EvaluatedModelIdentity,
    JudgingContextPolicy,
    ModelDescriptor,
    RunConfig,
    resolve_run_config,
)


def test_judging_context_policy_uses_capacity_tiers_at_exact_threshold():
    policy = JudgingContextPolicy()

    assert policy.contract_version == "3"
    assert policy.capacity_reserve(200_000) == 18_000
    assert policy.capacity_reserve(200_001) == 18_000
    assert policy.raw_window_target(200_000) == 50_000
    assert policy.raw_window_target(200_001) == 128_000
    assert policy.raw_window_target(1_050_000) == 128_000
    assert policy.small_model_raw_target_tokens == 50_000
    assert policy.large_model_raw_target_tokens == 128_000
    assert policy.finding_max_tokens == 4_000
    assert "target_input_tokens" not in policy.model_dump()
    assert "token_estimator" not in policy.model_dump()


@pytest.mark.parametrize("model_limit", [True, 0, -1, 1.5])
def test_judging_context_policy_rejects_invalid_capacity(model_limit):
    with pytest.raises(ValueError, match="model_limit must be a positive integer"):
        JudgingContextPolicy().capacity_reserve(model_limit)
    with pytest.raises(ValueError, match="model_limit must be a positive integer"):
        JudgingContextPolicy().raw_window_target(model_limit)


def valid_request(**changes):
    values = {
        "model_catalog_version": "sha256:model",
        "rubric_catalog_version": "sha256:rubric",
        "judge_models": ("claude:claude-sonnet-5", "codex:gpt-5.6-sol"),
        "challenge_judge_models": ("codex:gpt-5.6-sol",),
        "proposal_model": "codex:gpt-5.6-sol",
        "proposal_evaluator_model": "claude:claude-sonnet-5",
        "rubrics": ("judge.tool_choice",),
        "candidate_budget": 3,
        "force": False,
    }
    values.update(changes)
    return values


def test_run_config_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        RunConfig.model_validate({**valid_request(), "unexpected": False})


@pytest.mark.parametrize(
    "judges",
    [
        ("judge-a",),
        ("judge-a", "judge-b"),
        ("judge-a", "judge-b", "judge-c"),
    ],
)
def test_run_config_accepts_one_through_three_judges(judges):
    request = RunConfig.model_validate(
        valid_request(judge_models=judges, challenge_judge_models=judges)
    )
    assert request.judge_models == judges
    assert request.challenge_judge_models == judges


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"judge_models": ("judge-a", "judge-a")}, "judge_models must be unique"),
        (
            {"challenge_judge_models": ("judge-a", "judge-a")},
            "challenge_judge_models must be unique",
        ),
        ({"judge_models": ("judge-a", "")}, "judge_models must contain nonblank"),
        ({"rubrics": ("judge.tool_choice", "judge.tool_choice")}, "rubrics must be unique"),
        ({"proposal_model": ""}, "proposal_model must be nonblank"),
        ({"judge_models": ()}, "one through three"),
        ({"judge_models": ("a", "b", "c", "d")}, "one through three"),
        ({"challenge_judge_models": ()}, "one through three"),
        ({"challenge_judge_models": ("a", "b", "c", "d")}, "one through three"),
    ],
)
def test_run_config_rejects_invalid_explicit_selection(changes, message):
    with pytest.raises(ValidationError, match=message):
        RunConfig.model_validate(valid_request(**changes))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_budget", True),
        ("candidate_budget", 0),
        ("candidate_budget", 11),
        ("force", 0),
    ],
)
def test_run_config_rejects_non_strict_controls(field, value):
    with pytest.raises(ValidationError):
        RunConfig.model_validate(valid_request(**{field: value}))


def test_run_config_is_frozen():
    request = RunConfig.model_validate(valid_request())

    with pytest.raises(ValidationError):
        request.force = True


def _all_executables(name: str) -> str:
    return f"/bin/{name}"


def _catalog_request(**changes):
    models = build_model_catalog(which=_all_executables)
    rubrics = build_rubric_catalog()
    request_values = {
        "model_catalog_version": models.catalog_version,
        "rubric_catalog_version": rubrics.catalog_version,
        "rubrics": (rubrics.rubrics[0].id,),
    }
    request_values.update(changes)
    request = RunConfig.model_validate(valid_request(**request_values))
    return models, rubrics, request


def test_resolve_run_config_pins_exact_order_descriptors_and_pipeline_version():
    models, rubrics, request = _catalog_request(
        judge_models=("codex:gpt-5.6-sol", "agy:gemini-3.1-pro-high"),
        rubrics=tuple(rubric.id for rubric in build_rubric_catalog().rubrics[:2]),
    )

    effective = resolve_run_config(
        request,
        model_catalog=models,
        rubric_catalog=rubrics,
    )

    assert effective.pipeline_version == PIPELINE_VERSION == "8"
    assert [judge.id for judge in effective.models.judges] == [
        "codex:gpt-5.6-sol",
        "agy:gemini-3.1-pro-high",
    ]
    assert [judge.position for judge in effective.models.judges] == [1, 2]
    assert [judge.id for judge in effective.models.challenge_judges] == ["codex:gpt-5.6-sol"]
    assert [judge.position for judge in effective.models.challenge_judges] == [1]
    assert effective.models.proposal_writer == models.model(request.proposal_model)
    assert effective.models.proposal_evaluator == models.model(request.proposal_evaluator_model)
    assert effective.rubrics == rubrics.rubrics[:2]

    restored = EffectiveRunConfig.model_validate_json(effective.model_dump_json())
    assert restored == effective


def test_capacity_metadata_bumps_model_and_effective_config_schema_versions():
    assert MODEL_CATALOG_SCHEMA_VERSION == "6"
    assert RUBRIC_CATALOG_SCHEMA_VERSION == "1"
    assert EFFECTIVE_RUN_CONFIG_SCHEMA_VERSION == "5"
    assert EffectiveRunConfig.model_fields["schema_version"].default == "5"


def test_resolve_run_config_defaults_empty_rubrics_to_exact_catalog_snapshot():
    models, rubrics, request = _catalog_request(rubrics=())

    effective = resolve_run_config(
        request,
        model_catalog=models,
        rubric_catalog=rubrics,
    )

    assert effective.rubrics == rubrics.rubrics


@pytest.mark.parametrize(
    ("version_field", "message"),
    [
        ("model_catalog_version", "stale model catalog version"),
        ("rubric_catalog_version", "stale rubric catalog version"),
    ],
)
def test_resolve_run_config_rejects_stale_catalog_versions(version_field, message):
    models, rubrics, request = _catalog_request()
    stale = request.model_copy(update={version_field: "sha256:stale"})

    with pytest.raises(ValueError, match=message):
        resolve_run_config(stale, model_catalog=models, rubric_catalog=rubrics)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"judge_models": ("missing",)}, "judge model missing is unknown or unavailable"),
        (
            {"challenge_judge_models": ("missing",)},
            "challenge judge model missing is unknown or unavailable",
        ),
        (
            {"proposal_model": "openai:gpt-4o"},
            "unknown or unavailable proposal model",
        ),
        ({"proposal_model": "missing"}, "unknown or unavailable proposal model"),
        ({"proposal_evaluator_model": "missing"}, "unknown or unavailable"),
        ({"rubrics": ("judge.missing",)}, "unknown rubric"),
    ],
)
def test_resolve_run_config_rejects_wrong_role_or_unknown_ids(changes, message):
    models, rubrics, request = _catalog_request(**changes)

    with pytest.raises(ValueError, match=message):
        resolve_run_config(request, model_catalog=models, rubric_catalog=rubrics)


def test_resolve_run_config_rejects_model_without_requested_role():
    models, rubrics, request = _catalog_request(
        judge_models=("test:writer-only", "codex:gpt-5.6-sol"),
    )
    writer_only = ModelDescriptor(
        id="test:writer-only",
        label="Writer only",
        provider="test",
        provider_model="writer-only",
        family="example",
        supported_roles=("proposal_writer",),
    )
    restricted_catalog = models.model_copy(
        update={"available_models": (*models.available_models, writer_only)}
    )

    with pytest.raises(ValueError, match="does not support judging"):
        resolve_run_config(
            request,
            model_catalog=restricted_catalog,
            rubric_catalog=rubrics,
        )


def test_resolve_run_config_emits_evaluated_family_warning_without_changing_order():
    models, rubrics, request = _catalog_request(
        judge_models=("claude:claude-sonnet-5", "codex:gpt-5.6-sol"),
    )

    effective = resolve_run_config(
        request,
        model_catalog=models,
        rubric_catalog=rubrics,
        evaluated_models=(EvaluatedModelIdentity(id="provider-resolved-model", family="openai"),),
    )

    assert [judge.id for judge in effective.models.judges] == list(request.judge_models)
    warning = next(
        warning
        for warning in effective.selection_warnings
        if warning.code == "judge_evaluated_family_overlap"
    )
    assert warning.affected_roles == ("judge_2", "evaluated_agent")
    assert warning.selected_model_ids == ("codex:gpt-5.6-sol", "provider-resolved-model")
    assert warning.compared_families == ("openai",)


def test_resolve_run_config_warns_for_challenge_panel_family_bias():
    models, rubrics, request = _catalog_request(
        challenge_judge_models=(
            "claude:claude-sonnet-5",
            "claude:claude-haiku-4-5",
        ),
    )

    effective = resolve_run_config(
        request,
        model_catalog=models,
        rubric_catalog=rubrics,
        evaluated_models=(EvaluatedModelIdentity(id="evaluated-claude", family="anthropic"),),
    )

    warnings = {warning.code: warning for warning in effective.selection_warnings}
    assert warnings["challenge_judge_family_overlap"].affected_roles == (
        "challenge_judge_1",
        "challenge_judge_2",
    )
    assert warnings["low_challenge_judge_family_diversity"].selected_model_ids == (
        "claude:claude-sonnet-5",
        "claude:claude-haiku-4-5",
    )
    assert warnings["challenge_judge_evaluated_family_overlap"].affected_roles == (
        "challenge_judge_1",
        "challenge_judge_2",
        "evaluated_agent",
    )

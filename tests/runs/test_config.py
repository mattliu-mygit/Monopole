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
    ModelDescriptor,
    RunConfig,
    resolve_run_config,
)


def valid_request(**changes):
    values = {
        "model_catalog_version": "sha256:model",
        "rubric_catalog_version": "sha256:rubric",
        "judge_backend": "cli",
        "review_depth": "selective",
        "judge_models": ("claude-sonnet-5", "gpt-5.6-sol"),
        "second_opinion_margin": 0.10,
        "proposal_model": "gpt-5.6-sol",
        "proposal_evaluator_model": "claude-sonnet-5",
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
    ("depth", "judges", "margin"),
    [
        ("primary", ("judge-a",), None),
        ("selective", ("judge-a", "judge-b"), 0.10),
        ("selective", ("judge-a", "judge-b", "judge-c"), 0.10),
        ("full_panel", ("judge-a", "judge-b", "judge-c"), None),
    ],
)
def test_run_config_accepts_exact_review_cardinality(depth, judges, margin):
    request = RunConfig.model_validate(
        valid_request(
            review_depth=depth,
            judge_models=judges,
            second_opinion_margin=margin,
        )
    )
    assert request.judge_models == judges


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"judge_models": ("judge-a", "judge-a")}, "judge_models must be unique"),
        ({"judge_models": ("judge-a", "")}, "judge_models must contain nonblank"),
        ({"rubrics": ("judge.tool_choice", "judge.tool_choice")}, "rubrics must be unique"),
        ({"proposal_model": ""}, "proposal_model must be nonblank"),
        (
            {
                "review_depth": "primary",
                "judge_models": ("judge-a", "judge-b"),
                "second_opinion_margin": None,
            },
            "primary review requires exactly 1 judge",
        ),
        (
            {
                "review_depth": "selective",
                "judge_models": ("judge-a", "judge-b"),
                "second_opinion_margin": None,
            },
            "selective review requires second_opinion_margin",
        ),
        ({"second_opinion_margin": 0.51}, "between 0 and 0.5"),
        (
            {
                "review_depth": "full_panel",
                "judge_models": ("judge-a", "judge-b", "judge-c"),
                "second_opinion_margin": 0.1,
            },
            "must be null",
        ),
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
        judge_models=("gpt-5.6-sol", "claude-sonnet-5"),
        rubrics=tuple(rubric.id for rubric in build_rubric_catalog().rubrics[:2]),
    )

    effective = resolve_run_config(
        request,
        model_catalog=models,
        rubric_catalog=rubrics,
    )

    assert effective.pipeline_version == PIPELINE_VERSION == "3"
    assert [judge.id for judge in effective.models.judges] == [
        "gpt-5.6-sol",
        "claude-sonnet-5",
    ]
    assert [judge.position for judge in effective.models.judges] == [1, 2]
    assert effective.models.proposal_writer == models.model(request.proposal_model)
    assert effective.models.proposal_evaluator == models.model(request.proposal_evaluator_model)
    assert effective.rubrics == rubrics.rubrics[:2]

    restored = EffectiveRunConfig.model_validate_json(effective.model_dump_json())
    assert restored == effective


def test_pipeline_bump_keeps_shape_only_config_schema_versions_unchanged():
    assert MODEL_CATALOG_SCHEMA_VERSION == "1"
    assert RUBRIC_CATALOG_SCHEMA_VERSION == "1"
    assert EFFECTIVE_RUN_CONFIG_SCHEMA_VERSION == "1"
    assert EffectiveRunConfig.model_fields["schema_version"].default == "1"


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
        ({"judge_backend": "missing"}, "unknown judge backend"),
        ({"judge_models": ("gpt-oss-20b", "claude-sonnet-5")}, "does not support judging on cli"),
        ({"proposal_model": "gpt-oss-20b"}, "unknown or unavailable proposal model"),
        ({"proposal_model": "missing"}, "unknown or unavailable proposal model"),
        ({"proposal_evaluator_model": "gpt-4o"}, "unknown or unavailable for cli"),
        ({"proposal_evaluator_model": "missing"}, "unknown or unavailable for cli"),
        ({"rubrics": ("judge.missing",)}, "unknown rubric"),
    ],
)
def test_resolve_run_config_rejects_wrong_backend_role_or_unknown_ids(changes, message):
    models, rubrics, request = _catalog_request(**changes)

    with pytest.raises(ValueError, match=message):
        resolve_run_config(request, model_catalog=models, rubric_catalog=rubrics)


def test_resolve_run_config_rejects_model_without_requested_role():
    models, rubrics, request = _catalog_request(
        judge_models=("writer-only", "gpt-5.6-sol"),
    )
    writer_only = ModelDescriptor(
        id="writer-only",
        label="Writer only",
        family="example",
        backend="cli",
        supported_roles=("proposal_writer",),
    )
    cli = models.backend("cli")
    restricted_cli = cli.model_copy(
        update={"available_models": (*cli.available_models, writer_only)}
    )
    restricted_catalog = models.model_copy(
        update={"judge_backends": {**models.judge_backends, "cli": restricted_cli}}
    )

    with pytest.raises(ValueError, match="does not support judging on cli"):
        resolve_run_config(
            request,
            model_catalog=restricted_catalog,
            rubric_catalog=rubrics,
        )


def test_resolve_run_config_emits_evaluated_family_warning_without_changing_order():
    models, rubrics, request = _catalog_request(
        judge_models=("claude-sonnet-5", "gpt-5.6-sol"),
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
    assert warning.selected_model_ids == ("gpt-5.6-sol", "provider-resolved-model")
    assert warning.compared_families == ("openai",)

from __future__ import annotations

import re
from dataclasses import replace

import pytest
from pydantic import ValidationError

import weave_agent_signals.catalogs as catalogs_module
from weave_agent_signals.catalogs import (
    build_model_catalog,
    build_rubric_catalog,
)
from weave_agent_signals.judges.rubrics import SESSION_RUBRICS
from weave_agent_signals.run_config import ModelDescriptor


def _all_executables(name: str) -> str:
    return f"/bin/{name}"


def _assert_sha256_version(value: str) -> None:
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", value)


def test_model_catalog_is_stable_role_oriented_and_immutable():
    first = build_model_catalog(which=_all_executables)
    second = build_model_catalog(which=_all_executables)

    assert first.catalog_version == second.catalog_version
    _assert_sha256_version(first.catalog_version)
    data = first.model_dump(mode="json")
    assert set(data) == {
        "catalog_version",
        "proposal",
        "recommended_judge_backend",
        "judge_backends",
    }
    assert data["recommended_judge_backend"] == "cli"

    cli = data["judge_backends"]["cli"]
    assert len(cli["recommended_judges"]) == 3
    by_id = {model["id"]: model for model in cli["available_models"]}
    assert len({by_id[model_id]["family"] for model_id in cli["recommended_judges"]}) == 2
    assert set(cli["proposal_evaluator_preferences"]) == {
        model["id"]
        for model in cli["available_models"]
        if "proposal_evaluator" in model["supported_roles"]
    }
    for model in data["proposal"]["available_models"]:
        assert set(model) == {
            "id",
            "label",
            "family",
            "backend",
            "supported_roles",
            "max_input_tokens",
        }
        assert "proposal_writer" in model["supported_roles"]

    with pytest.raises(ValidationError):
        first.catalog_version = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        first.judge_backends["cli"] = first.judge_backends["cli"]  # type: ignore[index]


def test_model_catalog_version_tracks_local_availability():
    all_available = build_model_catalog(which=_all_executables)
    codex_only = build_model_catalog(which=lambda name: "/bin/codex" if name == "codex" else None)
    none_available = build_model_catalog(which=lambda _name: None)

    versions = {
        all_available.catalog_version,
        codex_only.catalog_version,
        none_available.catalog_version,
    }
    assert len(versions) == 3
    proposal_models = codex_only.model_dump(mode="json")["proposal"]["available_models"]
    assert [model["id"] for model in proposal_models] == ["gpt-5.6-sol"]
    assert none_available.model_dump(mode="json")["proposal"] == {
        "available_models": [],
        "recommended_model": None,
    }
    assert (
        all_available.model_dump(mode="json")["judge_backends"]["wandb"]["available_models"]
        == none_available.model_dump(mode="json")["judge_backends"]["wandb"]["available_models"]
    )


def test_every_model_declares_enough_input_context():
    catalog = build_model_catalog(which=lambda _name: "/usr/bin/model")
    models = [
        *catalog.proposal.available_models,
        *(
            model
            for backend in catalog.judge_backends.values()
            for model in backend.available_models
        ),
    ]
    assert models
    assert all(model.max_input_tokens >= 128_000 for model in models)


def test_model_catalog_version_tracks_recommendations_and_evaluator_order():
    base = build_model_catalog(which=_all_executables)
    cli = base.model_dump(mode="json")["judge_backends"]["cli"]

    recommendation_changed = build_model_catalog(
        which=_all_executables,
        recommended_judges={"cli": tuple(reversed(cli["recommended_judges"]))},
    )
    evaluator_changed = build_model_catalog(
        which=_all_executables,
        evaluator_preferences={"cli": tuple(reversed(cli["proposal_evaluator_preferences"]))},
    )

    assert recommendation_changed.catalog_version != base.catalog_version
    assert evaluator_changed.catalog_version != base.catalog_version


def test_rubric_catalog_is_stable_and_descriptors_have_exact_shape():
    first = build_rubric_catalog()
    second = build_rubric_catalog()

    assert first.catalog_version == second.catalog_version
    _assert_sha256_version(first.catalog_version)
    data = first.model_dump(mode="json")
    assert set(data) == {"catalog_version", "rubrics"}
    assert {descriptor["id"] for descriptor in data["rubrics"]} == set(SESSION_RUBRICS)
    for descriptor in data["rubrics"]:
        assert set(descriptor) == {
            "id",
            "label",
            "evaluation_unit",
            "version",
            "content_digest",
            "pass_threshold",
        }
        assert descriptor["evaluation_unit"] == "session"
        assert descriptor["version"] == "v4"
        _assert_sha256_version(descriptor["content_digest"])


def test_rubric_and_catalog_digests_track_content_and_version_deterministically():
    rubrics = list(SESSION_RUBRICS.values())
    base = build_rubric_catalog(rubrics=rubrics)
    repeated = build_rubric_catalog(rubrics=rubrics)
    threshold_changed = build_rubric_catalog(
        rubrics=[replace(rubrics[0], threshold=0.6), *rubrics[1:]]
    )
    version_changed = build_rubric_catalog(
        rubrics=[replace(rubrics[0], version="v1"), *rubrics[1:]]
    )
    prompt_changed = build_rubric_catalog(
        rubrics=[
            replace(rubrics[0], system_prompt=rubrics[0].system_prompt + "\nMaterial change."),
            *rubrics[1:],
        ]
    )

    assert repeated == base
    assert threshold_changed.catalog_version != base.catalog_version
    assert version_changed.catalog_version != base.catalog_version
    assert prompt_changed.catalog_version != base.catalog_version
    assert version_changed.rubrics[0].content_digest != base.rubrics[0].content_digest
    assert prompt_changed.rubrics[0].content_digest != base.rubrics[0].content_digest


def test_descriptor_membership_is_not_limited_by_preference_registries(monkeypatch):
    added = ModelDescriptor(
        id="deepseek-r1",
        label="DeepSeek R1",
        family="deepseek",
        backend="wandb",
        supported_roles=("judge", "proposal_evaluator"),
    )
    monkeypatch.setattr(
        catalogs_module,
        "_MODEL_DESCRIPTORS",
        (*catalogs_module._MODEL_DESCRIPTORS, added),
    )

    models = build_model_catalog(which=_all_executables)
    wandb = models.model_dump(mode="json")["judge_backends"]["wandb"]
    available_ids = [descriptor["id"] for descriptor in wandb["available_models"]]

    assert available_ids[-1] == added.id
    assert wandb["proposal_evaluator_preferences"][-1] == added.id


def test_model_descriptor_serializes_only_its_public_fields():
    descriptor = ModelDescriptor(
        id="example",
        label="Example",
        family="example-family",
        backend="example-backend",
        supported_roles=("judge",),
    )

    assert descriptor.model_dump(mode="json") == {
        "id": "example",
        "label": "Example",
        "family": "example-family",
        "backend": "example-backend",
        "supported_roles": ["judge"],
        "max_input_tokens": 128_000,
    }


@pytest.mark.parametrize("value", [0, -1, True, "128000"])
def test_model_descriptor_rejects_invalid_input_token_limits(value):
    with pytest.raises(ValidationError):
        ModelDescriptor(
            id="example",
            label="Example",
            family="example-family",
            backend="example-backend",
            supported_roles=("judge",),
            max_input_tokens=value,
        )

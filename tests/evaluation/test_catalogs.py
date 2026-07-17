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
        "available_models",
        "recommended_proposal_model",
        "recommended_judges",
        "recommended_challenge_judges",
        "proposal_evaluator_preferences",
    }
    by_id = {model["id"]: model for model in data["available_models"]}
    assert all(model["id"].startswith(f"{model['provider']}:") for model in by_id.values())
    assert len(data["recommended_judges"]) == 3
    assert all(model_id in by_id for model_id in data["recommended_judges"])
    assert data["recommended_challenge_judges"] == [data["recommended_judges"][0]]
    assert set(data["proposal_evaluator_preferences"]) == {
        model["id"]
        for model in data["available_models"]
        if "proposal_evaluator" in model["supported_roles"]
    }
    for model in data["available_models"]:
        assert set(model) == {
            "id",
            "label",
            "provider",
            "provider_model",
            "family",
            "supported_roles",
            "max_input_tokens",
            "token_counter",
        }

    with pytest.raises(ValidationError):
        first.catalog_version = "changed"  # type: ignore[misc]


def test_model_catalog_is_one_provider_qualified_source_for_every_role():
    catalog = build_model_catalog(which=_all_executables)
    data = catalog.model_dump(mode="json")

    assert set(data) == {
        "catalog_version",
        "available_models",
        "recommended_proposal_model",
        "recommended_judges",
        "recommended_challenge_judges",
        "proposal_evaluator_preferences",
    }
    by_id = {model["id"]: model for model in data["available_models"]}
    assert "agy:gemini-3.5-flash-high" in by_id
    assert "agy:gpt-oss-120b-medium" in by_id
    assert by_id["agy:gemini-3.5-flash-high"]["provider"] == "agy"
    assert by_id["agy:gemini-3.5-flash-high"]["provider_model"] == ("Gemini 3.5 Flash (High)")
    assert by_id["agy:gemini-3.5-flash-high"]["family"] == "google"
    assert by_id["agy:gpt-oss-120b-medium"]["family"] == "openai"
    for recommendation in (
        data["recommended_proposal_model"],
        *data["recommended_judges"],
        *data["recommended_challenge_judges"],
        *data["proposal_evaluator_preferences"],
    ):
        assert recommendation in by_id


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
    codex_models = [
        model
        for model in codex_only.model_dump(mode="json")["available_models"]
        if model["provider"] == "codex"
    ]
    assert [model["id"] for model in codex_models] == [
        "codex:gpt-5.6-sol",
        "codex:gpt-5.6-terra",
        "codex:gpt-5.6-luna",
        "codex:gpt-5.5",
        "codex:gpt-5.4",
        "codex:gpt-5.4-mini",
    ]
    assert none_available.recommended_proposal_model is None
    assert {model.provider for model in none_available.available_models} == {"wandb", "openai"}


def test_codex_models_are_available_for_every_run_role():
    catalog = build_model_catalog(which=lambda name: "/bin/codex" if name == "codex" else None)
    expected_ids = {
        "codex:gpt-5.6-sol",
        "codex:gpt-5.6-terra",
        "codex:gpt-5.6-luna",
        "codex:gpt-5.5",
        "codex:gpt-5.4",
        "codex:gpt-5.4-mini",
    }

    by_id = {model.id: model for model in catalog.available_models}

    assert expected_ids <= set(by_id)
    for model_id in expected_ids:
        assert by_id[model_id].provider == "codex"
        assert set(by_id[model_id].supported_roles) == {
            "proposal_writer",
            "judge",
            "proposal_evaluator",
        }
        assert by_id[model_id].max_input_tokens == 272_000
        assert by_id[model_id].token_counter == "o200k_base"


def test_every_model_declares_exact_input_capacity_and_token_counter():
    catalog = build_model_catalog(which=lambda _name: "/usr/bin/model")
    models = {model.id: model for model in catalog.available_models}

    assert {
        model_id: (model.max_input_tokens, model.token_counter)
        for model_id, model in models.items()
    } == {
        "codex:gpt-5.6-sol": (272_000, "o200k_base"),
        "codex:gpt-5.6-terra": (272_000, "o200k_base"),
        "codex:gpt-5.6-luna": (272_000, "o200k_base"),
        "codex:gpt-5.5": (272_000, "o200k_base"),
        "codex:gpt-5.4": (272_000, "o200k_base"),
        "codex:gpt-5.4-mini": (272_000, "o200k_base"),
        "claude:claude-sonnet-5": (200_000, "utf8_bytes_div_3"),
        "claude:claude-haiku-4-5": (200_000, "utf8_bytes_div_3"),
        "agy:gemini-3.5-flash-medium": (1_048_576, "utf8_bytes_div_3"),
        "agy:gemini-3.5-flash-high": (1_048_576, "utf8_bytes_div_3"),
        "agy:gemini-3.5-flash-low": (1_048_576, "utf8_bytes_div_3"),
        "agy:gemini-3.1-pro-low": (1_048_576, "utf8_bytes_div_3"),
        "agy:gemini-3.1-pro-high": (1_048_576, "utf8_bytes_div_3"),
        "agy:gpt-oss-120b-medium": (131_072, "o200k_harmony"),
        "wandb:gpt-oss-20b": (131_072, "o200k_harmony"),
        "wandb:gpt-oss-120b": (131_072, "o200k_harmony"),
        "wandb:Llama-3.1-8B": (131_072, "utf8_bytes_div_3"),
        "wandb:granite-4.1-8b": (131_072, "utf8_bytes_div_3"),
        "openai:gpt-4o": (128_000, "o200k_base"),
        "openai:gpt-4o-mini": (128_000, "o200k_base"),
    }


def test_model_catalog_version_tracks_recommendations_and_evaluator_order():
    base = build_model_catalog(which=_all_executables)

    recommendation_changed = build_model_catalog(
        which=_all_executables,
        recommended_judges=tuple(reversed(base.recommended_judges)),
    )
    challenge_recommendation_changed = build_model_catalog(
        which=_all_executables,
        recommended_challenge_judges=(base.recommended_judges[1],),
    )
    evaluator_changed = build_model_catalog(
        which=_all_executables,
        evaluator_preferences=tuple(reversed(base.proposal_evaluator_preferences)),
    )

    assert recommendation_changed.catalog_version != base.catalog_version
    assert challenge_recommendation_changed.catalog_version != base.catalog_version
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
        id="wandb:deepseek-r1",
        label="DeepSeek R1",
        provider="wandb",
        provider_model="deepseek-r1",
        family="deepseek",
        supported_roles=("judge", "proposal_evaluator"),
    )
    monkeypatch.setattr(
        catalogs_module,
        "_MODEL_DESCRIPTORS",
        (*catalogs_module._MODEL_DESCRIPTORS, added),
    )

    models = build_model_catalog(which=_all_executables)
    available_ids = [descriptor.id for descriptor in models.available_models]

    assert added.id in available_ids
    assert models.proposal_evaluator_preferences[-1] == added.id


def test_model_descriptor_serializes_only_its_public_fields():
    descriptor = ModelDescriptor(
        id="example",
        label="Example",
        provider="example-provider",
        provider_model="example-model",
        family="example-family",
        supported_roles=("judge",),
    )

    assert descriptor.model_dump(mode="json") == {
        "id": "example",
        "label": "Example",
        "provider": "example-provider",
        "provider_model": "example-model",
        "family": "example-family",
        "supported_roles": ["judge"],
        "max_input_tokens": 128_000,
        "token_counter": "utf8_bytes_div_3",
    }


@pytest.mark.parametrize("value", [0, -1, True, "128000"])
def test_model_descriptor_rejects_invalid_input_token_limits(value):
    with pytest.raises(ValidationError):
        ModelDescriptor(
            id="example",
            label="Example",
            provider="example-provider",
            provider_model="example-model",
            family="example-family",
            supported_roles=("judge",),
            max_input_tokens=value,
        )


def test_model_descriptor_rejects_unknown_token_counter():
    with pytest.raises(ValidationError):
        ModelDescriptor(
            id="example",
            label="Example",
            provider="example-provider",
            provider_model="example-model",
            family="example-family",
            supported_roles=("judge",),
            token_counter="family-inferred",
        )

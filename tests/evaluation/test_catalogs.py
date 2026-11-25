from __future__ import annotations

import itertools
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


EXPECTED_WANDB_MODELS = {
    "MiniMaxAI/MiniMax-M3": 262_000,
    "Qwen/Qwen3.6-35B-A3B": 262_000,
    "Qwen/Qwen3.8-27B": 262_000,
    "deepseek-ai/DeepSeek-V3.1": 161_000,
    "deepseek-ai/DeepSeek-V4-Flash-0731": 262_000,
    "deepseek-ai/DeepSeek-V4-Pro-0813": 1_049_000,
    "deepseek-ai/DeepSeek-V4.1-Flash": 1_049_000,
    "deepseek-ai/DeepSeek-V4.1-Flash-1017": 1_000_000,
    "google/gemma-4-26B-A4B-it": 262_000,
    "google/gemma-4-31B-it": 262_000,
    "ibm-granite/granite-4.2-8b": 131_072,
    "meta-llama/Llama-3.1-8B-Instruct": 131_072,
    "meta-llama/Llama-3.3-70B-Instruct": 128_000,
    "moonshotai/Kimi-K2.6": 262_000,
    "moonshotai/Kimi-K2.7-Code": 262_000,
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B": 262_000,
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B": 262_000,
    "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B": 1_049_000,
    "openai/gpt-oss-120b": 131_072,
    "openai/gpt-oss-20b": 131_072,
    "reflection/Beam-1.0-501B-A23B": 1_049_000,
    "zai-org/GLM-5.2": 1_049_000,
    "zai-org/GLM-5.3": 1_049_000,
    "zai-org/GLM-5.3-Flash": 1_049_000,
}

MID_BUDGETS = {
    "question_budget_tokens": 4_000,
    "window_budget_tokens": 32_000,
    "whole_budget_tokens": 32_000,
    "merge_budget_tokens": 32_000,
}


def test_model_catalog_is_stable_role_oriented_and_immutable():
    first = build_model_catalog(which=_all_executables)
    second = build_model_catalog(which=_all_executables)

    assert first.catalog_version == second.catalog_version
    _assert_sha256_version(first.catalog_version)
    data = first.model_dump(mode="json")
    assert set(data) == {
        "catalog_version",
        "available_models",
        "judging_context",
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
            "question_budget_tokens",
            "window_budget_tokens",
            "whole_budget_tokens",
            "merge_budget_tokens",
        }

    with pytest.raises(ValidationError):
        first.catalog_version = "changed"  # type: ignore[misc]


def test_model_catalog_is_one_provider_qualified_source_for_every_role():
    catalog = build_model_catalog(which=_all_executables)
    data = catalog.model_dump(mode="json")

    assert set(data) == {
        "catalog_version",
        "available_models",
        "judging_context",
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


def test_wandb_descriptors_match_the_recorded_live_list():
    catalog = build_model_catalog(which=lambda _name: None)

    assert {
        model.provider_model for model in catalog.available_models if model.provider == "wandb"
    } == set(EXPECTED_WANDB_MODELS)


def test_wandb_models_pin_context_size_roles_and_token_counter():
    catalog = build_model_catalog(which=lambda _name: "/usr/bin/model")
    models = [model for model in catalog.available_models if model.provider == "wandb"]

    for model in models:
        assert model.max_input_tokens == EXPECTED_WANDB_MODELS[model.provider_model]
        assert set(model.supported_roles) == {
            "proposal_writer",
            "judge",
            "proposal_evaluator",
        }
        assert model.token_counter == (
            "o200k_harmony"
            if model.provider_model.startswith("openai/gpt-oss-")
            else "utf8_bytes_div_3"
        )


def test_recommendations_follow_each_role_preference_order():
    catalog = build_model_catalog(which=lambda _name: "/usr/bin/model")
    by_id = {model.id: model for model in catalog.available_models}

    assert catalog.recommended_proposal_model == "wandb:gpt-oss-120b"
    assert catalog.recommended_judges == (
        "claude:claude-sonnet-5",
        "codex:gpt-5.6-sol",
        "wandb:DeepSeek-V4.1-Flash",
    )
    assert catalog.recommended_challenge_judges == ("claude:claude-sonnet-5",)
    assert catalog.proposal_evaluator_preferences[:4] == (
        "wandb:gpt-oss-120b",
        "wandb:Llama-3.1-8B",
        "wandb:granite-4.2-8b",
        "wandb:gpt-oss-20b",
    )
    for model_id in catalog.proposal_evaluator_preferences[:4]:
        assert set(by_id[model_id].supported_roles) == {
            "proposal_writer",
            "judge",
            "proposal_evaluator",
        }


@pytest.mark.parametrize(
    "installed",
    [
        subset
        for size in range(4)
        for subset in itertools.combinations(("claude", "codex", "agy"), size)
    ],
    ids=lambda installed: "+".join(installed) or "none",
)
def test_recommended_judges_span_three_families(installed):
    catalog = build_model_catalog(which=lambda name: f"/bin/{name}" if name in installed else None)
    families = [catalog.model(model_id).family for model_id in catalog.recommended_judges]

    assert len(set(families)) == len(families) == 3


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
        "codex:gpt-5.4-mini",
    ]
    assert none_available.recommended_proposal_model == "wandb:gpt-oss-120b"
    assert {model.provider for model in none_available.available_models} == {"wandb", "openai"}


def test_codex_models_are_available_for_every_run_role():
    catalog = build_model_catalog(which=lambda name: "/bin/codex" if name == "codex" else None)
    expected_ids = {
        "codex:gpt-5.6-sol",
        "codex:gpt-5.6-terra",
        "codex:gpt-5.6-luna",
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


def test_every_non_wandb_model_declares_exact_input_capacity_and_token_counter():
    catalog = build_model_catalog(which=lambda _name: "/usr/bin/model")
    models = {model.id: model for model in catalog.available_models if model.provider != "wandb"}

    expected = {
        model_id: (model.max_input_tokens, model.token_counter)
        for model_id, model in models.items()
    }
    assert expected == {
        "codex:gpt-5.6-sol": (272_000, "o200k_base"),
        "codex:gpt-5.6-terra": (272_000, "o200k_base"),
        "codex:gpt-5.6-luna": (272_000, "o200k_base"),
        "codex:gpt-5.4-mini": (272_000, "o200k_base"),
        "claude:claude-sonnet-5": (1_000_000, "utf8_bytes_div_3"),
        "claude:claude-haiku-4-5": (200_000, "utf8_bytes_div_3"),
        "devin:claude-opus-5-5-medium": (1_000_000, "utf8_bytes_div_3"),
        "devin:claude-fable-5-1-medium": (1_000_000, "utf8_bytes_div_3"),
        "agy:gemini-3.5-flash-medium": (1_048_576, "utf8_bytes_div_3"),
        "agy:gemini-3.5-flash-high": (1_048_576, "utf8_bytes_div_3"),
        "agy:gemini-3.5-flash-low": (1_048_576, "utf8_bytes_div_3"),
        "agy:gemini-3.1-pro-low": (1_048_576, "utf8_bytes_div_3"),
        "agy:gemini-3.1-pro-high": (1_048_576, "utf8_bytes_div_3"),
        "agy:gpt-oss-120b-medium": (131_072, "o200k_harmony"),
        "openai:gpt-4o": (128_000, "o200k_base"),
        "openai:gpt-4o-mini": (128_000, "o200k_base"),
    }


def test_claude_sonnet_5_has_a_1m_window():
    sonnet = build_model_catalog(which=_all_executables).model("claude:claude-sonnet-5")

    assert sonnet.max_input_tokens == 1_000_000


def test_retired_codex_models_are_gone():
    catalog = build_model_catalog(which=_all_executables)

    assert not {"codex:gpt-5.4", "codex:gpt-5.5"} & {model.id for model in catalog.available_models}


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


def test_changing_a_budget_changes_the_catalog_version(monkeypatch):
    base = build_model_catalog(which=_all_executables)
    changed = tuple(
        ModelDescriptor.model_validate({**descriptor.model_dump(), "window_budget_tokens": 100_000})
        if descriptor.id == "claude:claude-sonnet-5"
        else descriptor
        for descriptor in catalogs_module._MODEL_DESCRIPTORS
    )
    monkeypatch.setattr(catalogs_module, "_MODEL_DESCRIPTORS", changed)

    assert build_model_catalog(which=_all_executables).catalog_version != base.catalog_version


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
        **MID_BUDGETS,
    )
    monkeypatch.setattr(
        catalogs_module,
        "_MODEL_DESCRIPTORS",
        (*catalogs_module._MODEL_DESCRIPTORS, added),
    )

    models = build_model_catalog(which=_all_executables)
    available_ids = [descriptor.id for descriptor in models.available_models]

    assert added.id in available_ids
    assert added.id in models.proposal_evaluator_preferences


def test_model_descriptor_serializes_only_its_public_fields():
    descriptor = ModelDescriptor(
        id="example",
        label="Example",
        provider="example-provider",
        provider_model="example-model",
        family="example-family",
        supported_roles=("judge",),
        **MID_BUDGETS,
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
        "question_budget_tokens": 4_000,
        "window_budget_tokens": 32_000,
        "whole_budget_tokens": 32_000,
        "merge_budget_tokens": 32_000,
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
            **MID_BUDGETS,
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
            **MID_BUDGETS,
        )


@pytest.mark.parametrize(
    ("question", "window", "whole", "merge", "message"),
    [
        pytest.param(
            8_000,
            4_000,
            8_000,
            8_000,
            "question_budget_tokens must not exceed window_budget_tokens",
            id="question-above-window",
        ),
        pytest.param(
            8_000,
            8_000,
            8_000,
            4_000,
            "question_budget_tokens must not exceed merge_budget_tokens",
            id="question-above-merge",
        ),
        pytest.param(
            2_000,
            8_000,
            4_000,
            4_000,
            "window_budget_tokens must not exceed whole_budget_tokens",
            id="window-above-whole",
        ),
        # With no window budget, the question budget is compared with the whole budget.
        pytest.param(
            8_000,
            None,
            4_000,
            8_000,
            "question_budget_tokens must not exceed whole_budget_tokens",
            id="no-window-question-above-whole",
        ),
    ],
)
def test_budgets_out_of_order_are_rejected(question, window, whole, merge, message):
    with pytest.raises(ValueError, match=f"model example {message}"):
        ModelDescriptor(
            id="example",
            label="Example",
            provider="example-provider",
            provider_model="example-model",
            family="example-family",
            supported_roles=("judge",),
            question_budget_tokens=question,
            window_budget_tokens=window,
            whole_budget_tokens=whole,
            merge_budget_tokens=merge,
        )


@pytest.mark.parametrize(
    ("field", "question", "window", "whole", "merge"),
    [
        ("question_budget_tokens", 23_000, None, None, 23_000),
        ("window_budget_tokens", 2_000, 30_000, None, 16_000),
        ("whole_budget_tokens", 2_000, 8_000, 30_000, 16_000),
        ("merge_budget_tokens", 2_000, 8_000, 8_000, 30_000),
    ],
)
def test_budget_larger_than_usable_context_is_rejected(field, question, window, whole, merge):
    # A 32,800-token model reserves 9,840 tokens, which leaves 22,960 for evidence.
    with pytest.raises(
        ValueError,
        match=f"model example {field} must not exceed the usable context of 22960",
    ):
        ModelDescriptor(
            id="example",
            label="Example",
            provider="example-provider",
            provider_model="example-model",
            family="example-family",
            supported_roles=("judge",),
            max_input_tokens=32_800,
            question_budget_tokens=question,
            window_budget_tokens=window,
            whole_budget_tokens=whole,
            merge_budget_tokens=merge,
        )


def test_budgets_equal_to_usable_context_are_accepted():
    descriptor = ModelDescriptor(
        id="example",
        label="Example",
        provider="example-provider",
        provider_model="example-model",
        family="example-family",
        supported_roles=("judge",),
        max_input_tokens=32_800,
        question_budget_tokens=22_960,
        window_budget_tokens=22_960,
        whole_budget_tokens=22_960,
        merge_budget_tokens=22_960,
    )

    assert descriptor.merge_budget_tokens == 22_960


def test_deepseek_v4_pro_caps_its_window_and_whole_budgets_at_96k():
    deepseek = build_model_catalog(which=lambda _name: None).model("wandb:DeepSeek-V4-Pro-0813")

    assert (
        deepseek.question_budget_tokens,
        deepseek.window_budget_tokens,
        deepseek.whole_budget_tokens,
        deepseek.merge_budget_tokens,
    ) == (8_000, 96_000, 96_000, 128_000)


@pytest.mark.parametrize(("budget_class", "cap"), [("frontier", 128_001), ("small_dense", 1_000)])
def test_raw_budget_cap_must_fit_its_budget_class(budget_class, cap):
    with pytest.raises(
        ValueError,
        match=(
            f"model wandb:example raw_budget_cap_tokens must not exceed the {budget_class} "
            "window budget"
        ),
    ):
        catalogs_module._wandb_model(
            "wandb:example",
            "Example",
            "example/example",
            "example",
            1_000_000,
            budget_class,
            raw_budget_cap_tokens=cap,
        )

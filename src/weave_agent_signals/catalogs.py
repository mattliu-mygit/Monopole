"""Versioned model and rubric catalogs for evaluation-run configuration.

Catalogs are the user-visible selection boundary. Their versions cover every
descriptor and recommendation returned by the API, including detected local
CLI availability, so a saved configuration can fail closed when that boundary
changes before a run starts.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from types import MappingProxyType

from weave_agent_signals.judges.families import model_family
from weave_agent_signals.judges.rubrics import SESSION_RUBRICS, Rubric
from weave_agent_signals.judges.tokens import TokenCounterName
from weave_agent_signals.run_config import (
    DEFAULT_JUDGING_CONTEXT_POLICY,
    MODEL_CATALOG_SCHEMA_VERSION,
    RUBRIC_CATALOG_SCHEMA_VERSION,
    ModelCatalog,
    ModelDescriptor,
    RubricCatalog,
    RubricDescriptor,
)

_PROPOSAL_ROLE = "proposal_writer"
_JUDGE_ROLE = "judge"
_PROPOSAL_EVALUATOR_ROLE = "proposal_evaluator"


def _wandb_model(
    model_id: str,
    label: str,
    provider_model: str,
    family: str,
    max_input_tokens: int,
    token_counter: TokenCounterName = "utf8_bytes_div_3",
) -> ModelDescriptor:
    return ModelDescriptor(
        id=model_id,
        label=label,
        provider="wandb",
        provider_model=provider_model,
        family=family,
        supported_roles=(_PROPOSAL_ROLE, _JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
        max_input_tokens=max_input_tokens,
        token_counter=token_counter,
    )


_WANDB_MODELS = tuple(
    _wandb_model(*spec)
    for spec in (
        (
            "wandb:Mellum2-12B-A2.5B-Instruct",
            "Mellum2 12B A2.5B",
            "JetBrains/Mellum2-12B-A2.5B-Instruct",
            "jetbrains",
            131_000,
        ),
        ("wandb:MiniMax-M2.5", "MiniMax M2.5", "MiniMaxAI/MiniMax-M2.5", "minimax", 197_000),
        (
            "wandb:Qwen3-14B-Instruct",
            "Qwen3 14B Instruct",
            "OpenPipe/Qwen3-14B-Instruct",
            "qwen",
            32_800,
        ),
        (
            "wandb:Qwen3-30B-A3B-Instruct-2507",
            "Qwen3 30B A3B Instruct",
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "qwen",
            262_000,
        ),
        (
            "wandb:Qwen3-Coder-480B-A35B-Instruct",
            "Qwen3 Coder 480B A35B",
            "Qwen/Qwen3-Coder-480B-A35B-Instruct",
            "qwen",
            262_000,
        ),
        ("wandb:Qwen3.5-35B-A3B", "Qwen3.5 35B A3B", "Qwen/Qwen3.5-35B-A3B", "qwen", 262_000),
        ("wandb:Qwen3.6-27B", "Qwen3.6 27B", "Qwen/Qwen3.6-27B", "qwen", 262_000),
        ("wandb:Qwen3.6-35B-A3B", "Qwen3.6 35B A3B", "Qwen/Qwen3.6-35B-A3B", "qwen", 262_000),
        ("wandb:DeepSeek-V3.1", "DeepSeek V3.1", "deepseek-ai/DeepSeek-V3.1", "deepseek", 161_000),
        (
            "wandb:DeepSeek-V4-Flash",
            "DeepSeek V4 Flash",
            "deepseek-ai/DeepSeek-V4-Flash",
            "deepseek",
            1_049_000,
        ),
        (
            "wandb:DeepSeek-V4-Pro",
            "DeepSeek V4 Pro",
            "deepseek-ai/DeepSeek-V4-Pro",
            "deepseek",
            1_049_000,
        ),
        ("wandb:gemma-4-31B-it", "Gemma 4 31B", "google/gemma-4-31B-it", "google", 262_000),
        (
            "wandb:granite-4.1-8b",
            "Granite 4.1 8B",
            "ibm-granite/granite-4.1-8b",
            "ibm",
            131_072,
        ),
        (
            "wandb:Llama-3.1-70B",
            "Llama 3.1 70B",
            "meta-llama/Llama-3.1-70B-Instruct",
            "meta",
            128_000,
        ),
        (
            "wandb:Llama-3.1-8B",
            "Llama 3.1 8B",
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta",
            131_072,
        ),
        (
            "wandb:Llama-3.3-70B",
            "Llama 3.3 70B",
            "meta-llama/Llama-3.3-70B-Instruct",
            "meta",
            128_000,
        ),
        ("wandb:Kimi-K2.6", "Kimi K2.6", "moonshotai/Kimi-K2.6", "moonshot", 262_000),
        (
            "wandb:Kimi-K2.7-Code",
            "Kimi K2.7 Code",
            "moonshotai/Kimi-K2.7-Code",
            "moonshot",
            262_000,
        ),
        (
            "wandb:Nemotron-3-Super-120B",
            "Nemotron 3 Super 120B",
            "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
            "nvidia",
            262_000,
        ),
        (
            "wandb:Nemotron-3-Ultra-550B",
            "Nemotron 3 Ultra 550B",
            "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
            "nvidia",
            262_000,
        ),
        (
            "wandb:gpt-oss-20b",
            "GPT-OSS 20B",
            "openai/gpt-oss-20b",
            "openai",
            131_072,
            "o200k_harmony",
        ),
        (
            "wandb:gpt-oss-120b",
            "GPT-OSS 120B",
            "openai/gpt-oss-120b",
            "openai",
            131_072,
            "o200k_harmony",
        ),
        ("wandb:GLM-5.1", "GLM 5.1", "zai-org/GLM-5.1", "zai", 203_000),
        ("wandb:GLM-5.2", "GLM 5.2", "zai-org/GLM-5.2", "zai", 262_000),
    )
)


def _digest(value: object) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


_MODEL_DESCRIPTORS = (
    ModelDescriptor(
        id="claude:claude-sonnet-5",
        label="Claude Sonnet 5",
        provider="claude",
        provider_model="claude-sonnet-5",
        family="anthropic",
        supported_roles=(_PROPOSAL_ROLE, _JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
        max_input_tokens=200_000,
    ),
    ModelDescriptor(
        id="codex:gpt-5.6-sol",
        label="GPT-5.6 Sol",
        provider="codex",
        provider_model="gpt-5.6-sol",
        family="openai",
        supported_roles=(_PROPOSAL_ROLE, _JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
        max_input_tokens=272_000,
        token_counter="o200k_base",
    ),
    ModelDescriptor(
        id="codex:gpt-5.6-terra",
        label="GPT-5.6 Terra",
        provider="codex",
        provider_model="gpt-5.6-terra",
        family="openai",
        supported_roles=(_PROPOSAL_ROLE, _JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
        max_input_tokens=272_000,
        token_counter="o200k_base",
    ),
    ModelDescriptor(
        id="codex:gpt-5.6-luna",
        label="GPT-5.6 Luna",
        provider="codex",
        provider_model="gpt-5.6-luna",
        family="openai",
        supported_roles=(_PROPOSAL_ROLE, _JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
        max_input_tokens=272_000,
        token_counter="o200k_base",
    ),
    ModelDescriptor(
        id="codex:gpt-5.5",
        label="GPT-5.5",
        provider="codex",
        provider_model="gpt-5.5",
        family="openai",
        supported_roles=(_PROPOSAL_ROLE, _JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
        max_input_tokens=272_000,
        token_counter="o200k_base",
    ),
    ModelDescriptor(
        id="codex:gpt-5.4",
        label="GPT-5.4",
        provider="codex",
        provider_model="gpt-5.4",
        family="openai",
        supported_roles=(_PROPOSAL_ROLE, _JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
        max_input_tokens=272_000,
        token_counter="o200k_base",
    ),
    ModelDescriptor(
        id="codex:gpt-5.4-mini",
        label="GPT-5.4 Mini",
        provider="codex",
        provider_model="gpt-5.4-mini",
        family="openai",
        supported_roles=(_PROPOSAL_ROLE, _JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
        max_input_tokens=272_000,
        token_counter="o200k_base",
    ),
    ModelDescriptor(
        id="claude:claude-haiku-4-5",
        label="Claude Haiku 4.5",
        provider="claude",
        provider_model="claude-haiku-4-5",
        family="anthropic",
        supported_roles=(_PROPOSAL_ROLE, _JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
        max_input_tokens=200_000,
    ),
    ModelDescriptor(
        id="agy:gemini-3.5-flash-medium",
        label="Gemini 3.5 Flash (Medium)",
        provider="agy",
        provider_model="Gemini 3.5 Flash (Medium)",
        family="google",
        supported_roles=(_PROPOSAL_ROLE, _JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
        max_input_tokens=1_048_576,
    ),
    ModelDescriptor(
        id="agy:gemini-3.5-flash-high",
        label="Gemini 3.5 Flash (High)",
        provider="agy",
        provider_model="Gemini 3.5 Flash (High)",
        family="google",
        supported_roles=(_PROPOSAL_ROLE, _JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
        max_input_tokens=1_048_576,
    ),
    ModelDescriptor(
        id="agy:gemini-3.5-flash-low",
        label="Gemini 3.5 Flash (Low)",
        provider="agy",
        provider_model="Gemini 3.5 Flash (Low)",
        family="google",
        supported_roles=(_PROPOSAL_ROLE, _JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
        max_input_tokens=1_048_576,
    ),
    ModelDescriptor(
        id="agy:gemini-3.1-pro-low",
        label="Gemini 3.1 Pro (Low)",
        provider="agy",
        provider_model="Gemini 3.1 Pro (Low)",
        family="google",
        supported_roles=(_PROPOSAL_ROLE, _JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
        max_input_tokens=1_048_576,
    ),
    ModelDescriptor(
        id="agy:gemini-3.1-pro-high",
        label="Gemini 3.1 Pro (High)",
        provider="agy",
        provider_model="Gemini 3.1 Pro (High)",
        family="google",
        supported_roles=(_PROPOSAL_ROLE, _JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
        max_input_tokens=1_048_576,
    ),
    ModelDescriptor(
        id="agy:gpt-oss-120b-medium",
        label="GPT-OSS 120B (Medium)",
        provider="agy",
        provider_model="GPT-OSS 120B (Medium)",
        family="openai",
        supported_roles=(_PROPOSAL_ROLE, _JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
        max_input_tokens=131_072,
        token_counter="o200k_harmony",
    ),
    *_WANDB_MODELS,
    ModelDescriptor(
        id="openai:gpt-4o-mini",
        label="GPT-4o mini",
        provider="openai",
        provider_model="gpt-4o-mini",
        family="openai",
        supported_roles=(_JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
        max_input_tokens=128_000,
        token_counter="o200k_base",
    ),
    ModelDescriptor(
        id="openai:gpt-4o",
        label="GPT-4o",
        provider="openai",
        provider_model="gpt-4o",
        family="openai",
        supported_roles=(_JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
        max_input_tokens=128_000,
        token_counter="o200k_base",
    ),
)

_LOCAL_MODEL_EXECUTABLES = MappingProxyType(
    {
        descriptor.id: descriptor.provider
        for descriptor in _MODEL_DESCRIPTORS
        if descriptor.provider in {"claude", "codex", "agy"}
    }
)
_JUDGE_PREFERENCES = (
    "wandb:gpt-oss-120b",
    "wandb:Llama-3.1-8B",
    "wandb:granite-4.1-8b",
    "wandb:gpt-oss-20b",
    "claude:claude-sonnet-5",
    "codex:gpt-5.6-sol",
    "agy:gemini-3.1-pro-high",
    "claude:claude-haiku-4-5",
    "agy:gpt-oss-120b-medium",
    "openai:gpt-4o-mini",
    "openai:gpt-4o",
)
_EVALUATOR_PREFERENCES = (
    "wandb:gpt-oss-120b",
    "wandb:Llama-3.1-8B",
    "wandb:granite-4.1-8b",
    "wandb:gpt-oss-20b",
    "codex:gpt-5.6-sol",
    "agy:gemini-3.1-pro-high",
    "claude:claude-sonnet-5",
    "agy:gemini-3.5-flash-high",
    "agy:gpt-oss-120b-medium",
    "openai:gpt-4o",
    "openai:gpt-4o-mini",
)
_PROPOSAL_PREFERENCES = (
    "wandb:gpt-oss-120b",
    "wandb:Llama-3.1-8B",
    "wandb:granite-4.1-8b",
    "wandb:gpt-oss-20b",
    "codex:gpt-5.6-sol",
    "agy:gemini-3.1-pro-high",
    "claude:claude-sonnet-5",
    "agy:gemini-3.5-flash-high",
    "agy:gemini-3.5-flash-medium",
    "agy:gpt-oss-120b-medium",
    "codex:gpt-5.6-terra",
    "codex:gpt-5.6-luna",
    "codex:gpt-5.5",
    "codex:gpt-5.4",
    "codex:gpt-5.4-mini",
    "claude:claude-haiku-4-5",
)


def _descriptors_by_id() -> dict[str, ModelDescriptor]:
    descriptors: dict[str, ModelDescriptor] = {}
    for descriptor in _MODEL_DESCRIPTORS:
        if descriptor.id in descriptors:
            raise ValueError(f"duplicate model descriptor ID: {descriptor.id}")
        if not descriptor.id.startswith(f"{descriptor.provider}:"):
            raise ValueError(f"model descriptor ID is not provider-qualified: {descriptor.id}")
        inferred_family = model_family(descriptor.provider_model)
        if descriptor.family == "unknown" or (
            inferred_family != "unknown" and descriptor.family != inferred_family
        ):
            raise ValueError(f"model descriptor has an invalid family: {descriptor.id}")
        descriptors[descriptor.id] = descriptor
    return descriptors


def _ordered_descriptors(
    descriptors: Iterable[ModelDescriptor],
    preferred_ids: Sequence[str],
) -> tuple[ModelDescriptor, ...]:
    """Order known preferences first, then append every other descriptor by ID."""

    remaining = {descriptor.id: descriptor for descriptor in descriptors}
    ordered: list[ModelDescriptor] = []
    for model_id in preferred_ids:
        descriptor = remaining.pop(model_id, None)
        if descriptor is not None:
            ordered.append(descriptor)
    ordered.extend(sorted(remaining.values(), key=lambda descriptor: descriptor.id))
    return tuple(ordered)


def _recommended_judges(
    candidates: Sequence[str],
    descriptors: Mapping[str, ModelDescriptor],
) -> tuple[str, ...]:
    return tuple(
        model_id
        for model_id in candidates
        if (descriptor := descriptors.get(model_id)) is not None
        and _JUDGE_ROLE in descriptor.supported_roles
    )[:3]


def _validate_model_id_order(
    values: Sequence[str],
    *,
    available: Mapping[str, ModelDescriptor],
    role: str,
    label: str,
    require_all: bool = False,
) -> tuple[str, ...]:
    result = tuple(values)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must contain unique model IDs")
    unknown = [model_id for model_id in result if model_id not in available]
    if unknown:
        raise ValueError(f"{label} contains unavailable models: {', '.join(unknown)}")
    wrong_role = [
        model_id for model_id in result if role not in available[model_id].supported_roles
    ]
    if wrong_role:
        raise ValueError(f"{label} contains models without the {role} role")
    if require_all:
        capable = {
            model_id
            for model_id, descriptor in available.items()
            if role in descriptor.supported_roles
        }
        if set(result) != capable:
            raise ValueError(f"{label} must include every {role}-capable model")
    return result


def build_model_catalog(
    which: Callable[[str], str | None] = shutil.which,
    *,
    recommended_proposal_model: str | None = None,
    recommended_judges: Sequence[str] | None = None,
    recommended_challenge_judges: Sequence[str] | None = None,
    evaluator_preferences: Sequence[str] | None = None,
) -> ModelCatalog:
    """Build the unified provider-qualified model catalog.

    Descriptors expose role capabilities independent of local credentials.
    Local CLI models are included only when their provider executable is
    currently discoverable.
    """

    descriptors = _descriptors_by_id()
    executable_available = {
        executable: bool(which(executable)) for executable in set(_LOCAL_MODEL_EXECUTABLES.values())
    }
    available_descriptors = {
        model_id: descriptor
        for model_id, descriptor in descriptors.items()
        if model_id not in _LOCAL_MODEL_EXECUTABLES
        or executable_available[_LOCAL_MODEL_EXECUTABLES[model_id]]
    }

    ordered_models = _ordered_descriptors(
        available_descriptors.values(),
        _PROPOSAL_PREFERENCES,
    )
    available_by_id = {descriptor.id: descriptor for descriptor in ordered_models}
    proposal_ids = tuple(
        descriptor.id
        for descriptor in ordered_models
        if _PROPOSAL_ROLE in descriptor.supported_roles
    )
    proposal_recommendation = recommended_proposal_model or (
        proposal_ids[0] if proposal_ids else None
    )
    if proposal_recommendation is not None and proposal_recommendation not in available_by_id:
        raise ValueError("recommended proposal model is unavailable")
    if proposal_recommendation is not None and (
        _PROPOSAL_ROLE not in available_by_id[proposal_recommendation].supported_roles
    ):
        raise ValueError("recommended proposal model does not support proposal_writer")

    judge_ids = tuple(model_id for model_id in _JUDGE_PREFERENCES if model_id in available_by_id)
    judge_recommendations = _validate_model_id_order(
        recommended_judges or _recommended_judges(judge_ids, available_by_id),
        available=available_by_id,
        role=_JUDGE_ROLE,
        label="recommended judges",
    )
    if not 1 <= len(judge_recommendations) <= 3:
        raise ValueError("recommended judges must contain one through three models")

    challenge_recommendations = _validate_model_id_order(
        recommended_challenge_judges or judge_recommendations[:1],
        available=available_by_id,
        role=_JUDGE_ROLE,
        label="recommended challenge judges",
    )
    if not 1 <= len(challenge_recommendations) <= 3:
        raise ValueError("recommended challenge judges must contain one through three models")

    if evaluator_preferences is None:
        ordered_evaluators = _ordered_descriptors(
            (
                descriptor
                for descriptor in ordered_models
                if _PROPOSAL_EVALUATOR_ROLE in descriptor.supported_roles
            ),
            _EVALUATOR_PREFERENCES,
        )
        evaluator_values = tuple(descriptor.id for descriptor in ordered_evaluators)
    else:
        evaluator_values = tuple(evaluator_preferences)
    evaluator_order = _validate_model_id_order(
        evaluator_values,
        available=available_by_id,
        role=_PROPOSAL_EVALUATOR_ROLE,
        label="proposal evaluator preferences",
        require_all=True,
    )

    version_payload = {
        "schema_version": MODEL_CATALOG_SCHEMA_VERSION,
        "available_models": [model.model_dump(mode="json") for model in ordered_models],
        "judging_context": DEFAULT_JUDGING_CONTEXT_POLICY.model_dump(mode="json"),
        "recommended_proposal_model": proposal_recommendation,
        "recommended_judges": list(judge_recommendations),
        "recommended_challenge_judges": list(challenge_recommendations),
        "proposal_evaluator_preferences": list(evaluator_order),
    }
    return ModelCatalog(
        catalog_version=_digest(version_payload),
        available_models=ordered_models,
        judging_context=DEFAULT_JUDGING_CONTEXT_POLICY,
        recommended_proposal_model=proposal_recommendation,
        recommended_judges=judge_recommendations,
        recommended_challenge_judges=challenge_recommendations,
        proposal_evaluator_preferences=evaluator_order,
    )


def _rubric_content_digest(rubric: Rubric) -> str:
    return _digest(
        {
            "name": rubric.name,
            "scorer_name": rubric.scorer_name,
            "system_prompt": rubric.system_prompt,
            "criteria": dict(rubric.criteria),
            "tags_on_low": list(rubric.tags_on_low),
            "tags_on_high": list(rubric.tags_on_high),
            "threshold": rubric.threshold,
            "version": rubric.version,
            "evaluation_unit": rubric.evaluation_unit,
        }
    )


def build_rubric_catalog(
    rubrics: Mapping[str, Rubric] | Iterable[Rubric] | None = None,
) -> RubricCatalog:
    """Build descriptors from the rubric objects that remain the prompt source."""

    if rubrics is None:
        source = list(SESSION_RUBRICS.values())
    elif isinstance(rubrics, Mapping):
        source = list(rubrics.values())
    else:
        source = list(rubrics)

    descriptors: list[RubricDescriptor] = []
    seen: set[str] = set()
    for rubric in source:
        if rubric.scorer_name in seen:
            raise ValueError(f"duplicate rubric ID: {rubric.scorer_name}")
        if rubric.evaluation_unit != "session":
            raise ValueError(
                f"rubric {rubric.scorer_name} has invalid evaluation unit: {rubric.evaluation_unit}"
            )
        if not 0 <= rubric.threshold <= 1:
            raise ValueError(f"rubric {rubric.scorer_name} has an invalid threshold")
        if not rubric.version:
            raise ValueError(f"rubric {rubric.scorer_name} has no version")
        seen.add(rubric.scorer_name)
        descriptors.append(
            RubricDescriptor(
                id=rubric.scorer_name,
                label=rubric.name,
                evaluation_unit=rubric.evaluation_unit,
                version=rubric.version,
                content_digest=_rubric_content_digest(rubric),
                pass_threshold=rubric.threshold,
            )
        )

    rubric_tuple = tuple(descriptors)
    version_payload = {
        "schema_version": RUBRIC_CATALOG_SCHEMA_VERSION,
        "rubrics": [rubric.model_dump(mode="json") for rubric in rubric_tuple],
    }
    return RubricCatalog(
        catalog_version=_digest(version_payload),
        rubrics=rubric_tuple,
    )

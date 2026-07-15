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
from weave_agent_signals.judges.rubrics import RUBRICS, SESSION_RUBRICS, Rubric
from weave_agent_signals.run_config import (
    MODEL_CATALOG_SCHEMA_VERSION,
    RUBRIC_CATALOG_SCHEMA_VERSION,
    JudgeBackendCatalog,
    ModelCatalog,
    ModelDescriptor,
    ProposalCatalog,
    RubricCatalog,
    RubricDescriptor,
)

REVIEW_DEPTHS = ("primary", "selective", "full_panel")
DEFAULT_SECOND_OPINION_MARGIN = 0.10

_PROPOSAL_ROLE = "proposal_writer"
_JUDGE_ROLE = "judge"
_PROPOSAL_EVALUATOR_ROLE = "proposal_evaluator"


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
        id="claude-sonnet-5",
        label="Claude Sonnet 5",
        family="anthropic",
        backend="cli",
        supported_roles=(_PROPOSAL_ROLE, _JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
    ),
    ModelDescriptor(
        id="gpt-5.6-sol",
        label="GPT-5.6 Sol",
        family="openai",
        backend="cli",
        supported_roles=(_PROPOSAL_ROLE, _JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
    ),
    ModelDescriptor(
        id="claude-haiku-4-5",
        label="Claude Haiku 4.5",
        family="anthropic",
        backend="cli",
        supported_roles=(_PROPOSAL_ROLE, _JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
    ),
    ModelDescriptor(
        id="gpt-oss-20b",
        label="GPT-OSS 20B",
        family="openai",
        backend="wandb",
        supported_roles=(_JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
    ),
    ModelDescriptor(
        id="Llama-3.1-8B",
        label="Llama 3.1 8B",
        family="meta",
        backend="wandb",
        supported_roles=(_JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
    ),
    ModelDescriptor(
        id="granite-4.1-8b",
        label="Granite 4.1 8B",
        family="ibm",
        backend="wandb",
        supported_roles=(_JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
    ),
    ModelDescriptor(
        id="gpt-oss-120b",
        label="GPT-OSS 120B",
        family="openai",
        backend="wandb",
        supported_roles=(_JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
    ),
    ModelDescriptor(
        id="gpt-4o-mini",
        label="GPT-4o mini",
        family="openai",
        backend="openai",
        supported_roles=(_JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
    ),
    ModelDescriptor(
        id="gpt-4o",
        label="GPT-4o",
        family="openai",
        backend="openai",
        supported_roles=(_JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE),
    ),
)

_LOCAL_MODEL_EXECUTABLES = MappingProxyType(
    {
        "claude-sonnet-5": "claude",
        "claude-haiku-4-5": "claude",
        "gpt-5.6-sol": "codex",
    }
)
_JUDGE_PREFERENCES = MappingProxyType(
    {
        "cli": ("claude-sonnet-5", "gpt-5.6-sol", "claude-haiku-4-5"),
        "wandb": ("gpt-oss-20b", "Llama-3.1-8B", "granite-4.1-8b", "gpt-oss-120b"),
        "openai": ("gpt-4o-mini", "gpt-4o"),
    }
)
_EVALUATOR_PREFERENCES = MappingProxyType(
    {
        "cli": ("gpt-5.6-sol", "claude-sonnet-5", "claude-haiku-4-5"),
        "wandb": ("gpt-oss-120b", "gpt-oss-20b", "Llama-3.1-8B", "granite-4.1-8b"),
        "openai": ("gpt-4o", "gpt-4o-mini"),
    }
)
_PROPOSAL_PREFERENCES = ("gpt-5.6-sol", "claude-sonnet-5", "claude-haiku-4-5")
_BACKEND_PREFERENCES = ("cli", "wandb", "openai")


def _descriptors_by_id() -> dict[str, ModelDescriptor]:
    descriptors: dict[str, ModelDescriptor] = {}
    for descriptor in _MODEL_DESCRIPTORS:
        if descriptor.id in descriptors:
            raise ValueError(f"duplicate model descriptor ID: {descriptor.id}")
        if descriptor.family == "unknown" or descriptor.family != model_family(descriptor.id):
            raise ValueError(f"model descriptor has an invalid family: {descriptor.id}")
        descriptors[descriptor.id] = descriptor
    return descriptors


def _supported_review_depths(model_count: int) -> tuple[str, ...]:
    depths: list[str] = []
    if model_count >= 1:
        depths.append("primary")
    if model_count >= 2:
        depths.append("selective")
    if model_count >= 3:
        depths.append("full_panel")
    return tuple(depths)


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


def _ordered_backends(backends: Iterable[str]) -> tuple[str, ...]:
    remaining = set(backends)
    ordered: list[str] = []
    for backend in _BACKEND_PREFERENCES:
        if backend in remaining:
            remaining.remove(backend)
            ordered.append(backend)
    ordered.extend(sorted(remaining))
    return tuple(ordered)


def _recommended_judges(
    candidates: Sequence[str],
    descriptors: Mapping[str, ModelDescriptor],
) -> tuple[str, ...]:
    selected: list[str] = []
    selected_families: set[str] = set()
    for model_id in candidates:
        descriptor = descriptors.get(model_id)
        if descriptor is None or _JUDGE_ROLE not in descriptor.supported_roles:
            continue
        if descriptor.family in selected_families:
            continue
        selected.append(model_id)
        selected_families.add(descriptor.family)
        if len(selected) == 3:
            return tuple(selected)

    for model_id in candidates:
        if model_id in selected:
            continue
        descriptor = descriptors.get(model_id)
        if descriptor is None or _JUDGE_ROLE not in descriptor.supported_roles:
            continue
        selected.append(model_id)
        if len(selected) == 3:
            break
    return tuple(selected)


def _recommendation_depth(recommended_judges: Sequence[str]) -> str | None:
    if len(recommended_judges) == 1:
        return "primary"
    if len(recommended_judges) in {2, 3}:
        return "selective"
    if not recommended_judges:
        return None
    raise ValueError("recommended judge lists may contain at most three models")


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
    recommended_judges: Mapping[str, Sequence[str]] | None = None,
    evaluator_preferences: Mapping[str, Sequence[str]] | None = None,
) -> ModelCatalog:
    """Build the current role-oriented model catalog.

    Provider backends describe their selectable capabilities independent of
    local credentials. Local CLI models are included only when the executable
    that owns them is currently discoverable.
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

    proposal_models = _ordered_descriptors(
        (
            descriptor
            for descriptor in available_descriptors.values()
            if _PROPOSAL_ROLE in descriptor.supported_roles
        ),
        _PROPOSAL_PREFERENCES,
    )
    proposal = ProposalCatalog(
        available_models=proposal_models,
        recommended_model=proposal_models[0].id if proposal_models else None,
    )

    recommendation_overrides = dict(recommended_judges or {})
    evaluator_overrides = dict(evaluator_preferences or {})
    backend_names = _ordered_backends(
        descriptor.backend
        for descriptor in available_descriptors.values()
        if {_JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE} & set(descriptor.supported_roles)
    )
    unknown_overrides = (set(recommendation_overrides) | set(evaluator_overrides)) - set(
        backend_names
    )
    if unknown_overrides:
        raise ValueError("unknown backend overrides: " + ", ".join(sorted(unknown_overrides)))

    backend_catalogs: dict[str, JudgeBackendCatalog] = {}
    for backend_name in backend_names:
        backend_descriptors = _ordered_descriptors(
            (
                descriptor
                for descriptor in available_descriptors.values()
                if descriptor.backend == backend_name
                and {_JUDGE_ROLE, _PROPOSAL_EVALUATOR_ROLE} & set(descriptor.supported_roles)
            ),
            _JUDGE_PREFERENCES.get(backend_name, ()),
        )
        available_by_id = {descriptor.id: descriptor for descriptor in backend_descriptors}
        judge_ids = tuple(
            descriptor.id
            for descriptor in backend_descriptors
            if _JUDGE_ROLE in descriptor.supported_roles
        )

        if backend_name in recommendation_overrides:
            backend_recommendations = _validate_model_id_order(
                recommendation_overrides[backend_name],
                available=available_by_id,
                role=_JUDGE_ROLE,
                label=f"{backend_name} recommended judges",
            )
        else:
            backend_recommendations = _recommended_judges(
                judge_ids,
                available_by_id,
            )

        if backend_name in evaluator_overrides:
            backend_evaluator_preferences = _validate_model_id_order(
                evaluator_overrides[backend_name],
                available=available_by_id,
                role=_PROPOSAL_EVALUATOR_ROLE,
                label=f"{backend_name} evaluator preferences",
                require_all=True,
            )
        else:
            ordered_evaluators = _ordered_descriptors(
                (
                    descriptor
                    for descriptor in backend_descriptors
                    if _PROPOSAL_EVALUATOR_ROLE in descriptor.supported_roles
                ),
                _EVALUATOR_PREFERENCES.get(backend_name, ()),
            )
            backend_evaluator_preferences = _validate_model_id_order(
                [descriptor.id for descriptor in ordered_evaluators],
                available=available_by_id,
                role=_PROPOSAL_EVALUATOR_ROLE,
                label=f"{backend_name} evaluator preferences",
                require_all=True,
            )

        recommended_depth = _recommendation_depth(backend_recommendations)
        supported_depths = _supported_review_depths(len(judge_ids))
        if recommended_depth is not None and recommended_depth not in supported_depths:
            raise ValueError(
                f"{backend_name} recommendations do not satisfy a supported review depth"
            )
        backend_catalogs[backend_name] = JudgeBackendCatalog(
            available_models=backend_descriptors,
            recommended_judges=backend_recommendations,
            proposal_evaluator_preferences=backend_evaluator_preferences,
            recommended_review_depth=recommended_depth,
            supported_review_depths=supported_depths,
        )

    recommended_backend = next(
        (backend for backend in backend_names if backend_catalogs[backend].recommended_judges),
        None,
    )
    if recommended_backend is None:
        raise ValueError("model catalog has no judge-capable backend")
    review_defaults = MappingProxyType({"second_opinion_margin": DEFAULT_SECOND_OPINION_MARGIN})
    version_payload = {
        "schema_version": MODEL_CATALOG_SCHEMA_VERSION,
        "proposal": proposal.model_dump(mode="json"),
        "review_defaults": dict(review_defaults),
        "recommended_judge_backend": recommended_backend,
        "judge_backends": {
            name: backend.model_dump(mode="json") for name, backend in backend_catalogs.items()
        },
    }
    return ModelCatalog(
        catalog_version=_digest(version_payload),
        proposal=proposal,
        review_defaults=review_defaults,
        recommended_judge_backend=recommended_backend,
        judge_backends=MappingProxyType(backend_catalogs),
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
        source = [*RUBRICS.values(), *SESSION_RUBRICS.values()]
    elif isinstance(rubrics, Mapping):
        source = list(rubrics.values())
    else:
        source = list(rubrics)

    descriptors: list[RubricDescriptor] = []
    seen: set[str] = set()
    for rubric in source:
        if rubric.scorer_name in seen:
            raise ValueError(f"duplicate rubric ID: {rubric.scorer_name}")
        if rubric.evaluation_unit not in {"episode", "session"}:
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

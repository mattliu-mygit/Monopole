"""Canonical requested and effective evaluation-run configuration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    field_validator,
    model_validator,
)

from weave_agent_signals.judges.tokens import TokenCounterName

PIPELINE_VERSION = "8"
MODEL_CATALOG_SCHEMA_VERSION = "6"
RUBRIC_CATALOG_SCHEMA_VERSION = "1"
EFFECTIVE_RUN_CONFIG_SCHEMA_VERSION = "5"
MAX_CANDIDATE_BUDGET = 10

ModelRole = Literal["proposal_writer", "judge", "proposal_evaluator"]
EvaluationUnit = Literal["session"]


def _require_nonblank(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must be nonblank")
    return value


def _require_nonblank_unique(values: Sequence[str], field_name: str) -> None:
    if any(not value.strip() for value in values):
        raise ValueError(f"{field_name} must contain nonblank strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must be unique")


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class JudgingContextPolicy(StrictFrozenModel):
    contract_version: Literal["3"] = "3"
    large_model_threshold_tokens: Literal[200_000] = 200_000
    large_model_reserve_tokens: Annotated[int, Field(strict=True, ge=1)] = 18_000
    small_model_reserve_tokens: Annotated[int, Field(strict=True, ge=1)] = 18_000
    large_model_raw_target_tokens: Annotated[int, Field(strict=True, ge=1)] = 128_000
    small_model_raw_target_tokens: Annotated[int, Field(strict=True, ge=1)] = 50_000
    prompt_reserve_tokens: Annotated[int, Field(strict=True, ge=1)] = 6_000
    output_reserve_tokens: Annotated[int, Field(strict=True, ge=1)] = 4_000
    large_model_output_reserve_tokens: Annotated[int, Field(strict=True, ge=1)] = 10_000
    safety_reserve_tokens: Annotated[int, Field(strict=True, ge=1)] = 8_000
    digest_max_tokens: Annotated[int, Field(strict=True, ge=1)] = 1_000
    finding_max_tokens: Annotated[int, Field(strict=True, ge=1)] = 10_000
    overlap_turns: Literal[1] = 1
    max_chunks: Annotated[int, Field(strict=True, ge=1)] = 40

    def capacity_reserve(self, model_limit: int) -> int:
        """Return the configured reserve tier for a validated model capacity."""

        if isinstance(model_limit, bool) or not isinstance(model_limit, int) or model_limit <= 0:
            raise ValueError("model_limit must be a positive integer")
        if model_limit > self.large_model_threshold_tokens:
            return self.large_model_reserve_tokens
        return self.small_model_reserve_tokens

    def raw_window_target(self, model_limit: int) -> int:
        """Return the soft raw-window target for a validated model capacity."""

        self.capacity_reserve(model_limit)
        if model_limit > self.large_model_threshold_tokens:
            return self.large_model_raw_target_tokens
        return self.small_model_raw_target_tokens

    def generation_budget(self, model_limit: int) -> int:
        """Return the structured-generation allowance for a model capacity."""

        self.capacity_reserve(model_limit)
        if model_limit > self.large_model_threshold_tokens:
            return self.large_model_output_reserve_tokens
        return self.output_reserve_tokens


DEFAULT_JUDGING_CONTEXT_POLICY = JudgingContextPolicy()


class ModelDescriptor(StrictFrozenModel):
    id: StrictStr
    label: StrictStr
    provider: StrictStr
    provider_model: StrictStr
    family: StrictStr
    supported_roles: tuple[ModelRole, ...]
    max_input_tokens: Annotated[int, Field(strict=True, ge=1)] = 128_000
    token_counter: TokenCounterName = "utf8_bytes_div_3"

    @model_validator(mode="after")
    def validate_descriptor(self) -> ModelDescriptor:
        for field_name in ("id", "label", "provider", "provider_model", "family"):
            _require_nonblank(getattr(self, field_name), field_name)
        if not self.supported_roles:
            raise ValueError("supported_roles must not be empty")
        if len(self.supported_roles) != len(set(self.supported_roles)):
            raise ValueError("supported_roles must be unique")
        return self


class RubricDescriptor(StrictFrozenModel):
    id: StrictStr
    label: StrictStr
    evaluation_unit: EvaluationUnit
    version: StrictStr
    content_digest: StrictStr
    pass_threshold: Annotated[float, Field(ge=0, le=1)]

    @field_validator("pass_threshold", mode="before")
    @classmethod
    def validate_numeric_threshold(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("pass_threshold must be numeric")
        return float(value)

    @model_validator(mode="after")
    def validate_descriptor(self) -> RubricDescriptor:
        for field_name in ("id", "label", "version", "content_digest"):
            _require_nonblank(getattr(self, field_name), field_name)
        return self


class SelectionWarning(StrictFrozenModel):
    code: StrictStr
    message: StrictStr
    affected_roles: tuple[StrictStr, ...]
    selected_model_ids: tuple[StrictStr, ...]
    compared_families: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def validate_warning(self) -> SelectionWarning:
        _require_nonblank(self.code, "code")
        _require_nonblank(self.message, "message")
        _require_nonblank_unique(self.affected_roles, "affected_roles")
        if any(not value.strip() for value in self.selected_model_ids):
            raise ValueError("selected_model_ids must contain nonblank strings")
        if any(not value.strip() for value in self.compared_families):
            raise ValueError("compared_families must contain nonblank strings")
        return self


class ModelCatalog(StrictFrozenModel):
    catalog_version: StrictStr
    available_models: tuple[ModelDescriptor, ...]
    judging_context: JudgingContextPolicy
    recommended_proposal_model: StrictStr | None
    recommended_judges: tuple[StrictStr, ...]
    recommended_challenge_judges: tuple[StrictStr, ...]
    proposal_evaluator_preferences: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def validate_catalog(self) -> ModelCatalog:
        _require_nonblank(self.catalog_version, "catalog_version")
        ids = tuple(model.id for model in self.available_models)
        _require_nonblank_unique(ids, "model IDs")
        _require_nonblank_unique(self.recommended_judges, "recommended_judges")
        _require_nonblank_unique(
            self.recommended_challenge_judges,
            "recommended_challenge_judges",
        )
        _require_nonblank_unique(
            self.proposal_evaluator_preferences,
            "proposal_evaluator_preferences",
        )
        if self.recommended_proposal_model is not None:
            _require_nonblank(self.recommended_proposal_model, "recommended_proposal_model")
            if self.recommended_proposal_model not in ids:
                raise ValueError("recommended proposal model must be available")
            if "proposal_writer" not in self.model(self.recommended_proposal_model).supported_roles:
                raise ValueError("recommended proposal model must support proposal_writer")
        if not 1 <= len(self.recommended_judges) <= 3:
            raise ValueError("recommended judges must contain one through three models")
        if not set(self.recommended_judges) <= set(ids):
            raise ValueError("recommended judges must be available")
        if not 1 <= len(self.recommended_challenge_judges) <= 3:
            raise ValueError("recommended challenge judges must contain one through three models")
        if not set(self.recommended_challenge_judges) <= set(ids):
            raise ValueError("recommended challenge judges must be available")
        if not set(self.proposal_evaluator_preferences) <= set(ids):
            raise ValueError("proposal evaluator preferences must be available")
        if any(
            "judge" not in self.model(model_id).supported_roles
            for model_id in self.recommended_judges
        ):
            raise ValueError("recommended judges must support judging")
        if any(
            "judge" not in self.model(model_id).supported_roles
            for model_id in self.recommended_challenge_judges
        ):
            raise ValueError("recommended challenge judges must support judging")
        if any(
            "proposal_evaluator" not in self.model(model_id).supported_roles
            for model_id in self.proposal_evaluator_preferences
        ):
            raise ValueError("proposal evaluator preferences must support proposal evaluation")
        return self

    def model(self, model_id: str) -> ModelDescriptor:
        for descriptor in self.available_models:
            if descriptor.id == model_id:
                return descriptor
        raise KeyError(model_id)


class RubricCatalog(StrictFrozenModel):
    catalog_version: StrictStr
    rubrics: tuple[RubricDescriptor, ...]

    @model_validator(mode="after")
    def validate_catalog(self) -> RubricCatalog:
        _require_nonblank(self.catalog_version, "catalog_version")
        _require_nonblank_unique(tuple(rubric.id for rubric in self.rubrics), "rubric IDs")
        return self

    def rubric(self, rubric_id: str) -> RubricDescriptor:
        for descriptor in self.rubrics:
            if descriptor.id == rubric_id:
                return descriptor
        raise KeyError(rubric_id)


class RunConfig(StrictFrozenModel):
    model_catalog_version: StrictStr
    rubric_catalog_version: StrictStr
    judge_models: tuple[StrictStr, ...]
    challenge_judge_models: tuple[StrictStr, ...]
    proposal_model: StrictStr
    proposal_evaluator_model: StrictStr
    rubrics: tuple[StrictStr, ...]
    candidate_budget: Annotated[
        int,
        Field(strict=True, ge=1, le=MAX_CANDIDATE_BUDGET),
    ]
    force: StrictBool

    @model_validator(mode="after")
    def validate_request(self) -> RunConfig:
        for field_name in (
            "model_catalog_version",
            "rubric_catalog_version",
            "proposal_model",
            "proposal_evaluator_model",
        ):
            _require_nonblank(getattr(self, field_name), field_name)
        _require_nonblank_unique(self.judge_models, "judge_models")
        _require_nonblank_unique(self.challenge_judge_models, "challenge_judge_models")
        _require_nonblank_unique(self.rubrics, "rubrics")

        if not 1 <= len(self.judge_models) <= 3:
            raise ValueError("judge_models must contain one through three models")
        if not 1 <= len(self.challenge_judge_models) <= 3:
            raise ValueError("challenge_judge_models must contain one through three models")
        return self


class EvaluatedModelIdentity(StrictFrozenModel):
    id: StrictStr | None
    family: StrictStr

    @model_validator(mode="after")
    def validate_identity(self) -> EvaluatedModelIdentity:
        if self.id is not None:
            _require_nonblank(self.id, "id")
        _require_nonblank(self.family, "family")
        return self


class PositionedJudge(ModelDescriptor):
    role: Literal["judge"] = "judge"
    position: Annotated[int, Field(strict=True, ge=1)]

    @property
    def model(self) -> ModelDescriptor:
        return ModelDescriptor.model_validate(self.model_dump(exclude={"role", "position"}))


class EffectiveModelSelection(StrictFrozenModel):
    proposal_writer: ModelDescriptor
    judges: tuple[PositionedJudge, ...]
    challenge_judges: tuple[PositionedJudge, ...]
    proposal_evaluator: ModelDescriptor

    @model_validator(mode="after")
    def validate_selection(self) -> EffectiveModelSelection:
        if "proposal_writer" not in self.proposal_writer.supported_roles:
            raise ValueError("proposal writer does not support proposal_writer")
        if "proposal_evaluator" not in self.proposal_evaluator.supported_roles:
            raise ValueError("proposal evaluator does not support proposal_evaluator")
        if any("judge" not in judge.supported_roles for judge in self.judges):
            raise ValueError("every positioned judge must support judging")
        if any("judge" not in judge.supported_roles for judge in self.challenge_judges):
            raise ValueError("every challenge judge must support judging")
        for label, judges in (
            ("judge", self.judges),
            ("challenge judge", self.challenge_judges),
        ):
            if tuple(judge.position for judge in judges) != tuple(range(1, len(judges) + 1)):
                raise ValueError(f"{label} positions must be contiguous and 1-based")
            if len({judge.id for judge in judges}) != len(judges):
                raise ValueError(f"positioned {label}s must be unique")
            if not 1 <= len(judges) <= 3:
                raise ValueError(f"{label}s must contain one through three models")
        return self


class EffectiveRunConfig(StrictFrozenModel):
    schema_version: Literal["5"] = "5"
    pipeline_version: StrictStr
    model_catalog_version: StrictStr
    rubric_catalog_version: StrictStr
    models: EffectiveModelSelection
    rubrics: tuple[RubricDescriptor, ...]
    selection_warnings: tuple[SelectionWarning, ...]
    judging_context: JudgingContextPolicy
    candidate_budget: Annotated[
        int,
        Field(strict=True, ge=1, le=MAX_CANDIDATE_BUDGET),
    ]
    force: StrictBool

    @model_validator(mode="after")
    def validate_snapshot(self) -> EffectiveRunConfig:
        _require_nonblank(self.pipeline_version, "pipeline_version")
        _require_nonblank(self.model_catalog_version, "model_catalog_version")
        _require_nonblank(self.rubric_catalog_version, "rubric_catalog_version")
        if not self.rubrics:
            raise ValueError("effective configuration must contain at least one rubric")
        return self


def _panel_selection_warnings(
    judges: Sequence[ModelDescriptor],
    evaluated_models: Sequence[EvaluatedModelIdentity],
    *,
    code_prefix: str,
    role_prefix: str,
    panel_label: str,
) -> tuple[SelectionWarning, ...]:
    warnings: list[SelectionWarning] = []
    judge_families = tuple(judge.family for judge in judges)
    repeated_families = tuple(
        sorted(
            family
            for family in set(judge_families)
            if family != "unknown" and judge_families.count(family) > 1
        )
    )
    if repeated_families:
        warnings.append(
            SelectionWarning(
                code=f"{code_prefix}judge_family_overlap",
                message=f"Multiple selected {panel_label} judges share a model family.",
                affected_roles=tuple(
                    f"{role_prefix}_{position}"
                    for position, judge in enumerate(judges, start=1)
                    if judge.family in repeated_families
                ),
                selected_model_ids=tuple(
                    judge.id for judge in judges if judge.family in repeated_families
                ),
                compared_families=repeated_families,
            )
        )

    known_judge_families = {family for family in judge_families if family != "unknown"}
    if len(judges) > 1 and len(known_judge_families) < len(judges):
        warnings.append(
            SelectionWarning(
                code=f"low_{code_prefix}judge_family_diversity",
                message=f"The selected {panel_label} judge panel has fewer families than models.",
                affected_roles=tuple(
                    f"{role_prefix}_{position}" for position in range(1, len(judges) + 1)
                ),
                selected_model_ids=tuple(judge.id for judge in judges),
                compared_families=tuple(sorted(known_judge_families)),
            )
        )

    evaluated_families = {
        evaluated.family for evaluated in evaluated_models if evaluated.family != "unknown"
    }
    overlaps = tuple(sorted(known_judge_families & evaluated_families))
    if overlaps:
        warnings.append(
            SelectionWarning(
                code=f"{code_prefix}judge_evaluated_family_overlap",
                message=f"A selected {panel_label} judge shares a family with an evaluated model.",
                affected_roles=tuple(
                    [
                        f"{role_prefix}_{position}"
                        for position, judge in enumerate(judges, start=1)
                        if judge.family in overlaps
                    ]
                    + ["evaluated_agent"]
                ),
                selected_model_ids=tuple(judge.id for judge in judges if judge.family in overlaps)
                + tuple(
                    evaluated.id
                    for evaluated in evaluated_models
                    if evaluated.family in overlaps and evaluated.id is not None
                ),
                compared_families=overlaps,
            )
        )
    return tuple(warnings)


def _selection_warnings(
    judges: Sequence[ModelDescriptor],
    challenge_judges: Sequence[ModelDescriptor],
    proposal_writer: ModelDescriptor,
    proposal_evaluator: ModelDescriptor,
    evaluated_models: Sequence[EvaluatedModelIdentity],
) -> tuple[SelectionWarning, ...]:
    warnings = list(
        _panel_selection_warnings(
            judges,
            evaluated_models,
            code_prefix="",
            role_prefix="judge",
            panel_label="session",
        )
    )
    warnings.extend(
        _panel_selection_warnings(
            challenge_judges,
            evaluated_models,
            code_prefix="challenge_",
            role_prefix="challenge_judge",
            panel_label="B/C verification",
        )
    )

    if proposal_writer.family != "unknown" and proposal_writer.family == proposal_evaluator.family:
        warnings.append(
            SelectionWarning(
                code="proposal_evaluator_writer_family_overlap",
                message="The proposal writer and evaluator share a model family.",
                affected_roles=("proposal_writer", "proposal_evaluator"),
                selected_model_ids=(proposal_writer.id, proposal_evaluator.id),
                compared_families=(proposal_writer.family,),
            )
        )
    return tuple(warnings)


def resolve_run_config(
    requested: RunConfig,
    *,
    model_catalog: ModelCatalog,
    rubric_catalog: RubricCatalog,
    evaluated_models: Sequence[EvaluatedModelIdentity] = (),
    pipeline_version: str = PIPELINE_VERSION,
) -> EffectiveRunConfig:
    """Resolve an explicit request to immutable descriptors without substitution."""

    if requested.model_catalog_version != model_catalog.catalog_version:
        raise ValueError("stale model catalog version")
    if requested.rubric_catalog_version != rubric_catalog.catalog_version:
        raise ValueError("stale rubric catalog version")

    judges: list[ModelDescriptor] = []
    for model_id in requested.judge_models:
        try:
            descriptor = model_catalog.model(model_id)
        except KeyError as exc:
            raise ValueError(f"judge model {model_id} is unknown or unavailable") from exc
        if "judge" not in descriptor.supported_roles:
            raise ValueError(f"model {model_id} does not support judging")
        judges.append(descriptor)

    challenge_judges: list[ModelDescriptor] = []
    for model_id in requested.challenge_judge_models:
        try:
            descriptor = model_catalog.model(model_id)
        except KeyError as exc:
            raise ValueError(f"challenge judge model {model_id} is unknown or unavailable") from exc
        if "judge" not in descriptor.supported_roles:
            raise ValueError(f"model {model_id} does not support challenge judging")
        challenge_judges.append(descriptor)

    try:
        proposal_writer = model_catalog.model(requested.proposal_model)
    except KeyError as exc:
        raise ValueError(
            f"unknown or unavailable proposal model: {requested.proposal_model}"
        ) from exc
    if "proposal_writer" not in proposal_writer.supported_roles:
        raise ValueError(f"unknown or unavailable proposal model: {requested.proposal_model}")

    try:
        proposal_evaluator = model_catalog.model(requested.proposal_evaluator_model)
    except KeyError as exc:
        raise ValueError(
            f"proposal evaluator {requested.proposal_evaluator_model} is unknown or unavailable"
        ) from exc
    if "proposal_evaluator" not in proposal_evaluator.supported_roles:
        raise ValueError(
            f"model {requested.proposal_evaluator_model} does not support proposal evaluation"
        )

    if requested.rubrics:
        selected_rubrics: list[RubricDescriptor] = []
        for rubric_id in requested.rubrics:
            try:
                selected_rubrics.append(rubric_catalog.rubric(rubric_id))
            except KeyError as exc:
                raise ValueError(f"unknown rubric: {rubric_id}") from exc
    else:
        selected_rubrics = list(rubric_catalog.rubrics)

    warnings = _selection_warnings(
        judges,
        challenge_judges,
        proposal_writer,
        proposal_evaluator,
        evaluated_models,
    )
    return EffectiveRunConfig(
        pipeline_version=pipeline_version,
        model_catalog_version=model_catalog.catalog_version,
        rubric_catalog_version=rubric_catalog.catalog_version,
        models=EffectiveModelSelection(
            proposal_writer=proposal_writer,
            judges=tuple(
                PositionedJudge.model_validate(
                    {
                        **descriptor.model_dump(),
                        "position": position,
                    }
                )
                for position, descriptor in enumerate(judges, start=1)
            ),
            challenge_judges=tuple(
                PositionedJudge.model_validate(
                    {
                        **descriptor.model_dump(),
                        "position": position,
                    }
                )
                for position, descriptor in enumerate(challenge_judges, start=1)
            ),
            proposal_evaluator=proposal_evaluator,
        ),
        rubrics=tuple(selected_rubrics),
        selection_warnings=warnings,
        judging_context=DEFAULT_JUDGING_CONTEXT_POLICY,
        candidate_budget=requested.candidate_budget,
        force=requested.force,
    )

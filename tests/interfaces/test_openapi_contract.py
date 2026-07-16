from __future__ import annotations

from pathlib import Path

from weave_agent_signals.api import create_app


def test_every_api_operation_declares_a_success_response_schema() -> None:
    schema = create_app().openapi()

    for path, operations in schema["paths"].items():
        if not path.startswith("/api/"):
            continue
        for method, operation in operations.items():
            if method not in {"get", "post", "put", "delete", "patch"}:
                continue
            success = operation["responses"]["200"]["content"]["application/json"]
            assert "$ref" in success["schema"], f"{method.upper()} {path} has no response model"


def test_openapi_contains_partial_promotion_outcomes() -> None:
    schemas = create_app().openapi()["components"]["schemas"]

    review_status = schemas["ReflectionReviewResponse"]["properties"]["status"]
    outcome = schemas["PromotionTargetOutcomeResponse"]["properties"]
    assert "partial" in review_status["enum"]
    assert outcome["action"]["enum"] == ["create", "update"]
    assert outcome["status"]["enum"] == ["applied", "not_applied"]


def _referenced_schema(property_schema: dict) -> str | None:
    choices = property_schema.get("anyOf", [property_schema])
    return next((choice.get("$ref") for choice in choices if "$ref" in choice), None)


def test_run_transport_fields_used_by_ui_have_named_schemas() -> None:
    schemas = create_app().openapi()["components"]["schemas"]
    run = schemas["RunResponse"]["properties"]

    expected = {
        "scoring_progress": "ScoringProgressResponse",
        "scoring_result": "ScoringResultResponse",
        "judging_plan": "JudgingPlanResponse",
        "judging_progress": "JudgingProgressResponse",
        "judging_result": "JudgingProgressResponse",
        "reflecting_progress": "ReflectingProgressResponse",
        "reflecting_result": "ReflectionResultResponse",
        "reflection_review": "ReflectionReviewResponse",
    }
    for field, model in expected.items():
        assert _referenced_schema(run[field]) == f"#/components/schemas/{model}"

    judge_backends = schemas["ModelCatalogResponse"]["properties"]["judge_backends"]
    assert judge_backends["additionalProperties"]["$ref"] == (
        "#/components/schemas/JudgeBackendCatalog"
    )


def test_frontend_transport_does_not_cast_generated_run_or_session_responses() -> None:
    source = (Path(__file__).parents[2] / "frontend" / "src" / "api.ts").read_text(encoding="utf-8")

    assert "as unknown as Promise<Run>" not in source
    assert "as unknown as Promise<SessionDetail>" not in source

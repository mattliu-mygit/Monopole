from weave_agent_signals.catalogs import build_rubric_catalog
from weave_agent_signals.judges.rubrics import SESSION_RUBRICS
from weave_agent_signals.judges.runner import REVIEW_POLICY_VERSION


def test_all_six_model_rubrics_are_authoritative_session_v4_descriptors() -> None:
    assert list(SESSION_RUBRICS) == [
        "judge.verification",
        "judge.error_recovery",
        "judge.tool_choice",
        "judge.state_consistency",
        "judge.session_outcome",
        "judge.session_autonomy",
    ]
    assert all(rubric.evaluation_unit == "session" for rubric in SESSION_RUBRICS.values())
    assert all(rubric.version == "v4" for rubric in SESSION_RUBRICS.values())
    assert tuple(item.id for item in build_rubric_catalog().rubrics) == tuple(SESSION_RUBRICS)
    assert REVIEW_POLICY_VERSION == "3"


def test_rubric_prompts_keep_five_anchored_scores_and_insufficient_evidence() -> None:
    for rubric in SESSION_RUBRICS.values():
        assert tuple(rubric.criteria) == ("0", "0.25", "0.5", "0.75", "1")
        assert "insufficient_evidence" in rubric.system_prompt
        assert "Allowed evidence" in rubric.system_prompt

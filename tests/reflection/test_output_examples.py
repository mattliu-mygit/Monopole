from weave_agent_signals.runs.proposals import (
    REFLECTION_PROPOSAL_SCHEMA,
    parse_candidate_proposal,
)
from weave_agent_signals.runs.reflection import (
    _EVALUATOR_SCHEMA,
    _EVALUATOR_USER,
    _GEPA_OBJECTIVE_TEMPLATE,
)


def test_reflection_schemas_include_valid_canonical_examples() -> None:
    proposal = parse_candidate_proposal(REFLECTION_PROPOSAL_SCHEMA.examples[0])
    assert proposal.changes

    evaluation = _EVALUATOR_SCHEMA.examples[0]
    assert 0 <= evaluation["score"] <= 1
    assert evaluation["rationale"]


def test_reflection_prompts_do_not_duplicate_transport_output_contracts() -> None:
    assert "exact JSON layout" not in _GEPA_OBJECTIVE_TEMPLATE
    assert "proposal_layout" not in _GEPA_OBJECTIVE_TEMPLATE
    assert "Respond with JSON" not in _EVALUATOR_USER

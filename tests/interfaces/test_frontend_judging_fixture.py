import json
from pathlib import Path

from weave_agent_signals.judges.sliding import sliding_protocol_contract_manifest
from weave_agent_signals.runs.store import _encode_judging_plan, _judging_plan_matches_cohort


def test_frontend_judging_plan_fixture_matches_persisted_backend_contract() -> None:
    fixture_path = (
        Path(__file__).parents[2] / "frontend" / "tests" / "fixtures" / "judging-plan.json"
    )
    plan = json.loads(fixture_path.read_text())
    cohort = json.loads((fixture_path.parent / "turn-cohort.json").read_text())

    _encode_judging_plan(plan)
    assert plan["protocol"] == sliding_protocol_contract_manifest()
    assert _judging_plan_matches_cohort(plan, cohort)

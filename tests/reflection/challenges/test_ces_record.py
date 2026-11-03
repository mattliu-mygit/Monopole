from __future__ import annotations

import subprocess

import pytest

from weave_agent_signals.ces_catalog import VERDICT, Catalog
from weave_agent_signals.ces_client import CesSession
from weave_agent_signals.runs.challenges.ces_record import record_challenge, record_safely

from .test_contracts import _result


class _FakeClient:
    """The CES client calls record_challenge makes, recorded in order."""

    def __init__(self):
        self.comparisons = []
        self.added = []
        self.writes = []

    def create_comparison(self, members, *, key, attributes=None):
        self.comparisons.append((members, key, attributes))
        return "comparison-1"

    def add_scorers(self, catalog, specs):
        self.added.append(specs)
        return catalog

    def write_scores(self, catalog, pass_id, kind, target_id, scores):
        self.writes.append((pass_id, kind, target_id, scores))
        return ["verdict-score", "verdict-details"]

    def close(self):
        return None


def _with_sessions(result):
    return result.model_copy(
        update={
            arm: getattr(result, arm).model_copy(update={"session_transcript": f"{arm} session"})
            for arm in ("baseline", "candidate")
        }
    )


def test_a_decided_challenge_logs_both_arms_compares_them_and_scores_the_verdict():
    client = _FakeClient()
    logged = []
    result = _with_sessions(_result(winner="candidate"))

    def log(harness, transcript):
        logged.append((harness, transcript))
        return f"case-{len(logged)}"

    recorded = record_challenge(result, CesSession(client, Catalog("ev-t", "ev-s", {})), log=log)

    assert logged == [("codex", "baseline session"), ("codex", "candidate session")]
    assert client.comparisons == [
        (
            [("case-1", "A"), ("case-2", "B")],
            f"ab-comparison-v1:{result.challenge_id}",
            {"challenge_id": result.challenge_id, "candidate_id": "candidate-1"},
        )
    ]
    assert client.added == [(VERDICT,)]
    [(pass_id, kind, target, [verdict])] = client.writes
    assert (pass_id, kind, target) == (result.challenge_id, "comparison", "comparison-1")
    assert (verdict.scorer, verdict.value, verdict.tags, verdict.reason) == (
        "ab.verdict",
        1.0,
        ["candidate"],
        "1 of 1 judges chose candidate",
    )
    assert recorded.ces_record.to_dict() == {
        "baseline_case_id": "case-1",
        "candidate_case_id": "case-2",
        "comparison_id": "comparison-1",
        "verdict_score_id": "verdict-score",
    }
    assert recorded.challenge_id == result.challenge_id


@pytest.mark.parametrize(
    "result",
    [_with_sessions(_result(status="incomplete")), _result()],
    ids=["undecided", "no arm session files"],
)
def test_a_challenge_without_a_verdict_or_arm_sessions_is_not_recorded(result):
    client = _FakeClient()

    recorded = record_challenge(
        result, CesSession(client, None), log=lambda *_args: pytest.fail("must not log")
    )

    assert recorded is result
    assert (client.comparisons, client.writes) == ([], [])


def test_a_recording_failure_keeps_the_result_without_a_record(caplog):
    result = _with_sessions(_result())

    def unavailable():
        raise RuntimeError("CES is down")

    assert record_safely(result, unavailable) is result
    assert "was not recorded in CES: CES is down" in caplog.text


def test_a_failed_adapter_command_logs_its_stderr(caplog):
    result = _with_sessions(_result())

    def adapter_fails():
        raise subprocess.CalledProcessError(
            2, ["weave-agent-adapter"], stderr="no evaluations state\n"
        )

    assert record_safely(result, adapter_fails) is result
    assert "no evaluations state" in caplog.text

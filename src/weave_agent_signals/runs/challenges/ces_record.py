"""Record a decided challenge in CES as two arm sessions, their Comparison, and a verdict."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from collections.abc import Callable

from weave_agent_signals.ces_catalog import VERDICT
from weave_agent_signals.ces_client import CesSession
from weave_agent_signals.models import Score
from weave_agent_signals.runs.challenges.contracts import ChallengeCesRecord, ChallengeResult

ARMS_DATASET = "ab-arms"
_LOG = logging.getLogger(__name__)
_ADAPTER_HARNESS = {"codex": "codex", "claude": "claude-code"}
_VERDICT_VALUE = {"candidate": 1.0, "tie": 0.5, "baseline": 0.0}


def log_transcript(harness: str, transcript: str) -> str:
    """Log one arm's session file with weave-agent-adapter and return its session Case ID."""
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", encoding="utf-8") as handle:
        handle.write(transcript)
        handle.flush()
        completed = subprocess.run(
            [
                "weave-agent-adapter",
                "evaluations",
                "log-transcript",
                "--harness",
                _ADAPTER_HARNESS[harness],
                "--file",
                handle.name,
                "--dataset",
                ARMS_DATASET,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
    return json.loads(completed.stdout)["session_case_id"]


def record_challenge(
    result: ChallengeResult,
    ces: CesSession,
    *,
    log: Callable[[str, str], str] = log_transcript,
) -> ChallengeResult:
    """The result with its CES record, or unchanged when it has no verdict or arm sessions.

    The Comparison key and the verdict's pass ID are the challenge ID, so recording the same
    challenge again replays the same Comparison and Scores."""
    baseline, candidate = result.baseline, result.candidate
    if result.status != "complete" or baseline is None or candidate is None:
        return result
    if not baseline.session_transcript or not candidate.session_transcript:
        return result
    harness = result.execution.harness
    baseline_case = log(harness, baseline.session_transcript)
    candidate_case = log(harness, candidate.session_transcript)
    comparison_id = ces.client.create_comparison(
        [(baseline_case, "A"), (candidate_case, "B")],
        key=f"ab-comparison-v1:{result.challenge_id}",
        attributes={"challenge_id": result.challenge_id, "candidate_id": result.candidate_id},
    )
    agreeing = sum(judge.winner == result.winner for judge in result.judges)
    verdict = Score(
        VERDICT.name,
        _VERDICT_VALUE[result.winner],
        [result.winner],
        {
            "challenge_id": result.challenge_id,
            "candidate_id": result.candidate_id,
            "judge_winners": [judge.winner for judge in result.judges],
        },
        "comparison",
        reason=f"{agreeing} of {len(result.judges)} judges chose {result.winner}",
    )
    catalog = ces.client.add_scorers(ces.catalog, (VERDICT,))
    verdict_score_id, _details_id = ces.client.write_scores(
        catalog, result.challenge_id, "comparison", comparison_id, [verdict]
    )
    return result.model_copy(
        update={
            "ces_record": ChallengeCesRecord(
                baseline_case_id=baseline_case,
                candidate_case_id=candidate_case,
                comparison_id=comparison_id,
                verdict_score_id=verdict_score_id,
            )
        }
    )


def record_safely(
    result: ChallengeResult, ces_factory: Callable[[], CesSession]
) -> ChallengeResult:
    """record_challenge, except a failure keeps the result without a record.

    CES is a record of the challenge, not part of it, so CES or the adapter being unavailable
    must not fail the reflection run."""
    try:
        with ces_factory() as ces:
            return record_challenge(result, ces)
    except Exception as error:
        _LOG.warning("challenge %s was not recorded in CES: %s", result.challenge_id, error)
        return result

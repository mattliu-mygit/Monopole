"""Judge runner — orchestrates rubric + digest + inference → Score."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from weave_agent_signals.judges.digest import (
    build_judge_messages,
    build_session_digest,
    build_turn_digest,
)
from weave_agent_signals.judges.families import model_family, select_judges, select_panel
from weave_agent_signals.judges.inference import InferenceClient
from weave_agent_signals.judges.rubrics import RUBRICS, SESSION_RUBRICS, Rubric
from weave_agent_signals.models import Score, SessionView, TurnSpan

log = logging.getLogger("weave_agent_signals.judges")

DEFAULT_JUDGE_CANDIDATES_WANDB = [
    "gpt-oss-20b",
    "Llama-3.1-8B",
    "granite-4.1-8b",
]

DEFAULT_JUDGE_CANDIDATES_OPENAI = [
    "gpt-4o-mini",
]

# PoLL panels want judges from different model families (arXiv:2404.18796).
# The available roster differs by backend, so selection is backend-aware — a
# W&B model name sent to OpenAI (or vice versa) 404s.
POLL_JUDGE_CANDIDATES_WANDB = [
    "gpt-oss-120b",
    "DeepSeek-V4",
    "Qwen3-30B",
]

POLL_JUDGE_CANDIDATES_OPENAI = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1-mini",
]

_BACKEND_ROSTERS = {
    "wandb": {
        "default": DEFAULT_JUDGE_CANDIDATES_WANDB,
        "poll": POLL_JUDGE_CANDIDATES_WANDB,
        "escalation": "gpt-oss-120b",
    },
    "openai": {
        "default": DEFAULT_JUDGE_CANDIDATES_OPENAI,
        "poll": POLL_JUDGE_CANDIDATES_OPENAI,
        "escalation": "gpt-4o",
    },
    # TEMPORARY local CLI backend (see judges/cli_backend.py): Anthropic (claude),
    # OpenAI (codex) and Google (gemini) families, so a Claude agent gets a full
    # PoLL panel of 1 same-family (claude) + 2 cross-family (codex, gemini).
    "cli": {
        "default": ["claude-sonnet-5", "gpt-5.1", "gemini-2.5-pro"],
        "poll": ["claude-sonnet-5", "gpt-5.1", "gemini-2.5-pro"],
        "escalation": "gpt-5.1",
    },
}


def _roster(client: InferenceClient) -> dict:
    """Model roster (default/poll/escalation) for the client's backend."""
    return _BACKEND_ROSTERS.get(getattr(client, "backend", "openai"), _BACKEND_ROSTERS["openai"])


def judge_default_model(client: InferenceClient) -> str:
    """The most capable judge model for the client's backend.

    Used where judgment quality matters most (e.g. scoring reflector candidates).
    """
    return _roster(client)["escalation"]


def judge_turn(
    turn: TurnSpan,
    client: InferenceClient,
    rubrics: list[Rubric] | None = None,
    judge_candidates: list[str] | None = None,
) -> list[Score]:
    """Run all rubrics against a single turn, return scores."""
    if rubrics is None:
        rubrics = list(RUBRICS.values())
    if judge_candidates is None:
        judge_candidates = _roster(client)["default"]

    judges = select_judges(turn.model, judge_candidates, n=1)
    if not judges:
        log.warning("No eligible judge models for agent model %s", turn.model)
        return []
    judge_model = judges[0]

    digest = build_turn_digest(turn)
    scores: list[Score] = []

    for rubric in rubrics:
        try:
            score = _run_rubric(client, judge_model, rubric, digest, turn)
            if score is not None:
                scores.append(score)
        except Exception as e:
            log.warning("Judge %s failed on rubric %s: %s",
                        judge_model, rubric.scorer_name, e)
    return scores


ESCALATION_SPREAD_THRESHOLD = 0.3


def judge_session(
    session: SessionView,
    client: InferenceClient,
    rubrics: list[Rubric] | None = None,
    judge_candidates: list[str] | None = None,
    panel_size: int = 1,
) -> list[Score]:
    """Run rubrics on a session. panel_size > 1 uses PoLL mean-pooling."""
    if rubrics is None:
        rubrics = list(SESSION_RUBRICS.values())
    roster = _roster(client)
    if judge_candidates is None:
        judge_candidates = roster["poll"] if panel_size > 1 else roster["default"]

    agent_models = {t.model for t in session.turns if t.model}
    agent_model = next(iter(agent_models)) if len(agent_models) == 1 else None

    # A panel (panel_size > 1) uses the PoLL composition: 1 same-family + >=2
    # non-same-family (select_panel warns if the backend can't seat it). A single
    # judge stays strictly cross-family.
    if panel_size > 1:
        judges = select_panel(agent_model, judge_candidates)
    else:
        judges = select_judges(agent_model, judge_candidates, n=1)
    if not judges:
        log.warning("No eligible judge models for session %s", session.conversation_id[:12])
        return []

    digest = build_session_digest(session)
    scores: list[Score] = []

    for rubric in rubrics:
        try:
            if len(judges) == 1:
                score = _run_session_rubric(client, judges[0], rubric, digest, session, agent_model)
            else:
                score = _run_poll_panel(
                    client, judges, rubric, digest, session, agent_model,
                    escalation_model=roster["escalation"],
                )
            if score is not None:
                scores.append(score)
        except Exception as e:
            log.warning("Judge failed on session rubric %s: %s", rubric.scorer_name, e)

    return scores


def _run_poll_panel(
    client: InferenceClient,
    judges: list[str],
    rubric: Rubric,
    digest: str,
    session: SessionView,
    agent_model: str | None,
    escalation_model: str,
) -> Score | None:
    """Run multiple judges and mean-pool their scores (PoLL, arXiv:2404.18796)."""
    messages = build_judge_messages(
        rubric.system_prompt, rubric.criteria_text, digest, granularity="session",
    )
    panel_scores: list[float] = []
    panel_models: list[str] = []       # model as reported by the API
    panel_requested: list[str] = []    # model as requested (for dedup)
    panel_rationales: list[str] = []
    panel_usage: list[dict] = []

    for judge_model in judges:
        try:
            parsed, resp = client.chat_json(
                model=judge_model, messages=messages, temperature=0.0, max_tokens=512,
            )
            raw_score = parsed.get("score")
            if raw_score is None:
                continue
            value = max(0.0, min(1.0, float(raw_score)))
            panel_scores.append(value)
            panel_models.append(resp.model)
            panel_requested.append(judge_model)
            panel_rationales.append(parsed.get("rationale", ""))
            panel_usage.append(resp.usage)
        except Exception as e:
            log.warning("Panel judge %s failed on %s: %s", judge_model, rubric.scorer_name, e)

    if not panel_scores:
        return None

    spread = max(panel_scores) - min(panel_scores) if len(panel_scores) > 1 else 0.0
    escalated = False

    # Dedup on the requested name: the panel already tried these models, and the
    # API may echo a versioned name that wouldn't match the escalation string.
    if spread > ESCALATION_SPREAD_THRESHOLD and escalation_model not in panel_requested:
        try:
            parsed, resp = client.chat_json(
                model=escalation_model, messages=messages, temperature=0.0, max_tokens=512,
            )
            raw_score = parsed.get("score")
            if raw_score is not None:
                value = max(0.0, min(1.0, float(raw_score)))
                panel_scores.append(value)
                panel_models.append(resp.model)
                panel_requested.append(escalation_model)
                panel_rationales.append(parsed.get("rationale", ""))
                panel_usage.append(resp.usage)
                escalated = True
        except Exception as e:
            log.warning("Escalation judge %s failed: %s", escalation_model, e)

    mean_score = sum(panel_scores) / len(panel_scores)
    tags = list(rubric.tags_on_low if mean_score < rubric.threshold else rubric.tags_on_high)
    confidence = min(0.9, 0.6 + 0.1 * len(panel_scores))
    if spread > ESCALATION_SPREAD_THRESHOLD:
        confidence = max(0.5, confidence - 0.1)

    return Score(
        scorer=rubric.scorer_name,
        value=round(mean_score, 4),
        tags=tags,
        confidence=confidence,
        metadata={
            "panel_size": len(panel_scores),
            "panel_models": panel_models,
            "panel_families": [model_family(m) for m in panel_models],
            "panel_scores": panel_scores,
            "panel_rationales": panel_rationales,
            "panel_spread": round(spread, 4),
            "escalated": escalated,
            "agent_model": agent_model,
            "agent_family": model_family(agent_model) if agent_model else "unknown",
            "rubric": rubric.name,
            "usage": panel_usage,
            "scored_at": datetime.now(timezone.utc).isoformat(),
        },
        granularity="session",
        reason=panel_rationales[0] if panel_rationales else "",
    )


def _run_session_rubric(
    client: InferenceClient,
    judge_model: str,
    rubric: Rubric,
    digest: str,
    session: SessionView,
    agent_model: str | None,
) -> Score | None:
    """Run a single judge on a session rubric."""
    messages = build_judge_messages(
        rubric.system_prompt, rubric.criteria_text, digest, granularity="session",
    )
    parsed, resp = client.chat_json(
        model=judge_model, messages=messages, temperature=0.0, max_tokens=512,
    )

    raw_score = parsed.get("score")
    rationale = parsed.get("rationale", "")

    if raw_score is None:
        log.warning("Judge returned no score for %s: %s", rubric.scorer_name, parsed)
        return None

    value = max(0.0, min(1.0, float(raw_score)))
    tags = list(rubric.tags_on_low if value < rubric.threshold else rubric.tags_on_high)

    return Score(
        scorer=rubric.scorer_name,
        value=value,
        tags=tags,
        confidence=0.7,
        metadata={
            "judge_model": resp.model,
            "judge_family": model_family(resp.model),
            "agent_model": agent_model,
            "agent_family": model_family(agent_model) if agent_model else "unknown",
            "rubric": rubric.name,
            "rationale": rationale,
            "usage": resp.usage,
            "scored_at": datetime.now(timezone.utc).isoformat(),
        },
        granularity="session",
        reason=rationale,
    )


def _run_rubric(
    client: InferenceClient,
    judge_model: str,
    rubric: Rubric,
    digest: str,
    turn: TurnSpan,
) -> Score | None:
    messages = build_judge_messages(
        rubric.system_prompt,
        rubric.criteria_text,
        digest,
    )
    parsed, resp = client.chat_json(
        model=judge_model,
        messages=messages,
        temperature=0.0,
        max_tokens=512,
    )

    raw_score = parsed.get("score")
    rationale = parsed.get("rationale", "")

    if raw_score is None:
        log.warning("Judge returned no score for %s: %s", rubric.scorer_name, parsed)
        return None

    value = max(0.0, min(1.0, float(raw_score)))
    tags = list(rubric.tags_on_low if value < rubric.threshold else rubric.tags_on_high)

    return Score(
        scorer=rubric.scorer_name,
        value=value,
        tags=tags,
        confidence=0.7,
        metadata={
            "judge_model": resp.model,
            "judge_family": model_family(resp.model),
            "agent_model": turn.model,
            "agent_family": model_family(turn.model) if turn.model else "unknown",
            "rubric": rubric.name,
            "rationale": rationale,
            "usage": resp.usage,
            "scored_at": datetime.now(timezone.utc).isoformat(),
        },
        granularity="turn",
        reason=rationale,
    )

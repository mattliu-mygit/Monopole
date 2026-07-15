from collections import defaultdict
from urllib.parse import quote, unquote, urlparse

from weave_signal_monitoring.catalog import RECOMMENDATION_THRESHOLD
from weave_signal_monitoring.models import (
    FeedbackRecord,
    FlaggedConversation,
    SignalEvidence,
    SignalIdentity,
    TurnRecord,
)


class HydrationError(RuntimeError):
    pass


def _turn_id_from_ref(ref: str) -> str:
    parts = [unquote(part) for part in urlparse(ref).path.split("/") if part]
    if len(parts) < 2 or parts[-2] not in {"call", "agent_turn"} or not parts[-1]:
        raise HydrationError(f"eligible feedback has unsupported turn ref: {ref!r}")
    return parts[-1]


def _index_turns(turns: list[TurnRecord]) -> dict[str, TurnRecord]:
    indexed: dict[str, TurnRecord] = {}
    for turn in turns:
        existing = indexed.get(turn.turn_id)
        if existing is not None and existing != turn:
            raise HydrationError(f"inconsistent duplicate turn identity: {turn.turn_id}")
        indexed[turn.turn_id] = turn
    return indexed


def _bounded_preview(value: str, limit: int = 80) -> str:
    normalized = " ".join(value.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3].rstrip()}..."


def _display_name(conversation_id: str, turns: list[TurnRecord]) -> str:
    named = [turn for turn in turns if turn.display_name and turn.display_name.strip()]
    if named:
        return max(named, key=lambda turn: (turn.started_at, turn.turn_id)).display_name.strip()
    for turn in sorted(turns, key=lambda item: (item.started_at, item.turn_id)):
        if turn.user_request and turn.user_request.strip():
            return _bounded_preview(turn.user_request)
    return f"Conversation {conversation_id}"


def hydrate_conversations(
    *,
    identities: tuple[SignalIdentity, ...],
    feedback: list[FeedbackRecord],
    triggering_turns: list[TurnRecord],
    conversation_turns: list[TurnRecord],
    entity: str,
    project: str,
) -> list[FlaggedConversation]:
    identities_by_ref = {identity.scorer_ref: identity for identity in identities}
    if len(identities_by_ref) != len(identities):
        raise HydrationError("signal catalog contains duplicate scorer identities")

    feedback_by_id: dict[str, FeedbackRecord] = {}
    for row in feedback:
        existing = feedback_by_id.get(row.id)
        if existing is not None and existing != row:
            raise HydrationError(f"inconsistent duplicate feedback identity: {row.id}")
        feedback_by_id[row.id] = row

    selected_by_signal_turn: dict[tuple[str, str], FeedbackRecord] = {}
    for row in feedback_by_id.values():
        if row.runnable_ref not in identities_by_ref:
            continue
        if row.output.rating > RECOMMENDATION_THRESHOLD:
            continue
        turn_id = _turn_id_from_ref(row.weave_ref)
        key = (row.runnable_ref, turn_id)
        existing = selected_by_signal_turn.get(key)
        if existing is None or (row.created_at, row.id) > (existing.created_at, existing.id):
            selected_by_signal_turn[key] = row

    if not selected_by_signal_turn:
        return []

    triggering_by_id = _index_turns(triggering_turns)
    all_turns_by_id = _index_turns([*conversation_turns, *triggering_turns])
    evidence_by_conversation: dict[str, list[SignalEvidence]] = defaultdict(list)

    for (scorer_ref, turn_id), row in selected_by_signal_turn.items():
        turn = triggering_by_id.get(turn_id)
        if turn is None:
            raise HydrationError(f"missing triggering turn: {turn_id}")
        if not turn.conversation_id:
            raise HydrationError(f"turn {turn_id!r} has no conversation identity")
        identity = identities_by_ref[scorer_ref]
        evidence_by_conversation[turn.conversation_id].append(
            SignalEvidence(
                signal=identity.slug,
                version=identity.version,
                rating=row.output.rating,
                reason=row.output.reason,
                turn_id=turn_id,
                turn_started_at=turn.started_at,
            )
        )

    turns_by_conversation: dict[str, list[TurnRecord]] = defaultdict(list)
    for turn in all_turns_by_id.values():
        if turn.conversation_id in evidence_by_conversation:
            turns_by_conversation[turn.conversation_id].append(turn)

    conversations: list[FlaggedConversation] = []
    for conversation_id, evidence in evidence_by_conversation.items():
        turns = turns_by_conversation.get(conversation_id, [])
        if not turns:
            raise HydrationError(f"conversation {conversation_id!r} has no hydrated turns")
        evidence.sort(key=lambda item: (item.turn_started_at, item.signal, item.turn_id))
        triggering_ids = list(dict.fromkeys(item.turn_id for item in evidence))
        first_trigger = triggering_ids[0]
        conversations.append(
            FlaggedConversation(
                conversation_id=conversation_id,
                display_name=_display_name(conversation_id, turns),
                started_at=min(turn.started_at for turn in turns),
                last_activity_at=max(turn.ended_at or turn.started_at for turn in turns),
                lowest_rating=min(item.rating for item in evidence),
                signals=evidence,
                triggering_turn_ids=triggering_ids,
                wandb_url=(
                    f"https://wandb.ai/{quote(entity, safe='')}/{quote(project, safe='')}"
                    f"/r/call/{quote(first_trigger, safe='')}"
                ),
            )
        )

    conversations.sort(
        key=lambda item: (
            item.lowest_rating,
            -item.last_activity_at.timestamp(),
            item.conversation_id,
        )
    )
    return conversations

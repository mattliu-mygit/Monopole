import argparse
import json
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from weave_signal_monitoring.catalog import CATALOG
from weave_signal_monitoring.hydration import hydrate_conversations
from weave_signal_monitoring.models import (
    FlaggedConversation,
    HydrationEnvelope,
    HydrationWindow,
)
from weave_signal_monitoring.weave_gateway import (
    WeaveGateway,
    install_catalog,
)

DEFAULT_ENTITY = "weave-team"
DEFAULT_PROJECT = "agent-sessions"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="weave-signal-monitor")
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("install", help="install the exact versioned signal catalog")
    hydrate = commands.add_parser("hydrate", help="list conversations with low signals")
    hydrate.add_argument("--since", help="inclusive ISO-8601 feedback time")
    hydrate.add_argument("--until", help="inclusive ISO-8601 feedback time")
    hydrate.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timezone is required")
    return parsed.astimezone(timezone.utc)


def _print_error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, str) and value.endswith("+00:00"):
        return f"{value[:-6]}Z"
    return value


def _render_json(envelope: HydrationEnvelope) -> str:
    value = _json_value(envelope.model_dump(mode="json"))
    return json.dumps(value, indent=2) + "\n"


def _table_rows(conversations: list[FlaggedConversation]) -> list[list[str]]:
    rows: list[list[str]] = []
    for conversation in conversations:
        signals = ", ".join(dict.fromkeys(item.signal for item in conversation.signals))
        last_activity = (
            conversation.last_activity_at.astimezone(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        rows.append(
            [
                f"{conversation.lowest_rating:.2f}",
                conversation.display_name,
                signals,
                last_activity,
                conversation.wandb_url,
            ]
        )
    return rows


def _render_table(conversations: list[FlaggedConversation]) -> str:
    if not conversations:
        return "No low-signal conversations.\n"
    headers = ["RATING", "CONVERSATION", "SIGNALS", "LAST ACTIVITY", "W&B"]
    rows = _table_rows(conversations)
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def line(values: list[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(values)).rstrip()

    separator = ["-" * width for width in widths]
    return "\n".join([line(headers), line(separator), *(line(row) for row in rows)]) + "\n"


def run(
    argv: Sequence[str] | None = None,
    *,
    gateway_factory: Callable[[str, str], Any] = WeaveGateway,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> int:
    args = _parser().parse_args(argv)
    current = now().astimezone(timezone.utc)

    if args.command == "hydrate":
        try:
            since = _utc_datetime(args.since) if args.since else current - timedelta(hours=24)
            until = _utc_datetime(args.until) if args.until else current
        except (TypeError, ValueError):
            return _print_error("invalid time window")
        if since >= until:
            return _print_error("invalid time window")

    try:
        gateway = gateway_factory(args.entity, args.project)
    except Exception:
        return _print_error("initialization failed")

    if args.command == "install":
        try:
            report = install_catalog(gateway, CATALOG)
        except Exception:
            return _print_error("install failed")
        lines = [*(f"created {name}" for name in report.created)]
        lines.extend(f"reused {name}" for name in report.reused)
        if lines:
            print("\n".join(lines))
        return 0

    try:
        identities = gateway.resolve_identities(CATALOG)
    except Exception:
        return _print_error("monitor identity resolution failed")
    try:
        feedback = gateway.read_feedback(identities, since, until)
    except Exception:
        return _print_error("feedback query failed")
    try:
        triggering_turns = gateway.read_turns(tuple(row.weave_ref for row in feedback))
        conversation_ids = tuple(dict.fromkeys(turn.conversation_id for turn in triggering_turns))
        conversation_turns = gateway.read_conversation_turns(conversation_ids)
        conversations = hydrate_conversations(
            identities=identities,
            feedback=list(feedback),
            triggering_turns=list(triggering_turns),
            conversation_turns=list(conversation_turns),
            entity=args.entity,
            project=args.project,
        )
    except Exception:
        return _print_error("turn hydration failed")
    try:
        envelope = HydrationEnvelope(
            generated_at=current,
            entity=args.entity,
            project=args.project,
            window=HydrationWindow(since=since, until=until),
            conversations=conversations,
        )
        rendered = _render_json(envelope) if args.as_json else _render_table(conversations)
    except Exception:
        return _print_error("render validation failed")
    sys.stdout.write(rendered)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()

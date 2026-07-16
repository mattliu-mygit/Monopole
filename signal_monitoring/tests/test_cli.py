import json
from datetime import datetime, timedelta, timezone

from weave_signal_monitoring.catalog import CATALOG
from weave_signal_monitoring.cli import run
from weave_signal_monitoring.models import FeedbackRecord, SignalIdentity, TurnRecord
from weave_signal_monitoring.weave_gateway import InstallReport

NOW = datetime(2026, 7, 15, 20, 0, tzinfo=timezone.utc)


class FakeGateway:
    def __init__(self, entity, project, *, empty=False, fail_at=None):
        self.entity = entity
        self.project = project
        self.empty = empty
        self.fail_at = fail_at
        self.feedback_window = None
        self.created = []

    def get_definition(self, name):
        if name == CATALOG[-1].monitor_name:
            from weave_signal_monitoring.weave_gateway import definition_fingerprint

            return definition_fingerprint(CATALOG[-1])
        return None

    def create(self, definition, fingerprint):
        self.created.append(definition.monitor_name)

    def resolve_identities(self, catalog):
        if self.fail_at == "identity":
            raise RuntimeError("secret identity details")
        return (
            SignalIdentity(
                slug="user-frustration",
                version="v1",
                monitor_name="agent-signal-user-frustration-v1",
                scorer_ref="weave:///e/p/object/scorer:digest",
            ),
        )

    def read_feedback(self, identities, since, until):
        if self.fail_at == "feedback":
            raise RuntimeError("secret feedback details")
        self.feedback_window = (since, until)
        if self.empty:
            return ()
        return (
            FeedbackRecord(
                id="feedback-1",
                weave_ref="weave:///e/p/call/turn-1",
                runnable_ref=identities[0].scorer_ref,
                created_at=NOW,
                output={"rating": 0.25, "reason": "The user explicitly says this is wrong."},
            ),
        )

    def read_turns(self, refs):
        return (
            TurnRecord(
                turn_id="turn-1",
                conversation_id="conversation-1",
                started_at=NOW - timedelta(minutes=2),
                ended_at=NOW - timedelta(minutes=1),
                display_name="Deployment fix",
                user_request="Fix deployment",
            ),
        )

    def read_conversation_turns(self, conversation_ids):
        return self.read_turns(())


class GatewayFactory:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.instances = []

    def __call__(self, entity, project):
        instance = FakeGateway(entity, project, **self.kwargs)
        self.instances.append(instance)
        return instance


def test_hydrate_json_uses_24_hour_default_and_schema_one(capsys):
    factory = GatewayFactory()

    assert run(["hydrate", "--json"], gateway_factory=factory, now=lambda: NOW) == 0

    body = json.loads(capsys.readouterr().out)
    assert body["schema_version"] == 1
    assert body["window"] == {
        "since": "2026-07-14T20:00:00Z",
        "until": "2026-07-15T20:00:00Z",
    }
    assert list(body["conversations"][0]) == [
        "conversation_id",
        "display_name",
        "started_at",
        "last_activity_at",
        "lowest_rating",
        "signals",
        "triggering_turn_ids",
        "wandb_url",
    ]
    assert factory.instances[0].feedback_window == (NOW - timedelta(hours=24), NOW)


def test_hydrate_accepts_explicit_utc_window_and_scope(capsys):
    factory = GatewayFactory(empty=True)

    result = run(
        [
            "--entity",
            "custom-entity",
            "--project",
            "custom-project",
            "hydrate",
            "--since",
            "2026-07-15T18:00:00Z",
            "--until",
            "2026-07-15T19:00:00Z",
            "--json",
        ],
        gateway_factory=factory,
        now=lambda: NOW,
    )

    assert result == 0
    assert (factory.instances[0].entity, factory.instances[0].project) == (
        "custom-entity",
        "custom-project",
    )
    body = json.loads(capsys.readouterr().out)
    assert body["window"]["since"] == "2026-07-15T18:00:00Z"


def test_table_contains_only_review_fields(capsys):
    assert run(["hydrate"], gateway_factory=GatewayFactory(), now=lambda: NOW) == 0

    output = capsys.readouterr().out
    assert "RATING" in output and "CONVERSATION" in output
    assert "SIGNALS" in output and "LAST ACTIVITY" in output and "W&B" in output
    assert "Deployment fix" in output
    assert "user-frustration" in output


def test_empty_table_is_a_success(capsys):
    assert run(["hydrate"], gateway_factory=GatewayFactory(empty=True), now=lambda: NOW) == 0
    assert capsys.readouterr().out == "No low-signal conversations.\n"


def test_install_reports_created_and_reused(capsys, monkeypatch):
    monkeypatch.setattr(
        "weave_signal_monitoring.cli.install_catalog",
        lambda gateway, catalog: InstallReport(
            created=(CATALOG[0].monitor_name,), reused=(CATALOG[-1].monitor_name,)
        ),
    )

    assert run(["install"], gateway_factory=GatewayFactory(), now=lambda: NOW) == 0

    assert capsys.readouterr().out.splitlines() == [
        "created agent-signal-user-frustration-v1",
        "reused agent-signal-low-quality-response-v1",
    ]


def test_external_failure_is_sanitized_and_emits_no_partial_json(capsys):
    result = run(
        ["hydrate", "--json"],
        gateway_factory=GatewayFactory(fail_at="feedback"),
        now=lambda: NOW,
    )

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err == "error: feedback query failed\n"
    assert "secret" not in captured.err


def test_invalid_window_fails_before_gateway_creation(capsys):
    factory = GatewayFactory()

    result = run(
        [
            "hydrate",
            "--since",
            "2026-07-15T20:00:00Z",
            "--until",
            "2026-07-15T19:00:00Z",
        ],
        gateway_factory=factory,
        now=lambda: NOW,
    )

    assert result == 1
    assert factory.instances == []
    assert capsys.readouterr().err == "error: invalid time window\n"

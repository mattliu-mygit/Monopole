from types import SimpleNamespace

import pytest

from weave_signal_monitoring.catalog import CATALOG
from weave_signal_monitoring.weave_gateway import (
    SIGNAL_TRACE_ROLE,
    TRACE_ROLE_ATTRIBUTE,
    DefinitionConflict,
    LLMAsAJudgeScorer,
    WeaveGateway,
    build_monitor,
    definition_fingerprint,
    install_catalog,
)


class FakeMonitorStore:
    def __init__(self, existing=None, fail_on=None):
        self.existing = dict(existing or {})
        self.fail_on = fail_on
        self.created = []

    def get_definition(self, name):
        return self.existing.get(name)

    def create(self, definition, fingerprint):
        if definition.monitor_name == self.fail_on:
            raise RuntimeError("external create failed")
        self.created.append(definition.monitor_name)
        self.existing[definition.monitor_name] = fingerprint


def test_install_creates_each_missing_monitor_once():
    store = FakeMonitorStore()

    report = install_catalog(store, CATALOG)

    assert report.created == tuple(signal.monitor_name for signal in CATALOG)
    assert report.reused == ()


def test_install_reuses_exact_definitions():
    first = FakeMonitorStore()
    installed = install_catalog(first, CATALOG)
    second = FakeMonitorStore(first.existing)

    report = install_catalog(second, CATALOG)

    assert report.reused == installed.created
    assert report.created == ()
    assert second.created == []


def test_install_rejects_conflicting_definition_without_overwrite():
    store = FakeMonitorStore({CATALOG[0].monitor_name: "different"})

    with pytest.raises(DefinitionConflict, match=CATALOG[0].monitor_name):
        install_catalog(store, CATALOG)

    assert store.created == []
    assert store.existing[CATALOG[0].monitor_name] == "different"


def test_partial_failure_is_resumable_without_rollback():
    failing = FakeMonitorStore(fail_on=CATALOG[2].monitor_name)
    with pytest.raises(RuntimeError, match="external create failed"):
        install_catalog(failing, CATALOG)
    assert failing.created == [CATALOG[0].monitor_name, CATALOG[1].monitor_name]

    failing.fail_on = None
    report = install_catalog(failing, CATALOG)

    assert report.reused == tuple(signal.monitor_name for signal in CATALOG[:2])
    assert report.created == tuple(signal.monitor_name for signal in CATALOG[2:])


def test_definition_fingerprint_changes_with_monitor_behavior():
    baseline = definition_fingerprint(CATALOG[0])
    changed = CATALOG[0].__class__(
        slug=CATALOG[0].slug,
        version=CATALOG[0].version,
        description=CATALOG[0].description,
        rubric=f"{CATALOG[0].rubric} More strict.",
    )

    assert len(baseline) == 64
    assert definition_fingerprint(changed) != baseline


def test_build_monitor_uses_supported_agent_llm_judge():
    definition = CATALOG[0]
    fingerprint = definition_fingerprint(definition)

    monitor = build_monitor(definition, fingerprint)

    assert monitor.name == definition.monitor_name
    assert monitor.active is False
    assert monitor.sampling_rate == 1.0
    assert monitor.op_names == ["weave.genai.turn_ended"]
    assert monitor.description.endswith(f"catalog_sha256={fingerprint}")
    assert len(monitor.scorers) == 1
    scorer = monitor.scorers[0]
    assert isinstance(scorer, LLMAsAJudgeScorer)
    assert scorer.name == definition.scorer_name
    assert scorer.name.endswith("-scorer")
    assert scorer.model.name == definition.model_name
    assert scorer.model.llm_model_id == "coreweave/openai/gpt-oss-20b"
    assert scorer.model.default_params.response_format == "json_object"
    assert scorer.score.attributes == {TRACE_ROLE_ATTRIBUTE: SIGNAL_TRACE_ROLE}


def test_definition_fingerprint_authenticates_signal_trace_role(monkeypatch):
    baseline = definition_fingerprint(CATALOG[0])

    monkeypatch.setattr(
        "weave_signal_monitoring.weave_gateway.SIGNAL_TRACE_ROLE",
        "other_system",
    )

    assert definition_fingerprint(CATALOG[0]) != baseline


class FakeServer:
    def __init__(self, objects):
        self.objects = objects
        self.requests = []

    def objs_query(self, request):
        self.requests.append(request)
        names = set(request.filter.object_ids or [])
        return SimpleNamespace(objs=[item for item in self.objects if item.object_id in names])


def monitor_object(name, fingerprint, scorer_ref=None):
    return SimpleNamespace(
        object_id=name,
        val={
            "description": f"description\ncatalog_sha256={fingerprint}",
            "active": True,
            "scorers": [scorer_ref] if scorer_ref else [],
        },
    )


def test_gateway_reads_exact_latest_monitor_definition():
    fingerprint = definition_fingerprint(CATALOG[0])
    server = FakeServer([monitor_object(CATALOG[0].monitor_name, fingerprint)])
    gateway = WeaveGateway("entity", "project", client=SimpleNamespace(server=server))

    assert gateway.get_definition(CATALOG[0].monitor_name) == fingerprint
    request = server.requests[0]
    assert request.project_id == "entity/project"
    assert request.filter.object_ids == [CATALOG[0].monitor_name]
    assert request.filter.leaf_object_classes == ["Monitor"]
    assert request.filter.latest_only is True


def test_gateway_rejects_ambiguous_or_malformed_monitor_objects():
    fingerprint = definition_fingerprint(CATALOG[0])
    name = CATALOG[0].monitor_name
    gateway = WeaveGateway(
        "entity",
        "project",
        client=SimpleNamespace(
            server=FakeServer(
                [monitor_object(name, fingerprint), monitor_object(name, fingerprint)]
            )
        ),
    )
    with pytest.raises(DefinitionConflict, match="multiple latest"):
        gateway.get_definition(name)

    malformed = SimpleNamespace(object_id=name, val={"description": "no fingerprint"})
    gateway = WeaveGateway(
        "entity", "project", client=SimpleNamespace(server=FakeServer([malformed]))
    )
    with pytest.raises(DefinitionConflict, match="fingerprint"):
        gateway.get_definition(name)


def test_gateway_resolves_exact_scorer_refs_in_catalog_order():
    objects = []
    expected_refs = []
    for index, definition in enumerate(CATALOG):
        scorer_ref = f"weave:///entity/project/object/{definition.scorer_name}:digest-{index}"
        expected_refs.append(scorer_ref)
        objects.append(
            monitor_object(
                definition.monitor_name,
                definition_fingerprint(definition),
                scorer_ref,
            )
        )
    server = FakeServer(objects)
    gateway = WeaveGateway("entity", "project", client=SimpleNamespace(server=server))

    identities = gateway.resolve_identities(CATALOG)

    assert [identity.scorer_ref for identity in identities] == expected_refs
    assert [identity.slug for identity in identities] == [item.slug for item in CATALOG]


def test_gateway_create_activates_monitor_and_verifies_saved_fingerprint(monkeypatch):
    definition = CATALOG[0]
    fingerprint = definition_fingerprint(definition)

    class FakeMonitor:
        def __init__(self):
            self.activated = False

        def activate(self):
            self.activated = True

    monitor = FakeMonitor()
    monkeypatch.setattr(
        "weave_signal_monitoring.weave_gateway.build_monitor",
        lambda actual_definition, actual_fingerprint: monitor,
    )
    gateway = WeaveGateway("entity", "project", client=SimpleNamespace(server=FakeServer([])))
    monkeypatch.setattr(gateway, "get_definition", lambda name: fingerprint)

    gateway.create(definition, fingerprint)

    assert monitor.activated is True


def test_gateway_create_rejects_failed_postcondition(monkeypatch):
    definition = CATALOG[0]
    fingerprint = definition_fingerprint(definition)
    monitor = SimpleNamespace(activate=lambda: None)
    monkeypatch.setattr(
        "weave_signal_monitoring.weave_gateway.build_monitor",
        lambda actual_definition, actual_fingerprint: monitor,
    )
    gateway = WeaveGateway("entity", "project", client=SimpleNamespace(server=FakeServer([])))
    monkeypatch.setattr(gateway, "get_definition", lambda name: "different")

    with pytest.raises(DefinitionConflict, match="could not be verified"):
        gateway.create(definition, fingerprint)

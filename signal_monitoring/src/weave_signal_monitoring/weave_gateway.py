import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import weave
from weave import Monitor
from weave.flow.scorer import Scorer
from weave.trace.objectify import register_object
from weave.trace.op import op
from weave.trace_server.interface.builtin_object_classes.llm_structured_model import (
    LLMStructuredCompletionModel,
)
from weave.trace_server.trace_server_interface import ObjectVersionFilter, ObjQueryReq

from weave_signal_monitoring.catalog import (
    DEFAULT_MODEL,
    SAMPLING_RATE,
    TURN_OP_NAME,
    SignalDefinition,
)
from weave_signal_monitoring.models import SignalIdentity

_FINGERPRINT_RE = re.compile(r"^catalog_sha256=([0-9a-f]{64})$")


class DefinitionConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class InstallReport:
    created: tuple[str, ...]
    reused: tuple[str, ...]


class MonitorStore(Protocol):
    def get_definition(self, name: str) -> str | None: ...

    def create(self, definition: SignalDefinition, fingerprint: str) -> None: ...


@register_object
class SignalJudgeScorer(Scorer):
    model: LLMStructuredCompletionModel
    scoring_prompt: str

    @op
    def score(self, *, output: Any, **kwargs: Any) -> Any:
        prompt = self.scoring_prompt.format(output=output, **kwargs)
        return self.model.predict([{"role": "user", "content": prompt}])


def definition_fingerprint(definition: SignalDefinition) -> str:
    payload = {
        **asdict(definition),
        "monitor_name": definition.monitor_name,
        "scorer_name": definition.scorer_name,
        "scoring_prompt": definition.scoring_prompt,
        "model": DEFAULT_MODEL,
        "op_name": TURN_OP_NAME,
        "sampling_rate": SAMPLING_RATE,
        "active": True,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_monitor(definition: SignalDefinition, fingerprint: str) -> Monitor:
    model = LLMStructuredCompletionModel(
        name=f"{definition.scorer_name}-model",
        llm_model_id=DEFAULT_MODEL,
        default_params={"temperature": 0, "response_format": "json_object"},
    )
    scorer = SignalJudgeScorer(
        name=definition.scorer_name,
        model=model,
        scoring_prompt=definition.scoring_prompt,
    )
    return Monitor(
        name=definition.monitor_name,
        description=f"{definition.description}\ncatalog_sha256={fingerprint}",
        sampling_rate=SAMPLING_RATE,
        scorers=[scorer],
        op_names=[TURN_OP_NAME],
        active=False,
    )


def install_catalog(
    store: MonitorStore,
    catalog: tuple[SignalDefinition, ...],
) -> InstallReport:
    created: list[str] = []
    reused: list[str] = []
    for definition in catalog:
        fingerprint = definition_fingerprint(definition)
        existing = store.get_definition(definition.monitor_name)
        if existing == fingerprint:
            reused.append(definition.monitor_name)
            continue
        if existing is not None:
            raise DefinitionConflict(
                f"monitor {definition.monitor_name!r} exists with a conflicting definition"
            )
        store.create(definition, fingerprint)
        created.append(definition.monitor_name)
    return InstallReport(tuple(created), tuple(reused))


class WeaveGateway:
    def __init__(self, entity: str, project: str, *, client: Any | None = None):
        self.entity = entity
        self.project = project
        self.project_id = f"{entity}/{project}"
        self.client = client if client is not None else weave.init(self.project_id)

    def _read_monitor(self, name: str) -> Any | None:
        response = self.client.server.objs_query(
            ObjQueryReq(
                project_id=self.project_id,
                filter=ObjectVersionFilter(
                    object_ids=[name],
                    leaf_object_classes=["Monitor"],
                    latest_only=True,
                ),
            )
        )
        if not response.objs:
            return None
        if len(response.objs) != 1:
            raise DefinitionConflict(f"monitor {name!r} has multiple latest objects")
        return response.objs[0]

    @staticmethod
    def _fingerprint(monitor: Any, name: str) -> str:
        val = monitor.val
        if not isinstance(val, dict):
            raise DefinitionConflict(f"monitor {name!r} has a malformed definition")
        description = val.get("description")
        if not isinstance(description, str):
            raise DefinitionConflict(f"monitor {name!r} has no catalog fingerprint")
        matches = [
            match.group(1)
            for line in description.splitlines()
            if (match := _FINGERPRINT_RE.fullmatch(line))
        ]
        if len(matches) != 1:
            raise DefinitionConflict(f"monitor {name!r} has an invalid catalog fingerprint")
        return matches[0]

    def get_definition(self, name: str) -> str | None:
        monitor = self._read_monitor(name)
        if monitor is None:
            return None
        return self._fingerprint(monitor, name)

    def create(self, definition: SignalDefinition, fingerprint: str) -> None:
        monitor = build_monitor(definition, fingerprint)
        monitor.activate()
        if self.get_definition(definition.monitor_name) != fingerprint:
            raise DefinitionConflict(
                f"monitor {definition.monitor_name!r} could not be verified after activation"
            )

    def resolve_identities(
        self,
        catalog: tuple[SignalDefinition, ...],
    ) -> tuple[SignalIdentity, ...]:
        identities: list[SignalIdentity] = []
        for definition in catalog:
            monitor = self._read_monitor(definition.monitor_name)
            if monitor is None:
                raise DefinitionConflict(f"monitor {definition.monitor_name!r} is not installed")
            actual = self._fingerprint(monitor, definition.monitor_name)
            expected = definition_fingerprint(definition)
            if actual != expected:
                raise DefinitionConflict(
                    f"monitor {definition.monitor_name!r} has a conflicting definition"
                )
            scorers = monitor.val.get("scorers")
            if not isinstance(scorers, list) or len(scorers) != 1:
                raise DefinitionConflict(
                    f"monitor {definition.monitor_name!r} must contain exactly one scorer"
                )
            scorer_ref = scorers[0]
            if not isinstance(scorer_ref, str) or not scorer_ref.startswith("weave:///"):
                raise DefinitionConflict(
                    f"monitor {definition.monitor_name!r} contains a non-reference scorer"
                )
            identities.append(
                SignalIdentity(
                    slug=definition.slug,
                    version=definition.version,
                    monitor_name=definition.monitor_name,
                    scorer_ref=scorer_ref,
                )
            )
        return tuple(identities)

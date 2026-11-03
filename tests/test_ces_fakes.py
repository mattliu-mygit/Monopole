"""The hand-written CES fakes keep CesClient's call shapes, so tests can't pass on a fake that
production's client would reject."""

from __future__ import annotations

import inspect

import pytest

from tests.interfaces.routes.test_inspection import FakeCes as InspectionFake
from tests.interfaces.test_ces_commands import FakeCes as CommandsFake
from tests.interfaces.test_ces_commands import FakeJudgeCes
from tests.reflection.challenges.test_ces_record import _FakeClient as RecordFake
from tests.runs.stages.test_judging import _Ces as JudgingFake
from tests.runs.stages.test_scoring import TrackingCes
from tests.runs.test_cohort import FakeCes as CohortFake
from weave_agent_signals.ces_client import CesClient


def _shape(function) -> tuple[int, list[str]]:
    """How a call binds: the positional parameters after self, and the keyword-only names."""
    params = list(inspect.signature(function).parameters.values())[1:]
    positional = sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params)
    return positional, sorted(p.name for p in params if p.kind is p.KEYWORD_ONLY)


@pytest.mark.parametrize(
    "fake",
    [InspectionFake, CommandsFake, FakeJudgeCes, RecordFake, JudgingFake, TrackingCes, CohortFake],
    ids=lambda fake: f"{fake.__module__.rsplit('.', 1)[-1]}.{fake.__name__}",
)
def test_a_ces_fake_keeps_the_real_clients_call_shapes(fake):
    shared = [
        name
        for name, method in inspect.getmembers(fake, inspect.isfunction)
        if not name.startswith("_") and callable(getattr(CesClient, name, None))
    ]

    assert shared, "the fake implements none of CesClient's methods"
    for name in shared:
        assert _shape(getattr(fake, name)) == _shape(getattr(CesClient, name)), name

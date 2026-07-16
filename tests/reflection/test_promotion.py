"""Focused tests for registry-bounded, per-file promotion."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from weave_agent_signals.runs.bundles import BundleSnapshot, TargetSnapshot
from weave_agent_signals.runs.promotion import (
    PromotionApplyError,
    PromotionReceipt,
    StaleBaseError,
    TargetPromoter,
)
from weave_agent_signals.runs.targets import load_target_registry


def _registry(tmp_path: Path) -> Path:
    path = tmp_path / "targets.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "targets": [
                    {"kind": "file", "id": "agents", "path": "AGENTS.md"},
                    {"kind": "file", "id": "soul", "path": "SOUL.md"},
                    {"kind": "skill_collection", "id": "skills", "root": "skills"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _candidate(baseline: BundleSnapshot, **contents: str) -> BundleSnapshot:
    targets = {target.locator: target for target in baseline.targets}
    for locator, content in contents.items():
        previous = targets.get(locator)
        targets[locator] = TargetSnapshot(
            kind="file",
            locator=locator,
            exists=True,
            content=content,
            display_name=previous.display_name if previous else None,
            path=previous.path if previous else None,
        )
    return BundleSnapshot(tuple(targets.values()), baseline.scope)


def _promote(promoter: TargetPromoter, baseline: BundleSnapshot, requested: BundleSnapshot):
    return promoter.promote(
        promotion_id="promotion-1",
        run_id="run-1",
        candidate_id="candidate-1",
        past=baseline,
        evaluated_candidate=requested,
        promoted=requested,
        review_revision=2,
        acknowledge_unevaluated=False,
    )


def test_updates_exact_files_without_git(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("old", encoding="utf-8")
    promoter = TargetPromoter(load_target_registry(_registry(tmp_path)))
    baseline = promoter.capture()
    requested = _candidate(baseline, **{"file:agents": "new"})

    receipt = _promote(promoter, baseline, requested)

    assert (tmp_path / "AGENTS.md").read_text(encoding="utf-8") == "new"
    assert receipt.status == "applied"
    assert receipt.applied_locators == ("file:agents",)


def test_creates_only_admitted_skill_files(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("same", encoding="utf-8")
    promoter = TargetPromoter(load_target_registry(_registry(tmp_path)))
    baseline = promoter.capture()
    requested = _candidate(baseline, **{"skills:skills/new-skill/SKILL.md": "# New"})

    receipt = _promote(promoter, baseline, requested)

    assert (tmp_path / "skills/new-skill/SKILL.md").read_text(encoding="utf-8") == "# New"
    assert receipt.status == "applied"


def test_recursive_create_registers_target_before_publication(tmp_path: Path) -> None:
    registry_path = tmp_path / "targets.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "targets": [
                    {
                        "kind": "markdown_root",
                        "id": "repo",
                        "root": "repo",
                        "files": ["AGENTS.md"],
                        "allow_create": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "repo").mkdir()
    (tmp_path / "repo" / "AGENTS.md").write_text("old", encoding="utf-8")
    promoter = TargetPromoter(load_target_registry(registry_path))
    baseline = promoter.capture()
    requested = _candidate(
        baseline,
        **{"markdown:repo/skills/new/SKILL.md": "# New"},
    )

    receipt = _promote(promoter, baseline, requested)

    assert receipt.status == "applied"
    assert (tmp_path / "repo/skills/new/SKILL.md").read_text(encoding="utf-8") == "# New"
    document = json.loads(registry_path.read_text(encoding="utf-8"))
    assert document["targets"][0]["files"] == ["AGENTS.md", "skills/new/SKILL.md"]
    assert load_target_registry(registry_path).capture().locators == (
        "markdown:repo/AGENTS.md",
        "markdown:repo/skills/new/SKILL.md",
    )


def test_registry_drift_blocks_recursive_create_before_file_publication(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "targets.json"
    original = {
        "schema_version": "1",
        "targets": [
            {
                "kind": "markdown_root",
                "id": "repo",
                "root": "repo",
                "files": [],
                "allow_create": True,
            }
        ],
    }
    registry_path.write_text(json.dumps(original), encoding="utf-8")
    (tmp_path / "repo").mkdir()
    promoter = TargetPromoter(load_target_registry(registry_path))
    baseline = promoter.capture()
    requested = _candidate(baseline, **{"markdown:repo/new/AGENTS.md": "# New"})
    human_edit = json.dumps({**original, "note": "human edit"})
    registry_path.write_text(human_edit, encoding="utf-8")

    with pytest.raises(PromotionApplyError, match="target registry changed"):
        _promote(promoter, baseline, requested)

    assert registry_path.read_text(encoding="utf-8") == human_edit
    assert not (tmp_path / "repo/new/AGENTS.md").exists()
    assert not (tmp_path / "repo/new").exists()


def test_failed_recursive_create_restores_registration_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = tmp_path / "targets.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "targets": [
                    {
                        "kind": "markdown_root",
                        "id": "repo",
                        "root": "repo",
                        "files": [],
                        "allow_create": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "repo").mkdir()
    promoter = TargetPromoter(load_target_registry(registry_path))
    baseline = promoter.capture()
    requested = _candidate(baseline, **{"markdown:repo/new/AGENTS.md": "# New"})
    real_publish = promoter._publish
    attempts = 0

    def fail_once(item: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("link failed")
        real_publish(item)

    monkeypatch.setattr(promoter, "_publish", fail_once)

    first = _promote(promoter, baseline, requested)
    second = _promote(promoter, baseline, requested)

    assert first.status == "not_applied"
    assert second.status == "applied"
    assert (tmp_path / "repo/new/AGENTS.md").read_text(encoding="utf-8") == "# New"


def test_concurrent_promotions_are_serialized_before_drift_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "AGENTS.md"
    target.write_text("old", encoding="utf-8")
    promoter = TargetPromoter(load_target_registry(_registry(tmp_path)))
    baseline = promoter.capture()
    first_candidate = _candidate(baseline, **{"file:agents": "first"})
    second_candidate = _candidate(baseline, **{"file:agents": "second"})
    first_entered_publish = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    real_publish = promoter._publish

    def controlled_publish(item: object) -> None:
        if item.action.after.content == "first":
            first_entered_publish.set()
            assert release_first.wait(timeout=5)
        real_publish(item)

    def promote_second():
        second_started.set()
        return _promote(promoter, baseline, second_candidate)

    monkeypatch.setattr(promoter, "_publish", controlled_publish)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_promote, promoter, baseline, first_candidate)
        assert first_entered_publish.wait(timeout=5)
        second = executor.submit(promote_second)
        assert second_started.wait(timeout=5)
        release_first.set()
        assert first.result(timeout=5).status == "applied"
        with pytest.raises(StaleBaseError):
            second.result(timeout=5)

    assert target.read_text(encoding="utf-8") == "first"


def test_preflight_source_drift_blocks_every_write(tmp_path: Path) -> None:
    agents = tmp_path / "AGENTS.md"
    soul = tmp_path / "SOUL.md"
    agents.write_text("old agents", encoding="utf-8")
    soul.write_text("old soul", encoding="utf-8")
    promoter = TargetPromoter(load_target_registry(_registry(tmp_path)))
    baseline = promoter.capture()
    requested = _candidate(
        baseline,
        **{"file:agents": "new agents", "file:soul": "new soul"},
    )
    soul.write_text("human edit", encoding="utf-8")

    with pytest.raises(StaleBaseError) as caught:
        _promote(promoter, baseline, requested)

    assert caught.value.changed_locators == ("file:soul",)
    assert agents.read_text(encoding="utf-8") == "old agents"


def test_late_failure_keeps_completed_file_and_reports_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "AGENTS.md").write_text("old agents", encoding="utf-8")
    (tmp_path / "SOUL.md").write_text("old soul", encoding="utf-8")
    promoter = TargetPromoter(load_target_registry(_registry(tmp_path)))
    baseline = promoter.capture()
    requested = _candidate(
        baseline,
        **{"file:agents": "new agents", "file:soul": "new soul"},
    )
    real_publish = promoter._publish

    def fail_soul(item: object) -> None:
        if item.destination.name == "SOUL.md":
            raise OSError("disk full")
        real_publish(item)

    monkeypatch.setattr(promoter, "_publish", fail_soul)

    receipt = _promote(promoter, baseline, requested)

    assert receipt.status == "partial"
    assert receipt.applied_locators == ("file:agents",)
    assert receipt.not_applied_locators == ("file:soul",)
    assert (tmp_path / "AGENTS.md").read_text(encoding="utf-8") == "new agents"
    assert (tmp_path / "SOUL.md").read_text(encoding="utf-8") == "old soul"


def test_immediate_recheck_blocks_only_drifted_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agents = tmp_path / "AGENTS.md"
    soul = tmp_path / "SOUL.md"
    agents.write_text("old agents", encoding="utf-8")
    soul.write_text("old soul", encoding="utf-8")
    promoter = TargetPromoter(load_target_registry(_registry(tmp_path)))
    baseline = promoter.capture()
    requested = _candidate(
        baseline,
        **{"file:agents": "new agents", "file:soul": "new soul"},
    )
    real_publish = promoter._publish

    def drift_after_first(item: object) -> None:
        real_publish(item)
        if item.destination.name == "AGENTS.md":
            soul.write_text("human edit", encoding="utf-8")

    monkeypatch.setattr(promoter, "_publish", drift_after_first)

    receipt = _promote(promoter, baseline, requested)

    assert receipt.status == "partial"
    assert receipt.not_applied_locators == ("file:soul",)
    assert receipt.outcomes[1].reason == "source_drift"
    assert soul.read_text(encoding="utf-8") == "human edit"


def test_failed_atomic_replace_never_exposes_partial_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agents = tmp_path / "AGENTS.md"
    agents.write_text("old", encoding="utf-8")
    promoter = TargetPromoter(load_target_registry(_registry(tmp_path)))
    baseline = promoter.capture()
    requested = _candidate(baseline, **{"file:agents": "new"})

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("weave_agent_signals.runs.promotion.os.replace", fail_replace)
    receipt = _promote(promoter, baseline, requested)

    assert receipt.status == "not_applied"
    assert agents.read_text(encoding="utf-8") == "old"


def test_staging_failure_happens_before_any_file_is_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agents = tmp_path / "AGENTS.md"
    soul = tmp_path / "SOUL.md"
    agents.write_text("old agents", encoding="utf-8")
    soul.write_text("old soul", encoding="utf-8")
    promoter = TargetPromoter(load_target_registry(_registry(tmp_path)))
    baseline = promoter.capture()
    requested = _candidate(
        baseline,
        **{"file:agents": "new agents", "file:soul": "new soul"},
    )
    real_stage = promoter._stage

    def fail_soul(path: Path, content: str, action: str) -> Path:
        if path.name == "SOUL.md":
            raise OSError("cannot stage")
        return real_stage(path, content, action)

    monkeypatch.setattr(promoter, "_stage", fail_soul)

    with pytest.raises(Exception, match="cannot stage"):
        _promote(promoter, baseline, requested)

    assert agents.read_text(encoding="utf-8") == "old agents"
    assert soul.read_text(encoding="utf-8") == "old soul"


def test_receipt_round_trips_exact_per_file_outcomes(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("old", encoding="utf-8")
    promoter = TargetPromoter(load_target_registry(_registry(tmp_path)))
    baseline = promoter.capture()
    requested = _candidate(baseline, **{"file:agents": "new"})

    receipt = _promote(promoter, baseline, requested)

    assert PromotionReceipt.from_dict(receipt.to_dict()) == receipt


def test_receipt_rejects_an_applied_file_after_a_not_applied_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "AGENTS.md").write_text("old agents", encoding="utf-8")
    (tmp_path / "SOUL.md").write_text("old soul", encoding="utf-8")
    promoter = TargetPromoter(load_target_registry(_registry(tmp_path)))
    baseline = promoter.capture()
    requested = _candidate(
        baseline,
        **{"file:agents": "new agents", "file:soul": "new soul"},
    )
    monkeypatch.setattr(
        promoter,
        "_publish",
        lambda _item: (_ for _ in ()).throw(OSError("first failed")),
    )
    serialized = _promote(promoter, baseline, requested).to_dict()
    serialized["outcomes"][1] = {
        "locator": "file:soul",
        "action": "update",
        "status": "applied",
        "reason": None,
        "message": None,
    }

    with pytest.raises(Exception, match="after a not-applied"):
        PromotionReceipt.from_dict(serialized)

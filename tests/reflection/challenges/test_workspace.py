from pathlib import Path

import pytest

from weave_agent_signals.runs.bundles import (
    BundleSnapshot,
    ScopeDescriptor,
    TargetSnapshot,
)
from weave_agent_signals.runs.challenges.workspace import (
    WorkspaceFile,
    WorkspaceSnapshot,
    capture_workspace,
    describe_workspace_changes,
    materialize_arms,
)


def _bundles() -> tuple[BundleSnapshot, BundleSnapshot]:
    scope = ScopeDescriptor(kind="registry", target_id="targets-v1")
    baseline = BundleSnapshot(
        (
            TargetSnapshot(
                kind="file",
                locator="markdown:repo/AGENTS.md",
                exists=True,
                content="baseline agents\n",
            ),
            TargetSnapshot(
                kind="file",
                locator="markdown:repo/docs/GUIDE.md",
                exists=True,
                content="unchanged guide\n",
            ),
            TargetSnapshot(
                kind="file",
                locator="markdown:repo/new/POLICY.md",
                exists=False,
                content=None,
            ),
        ),
        scope,
    )
    candidate = BundleSnapshot(
        (
            TargetSnapshot(
                kind="file",
                locator="markdown:repo/AGENTS.md",
                exists=True,
                content="candidate agents\n",
            ),
            TargetSnapshot(
                kind="file",
                locator="markdown:repo/docs/GUIDE.md",
                exists=True,
                content="unchanged guide\n",
            ),
            TargetSnapshot(
                kind="file",
                locator="markdown:repo/new/POLICY.md",
                exists=True,
                content="new policy\n",
            ),
        ),
        scope,
    )
    return baseline, candidate


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    (root / "src").mkdir()
    (root / ".git").mkdir()
    (root / "node_modules" / "pkg").mkdir(parents=True)
    (root / "AGENTS.md").write_text("baseline agents\n", encoding="utf-8")
    (root / "docs" / "GUIDE.md").write_text("unchanged guide\n", encoding="utf-8")
    (root / "src" / "app.py").write_text("print('same')\n", encoding="utf-8")
    (root / ".git" / "config").write_text("secret-ish metadata", encoding="utf-8")
    (root / "node_modules" / "pkg" / "index.js").write_text("ignored", encoding="utf-8")
    return root


def test_workspace_capture_is_deterministic_and_excludes_runtime_bulk(tmp_path: Path) -> None:
    root = _workspace(tmp_path)

    first = capture_workspace(root)
    second = capture_workspace(root)

    assert first == second
    assert first.paths == ("AGENTS.md", "docs/GUIDE.md", "src/app.py")
    assert first.digest.startswith("sha256:")
    assert first.archive == second.archive


def test_workspace_capture_rejects_symlinks(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    (root / "linked.md").symlink_to(root / "AGENTS.md")

    with pytest.raises(ValueError, match="symlink"):
        capture_workspace(root)


def test_materialized_arms_differ_only_by_declared_candidate_actions(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    baseline, candidate = _bundles()
    paths = {
        "markdown:repo/AGENTS.md": root / "AGENTS.md",
        "markdown:repo/docs/GUIDE.md": root / "docs" / "GUIDE.md",
        "markdown:repo/new/POLICY.md": root / "new" / "POLICY.md",
    }

    arms = materialize_arms(
        capture_workspace(root),
        baseline=baseline,
        candidate=candidate,
        workspace_root=root,
        resolve_locator=paths.__getitem__,
    )

    assert arms.authoring.paths == ("src/app.py",)
    assert arms.baseline.read_text("AGENTS.md") == "baseline agents\n"
    assert arms.candidate.read_text("AGENTS.md") == "candidate agents\n"
    assert arms.candidate.read_text("new/POLICY.md") == "new policy\n"
    assert arms.baseline.read_text("docs/GUIDE.md") == "unchanged guide\n"
    assert arms.candidate.read_text("docs/GUIDE.md") == "unchanged guide\n"
    assert arms.baseline.read_text("src/app.py") == arms.candidate.read_text("src/app.py")


def test_workspace_changes_include_bounded_semantic_diffs() -> None:
    before = WorkspaceSnapshot(
        (
            WorkspaceFile("modified.txt", b"before\n"),
            WorkspaceFile("deleted.txt", b"gone\n"),
        )
    )
    after = WorkspaceSnapshot(
        (
            WorkspaceFile("modified.txt", b"after\n"),
            WorkspaceFile("added.txt", b"new\n"),
        )
    )

    changes = describe_workspace_changes(before, after)

    assert [item.path for item in changes] == ["added.txt", "deleted.txt", "modified.txt"]
    assert [item.action for item in changes] == ["added", "deleted", "modified"]
    assert "+new" in changes[0].diff
    assert "-gone" in changes[1].diff
    assert "-before" in changes[2].diff and "+after" in changes[2].diff


def test_materialization_rejects_workspace_that_no_longer_matches_a(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    baseline, candidate = _bundles()
    (root / "AGENTS.md").write_text("drifted\n", encoding="utf-8")
    paths = {
        "markdown:repo/AGENTS.md": root / "AGENTS.md",
        "markdown:repo/docs/GUIDE.md": root / "docs" / "GUIDE.md",
        "markdown:repo/new/POLICY.md": root / "new" / "POLICY.md",
    }

    with pytest.raises(ValueError, match="does not match baseline A"):
        materialize_arms(
            capture_workspace(root),
            baseline=baseline,
            candidate=candidate,
            workspace_root=root,
            resolve_locator=paths.__getitem__,
        )


def test_materialization_injects_a_and_b_after_common_preparation(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    baseline, candidate = _bundles()
    prepared = WorkspaceSnapshot(
        (
            WorkspaceFile("src/app.py", b"print('prepared')\n"),
            WorkspaceFile("task/README.md", b"public fixture\n"),
        )
    )
    paths = {
        "markdown:repo/AGENTS.md": root / "AGENTS.md",
        "markdown:repo/docs/GUIDE.md": root / "docs" / "GUIDE.md",
        "markdown:repo/new/POLICY.md": root / "new" / "POLICY.md",
    }

    arms = materialize_arms(
        prepared,
        baseline=baseline,
        candidate=candidate,
        workspace_root=root,
        resolve_locator=paths.__getitem__,
        verify_baseline=False,
    )

    assert arms.authoring.paths == ("src/app.py", "task/README.md")
    assert arms.baseline.read_text("AGENTS.md") == "baseline agents\n"
    assert arms.baseline.read_text("docs/GUIDE.md") == "unchanged guide\n"
    assert arms.candidate.read_text("AGENTS.md") == "candidate agents\n"
    assert arms.candidate.read_text("new/POLICY.md") == "new policy\n"
    assert arms.baseline.read_text("task/README.md") == arms.candidate.read_text("task/README.md")


def test_materialization_mirrors_home_instruction_targets_into_guest_home(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    root = home / "repo"
    codex_agents = home / ".codex" / "AGENTS.md"
    root.mkdir(parents=True)
    codex_agents.parent.mkdir()
    (root / "README.md").write_text("workspace\n", encoding="utf-8")
    codex_agents.write_text("baseline global\n", encoding="utf-8")
    baseline = BundleSnapshot(
        (
            TargetSnapshot(
                kind="file",
                locator="markdown:global-codex/AGENTS.md",
                exists=True,
                content="baseline global\n",
            ),
        )
    )
    candidate = BundleSnapshot(
        (
            TargetSnapshot(
                kind="file",
                locator="markdown:global-codex/AGENTS.md",
                exists=True,
                content="candidate global\n",
            ),
        )
    )

    arms = materialize_arms(
        capture_workspace(root),
        baseline=baseline,
        candidate=candidate,
        workspace_root=root,
        resolve_locator=lambda _locator: codex_agents,
        host_home=home,
        guest_home="/root",
    )

    assert arms.baseline.paths == arms.candidate.paths == ("README.md",)
    assert [(item.path, item.content) for item in arms.baseline_runtime_files] == [
        ("/root/.codex/AGENTS.md", b"baseline global\n")
    ]
    assert [(item.path, item.content) for item in arms.candidate_runtime_files] == [
        ("/root/.codex/AGENTS.md", b"candidate global\n")
    ]


def test_materialization_rejects_managed_target_outside_workspace_and_home(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path)
    baseline, candidate = _bundles()

    with pytest.raises(ValueError, match="outside the sandbox workspace and host home"):
        materialize_arms(
            capture_workspace(root),
            baseline=baseline,
            candidate=candidate,
            workspace_root=root,
            resolve_locator=lambda _locator: Path("/opt/outside.md"),
            host_home=tmp_path,
        )

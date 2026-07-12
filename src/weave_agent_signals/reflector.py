"""GEPA reflector: proposes CLAUDE.md/skills/memory edits from evaluation data.

The reflector is the weak-RSI loop. It:
1. Reads current artifacts (CLAUDE.md, commands, skills, memory)
2. Formats scored feedback + judge rationales as evaluation data
3. Runs GEPA optimize_anything to propose edits
4. Outputs diffs for human review (review gate)
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("weave_agent_signals.reflector")

ARTIFACT_GLOBS = [
    "CLAUDE.md",
    ".claude/commands/*.md",
    ".claude/skills/*.md",
]


@dataclass
class Artifact:
    name: str
    path: str
    content: str


@dataclass
class Proposal:
    artifacts: list[Artifact]
    rationale: str
    score_delta: float
    metadata: dict[str, Any] = field(default_factory=dict)


def extract_artifacts(project_root: str) -> list[Artifact]:
    """Read current CLAUDE.md, commands, and skills from the project."""
    root = Path(project_root)
    artifacts = []

    for glob_pattern in ARTIFACT_GLOBS:
        for path in root.glob(glob_pattern):
            if path.is_file():
                try:
                    content = path.read_text(encoding="utf-8")
                    rel = str(path.relative_to(root))
                    artifacts.append(
                        Artifact(
                            name=path.name,
                            path=rel,
                            content=content,
                        )
                    )
                except (OSError, UnicodeDecodeError) as e:
                    log.warning("Could not read %s: %s", path, e)

    return artifacts


def format_evaluation_batch(feedback: list[dict]) -> list[dict[str, Any]]:
    """Convert scored feedback into GEPA evaluation data entries.

    Each entry has:
    - score: float (the rating)
    - feedback: str (judge rationale + tags)
    - scorer: str (which scorer produced it)
    """
    entries: list[dict[str, Any]] = []

    for fb in feedback:
        ftype = fb.get("feedback_type", "")
        payload = fb.get("payload", {})
        rating = payload.get("rating")
        if rating is None:
            continue

        details = payload.get("details", {})
        rationale = details.get("rationale", payload.get("reason", ""))
        tags = payload.get("tags", [])

        feedback_text = rationale
        if tags:
            feedback_text += f" [tags: {', '.join(tags)}]"

        entries.append(
            {
                "score": float(rating),
                "feedback": feedback_text,
                "scorer": ftype.replace("weave_agent_signals.", ""),
            }
        )

    return entries


def _artifacts_to_candidate(artifacts: list[Artifact]) -> dict[str, str]:
    """Convert artifacts to a GEPA candidate dict."""
    return {a.path: a.content for a in artifacts}


def _candidate_to_artifacts(candidate: dict[str, str]) -> list[Artifact]:
    """Convert a GEPA candidate dict back to artifacts."""
    return [
        Artifact(name=Path(path).name, path=path, content=content)
        for path, content in candidate.items()
    ]


def render_proposal_diff(
    originals: list[Artifact],
    proposal: Proposal,
) -> str:
    """Render a unified diff between original and proposed artifacts."""
    original_map = {a.path: a.content for a in originals}
    proposed_map = {a.path: a.content for a in proposal.artifacts}

    all_paths = sorted(set(original_map) | set(proposed_map))
    diffs = []

    for path in all_paths:
        old = original_map.get(path, "")
        new = proposed_map.get(path, "")
        if old == new:
            continue
        diff_lines = list(
            difflib.unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
        if diff_lines:
            diffs.append("".join(diff_lines))

    if not diffs:
        return "No changes proposed."

    header = f"Rationale: {proposal.rationale}\n\n"
    return header + "\n".join(diffs)


def build_gepa_objective(coaching_text: str) -> str:
    """Build the GEPA optimization objective from coaching data."""
    return (
        "Improve the CLAUDE.md instructions and agent configuration files "
        "to increase the agent's evaluation scores. The agent is a Claude Code "
        "coding assistant whose performance is measured by deterministic outcome "
        "scorers (test pass rate, build success, lint compliance) and LLM judge "
        "rubrics (verification discipline, error recovery, tool choice, task completion).\n\n"
        "Current evaluation summary:\n"
        f"{coaching_text}\n\n"
        "Propose specific, actionable edits. Focus on the lowest-scoring dimensions. "
        "Changes should be concrete instructions, not vague advice."
    )


def build_gepa_background() -> str:
    """Build background context for the GEPA reflector."""
    return (
        "The artifacts being optimized are configuration files for Claude Code, "
        "an AI coding assistant. CLAUDE.md files contain project-level instructions "
        "that the agent follows during every session. Commands (.claude/commands/*.md) "
        "are slash commands the user can invoke. Skills (.claude/skills/*.md) are "
        "triggered by the agent when relevant.\n\n"
        "Key constraints:\n"
        "- Instructions must be clear, specific, and actionable\n"
        "- Avoid redundancy with built-in agent behavior\n"
        "- Focus on process quality: verification, error handling, tool usage\n"
        "- Changes are gated by human review before deployment\n"
        "- Each change flips the config_version hash for A/B measurement"
    )


ARTIFACT_JUDGE_SYSTEM = (
    "You predict how well a proposed CLAUDE.md instruction file will improve a "
    "coding agent's evaluation scores. You are given the agent's current failure "
    "patterns and a candidate instruction file. Score how well the candidate "
    "addresses those failures, from 0.0 (no improvement) to 1.0 (fully resolves "
    "them), and briefly justify it."
)


def _make_artifact_evaluator(
    judge_client: Any,
    judge_model: str | None,
    coaching_text: str,
    baseline: float,
):
    """Build a GEPA evaluator that scores each *proposed* candidate.

    Since a CLAUDE.md cannot be executed offline, ranking uses an LLM judge that
    predicts how well the candidate addresses the observed failure patterns.
    Without a judge client the score falls back to the baseline (constant), which
    disables ranking — GEPA can still generate a single proposal but cannot
    choose between competing ones.
    """

    def evaluator(proposed: dict[str, str]) -> tuple[float, dict]:
        artifact_text = "\n\n".join(f"## {path}\n{content}" for path, content in proposed.items())
        if judge_client is None:
            return baseline, {
                "ranking_enabled": False,
                "baseline": baseline,
                "predicted_quality": baseline,
            }
        messages = [
            {"role": "system", "content": ARTIFACT_JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"## Agent failure patterns\n\n{coaching_text}\n\n"
                    f"## Proposed CLAUDE.md\n\n{artifact_text}\n\n"
                    "## Instructions\n\nRespond with JSON:\n"
                    '{"score": <float 0.0-1.0>, "rationale": "<1-2 sentences>"}'
                ),
            },
        ]
        try:
            parsed, _ = judge_client.chat_json(
                model=judge_model,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )
            quality = max(0.0, min(1.0, float(parsed.get("score", baseline))))
            rationale = parsed.get("rationale", "")
        except Exception as e:  # judge unavailable → fall back, keep GEPA running
            log.warning("Artifact evaluator judge failed: %s", e)
            quality, rationale = baseline, "judge unavailable"
        return quality, {
            "ranking_enabled": True,
            "baseline": baseline,
            "predicted_quality": quality,
            "judge_rationale": rationale,
        }

    return evaluator


def run_reflection(
    project_root: str,
    feedback: list[dict],
    coaching_text: str,
    judge_client: Any = None,
    judge_model: str | None = None,
    model: str = "gpt-4o",
    max_iterations: int = 3,
) -> Proposal | None:
    """Run GEPA optimize_anything on the agent's artifacts.

    ``model`` drives GEPA's reflection (candidate generation); ``judge_client``
    (+ ``judge_model``) scores candidates so GEPA can rank them. Returns a
    Proposal with suggested edits, or None if nothing to optimize.
    """
    from gepa.optimize_anything import (
        EngineConfig,
        GEPAConfig,
        ReflectionConfig,
        optimize_anything,
    )

    artifacts = extract_artifacts(project_root)
    if not artifacts:
        log.warning("No artifacts found in %s", project_root)
        return None

    candidate = _artifacts_to_candidate(artifacts)
    eval_batch = format_evaluation_batch(feedback)

    if not eval_batch:
        log.warning("No evaluation data to optimize against")
        return None

    mean_score = sum(e["score"] for e in eval_batch) / len(eval_batch)
    evaluator = _make_artifact_evaluator(judge_client, judge_model, coaching_text, mean_score)

    objective = build_gepa_objective(coaching_text)
    background = build_gepa_background()

    config = GEPAConfig(
        engine=EngineConfig(
            max_candidate_proposals=max_iterations,
            display_progress_bar=False,
        ),
        reflection=ReflectionConfig(
            reflection_lm=model,
        ),
    )

    result = optimize_anything(
        seed_candidate=candidate,
        evaluator=evaluator,
        objective=objective,
        background=background,
        config=config,
    )

    best = result.best_candidate
    if best is None:
        return None

    if isinstance(best, str):
        proposed_artifacts = [
            Artifact(
                name="CLAUDE.md",
                path="CLAUDE.md",
                content=best,
            )
        ]
    else:
        proposed_artifacts = _candidate_to_artifacts(best)

    return Proposal(
        artifacts=proposed_artifacts,
        rationale=getattr(result, "rationale", "GEPA optimization"),
        score_delta=getattr(result, "best_score", 0.0) - mean_score,
        metadata={
            "iterations": getattr(result, "num_iterations", 0),
            "model": model,
            "ranking_enabled": judge_client is not None,
        },
    )

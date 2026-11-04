"""Session Score IDs Monopole wrote, for CES deployments without list-trajectory-rows.

Workaround: CES returns Scores on Trajectories only through that route, which ships with
the Trajectory-row stack (wandb/core#60962, #60963, #60954). Until it deploys,
`CesClient.session_scores` falls back to this index. `write_scores` appends the IDs of
the Scores it writes on Trajectories, and the newest ID per Scorer is read back with
get-score. The index sees only Scores written from this machine. Delete this module and
the fallback once every CES that Monopole reads serves the route.
"""

from __future__ import annotations

import json
import os

DEFAULT_PATH = "~/.weave-agent-signals/ces-session-scores.jsonl"


class ScoreIndex:
    def __init__(self, path: str | None = None):
        self.path = os.path.expanduser(
            path or os.environ.get("WEAVE_AGENT_SIGNALS_CES_SCORE_INDEX", DEFAULT_PATH)
        )

    def record(self, trajectory_id: str, scores: list[tuple[str, str]]) -> None:
        """Append (scorer_id, score_id) pairs written on one Trajectory, in write order."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            for scorer_id, score_id in scores:
                entry = {
                    "trajectory_id": trajectory_id,
                    "scorer_id": scorer_id,
                    "score_id": score_id,
                }
                fh.write(json.dumps(entry) + "\n")

    def newest(self, trajectory_ids=None) -> dict[str, dict[str, str]]:
        """Trajectory ID → {scorer_id: the newest score_id written for it}. None means every one."""
        wanted, found = None if trajectory_ids is None else set(trajectory_ids), {}
        try:
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    if wanted is None or entry.get("trajectory_id") in wanted:
                        found.setdefault(entry["trajectory_id"], {})[entry["scorer_id"]] = entry[
                            "score_id"
                        ]
        except OSError:
            pass
        return found

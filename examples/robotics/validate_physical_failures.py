"""Deliberately poor interaction parameters, never directly changed object state."""

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import ddm_mcts.robotics.physical_reasoning as reasoning
from ddm_mcts.robotics.interactions import push_candidates
from ddm_mcts.robotics.physical_goals import parse_physical_goals
from ddm_mcts.robotics.pose_control import PoseTarget
from ddm_mcts.robotics.reasoning_scene import panda_reasoning_scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    folder = Path("robotics_runs") / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_v5_failures_" + uuid4().hex[:12])
    folder.mkdir()
    original = reasoning.support_surface

    def poor_alignment(world, owner):
        surface = original(world, owner)
        if owner == "cylinder":
            p = list(surface.pose.position)
            p[0] += 0.023
            surface = replace(surface, pose=PoseTarget(tuple(p), surface.pose.quaternion))
        return surface

    world = panda_reasoning_scene(args.model)
    with patch.object(reasoning, "support_surface", poor_alignment):
        stack = reasoning.CompositionalPhysicalAgent(world).run_goal_instruction(
            "Put the cube on the cylinder, wait for 2 seconds, then topple the tower.", folder
        )
    attempts = []

    def ineffective(world, state, subject, **kwargs):
        attempts.append(1)
        return push_candidates(world, "cylinder")[2], {"speculative_substeps": 0, "controlled_fixture": "only low short pushes available"}

    world = panda_reasoning_scene(args.model)
    with patch.object(reasoning, "search_push", ineffective):
        push = reasoning.CompositionalPhysicalAgent(world).run_goal_instruction("Push the cylinder over, then wait for 2 seconds.", folder)
    report = {}
    for name, result in (("unstable_stack", stack), ("ineffective_push", push)):
        report[name] = {
            "success": result.success,
            "log": str(result.folder),
            "failure": result.trace.get("failure"),
            "later_wait_executed": any("wait_elapsed" in goal for goal in result.trace["goal_results"]),
        }
    report["ineffective_push"]["attempts"] = len(attempts)
    try:
        parse_physical_goals("Throw the sphere.")
    except ValueError as exc:
        report["unsupported"] = {"safely_rejected": True, "reason": str(exc)}
    (folder / "validation.json").write_text(json.dumps(report, indent=2, default=str))
    print(folder)
    assert not stack.success and not push.success and len(attempts) == 2
    assert all(not report[name]["later_wait_executed"] for name in ("unstable_stack", "ineffective_push"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

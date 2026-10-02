"""Opt-in real Qwen physical acceptance; three goals, one loaded model."""

import argparse
import json
from math import dist

from ddm_mcts.robotics.approach import ApproachReachTask
from ddm_mcts.robotics.camera import MujocoCameraObservationProvider
from ddm_mcts.robotics.core import RoboticsPlanner
from ddm_mcts.robotics.observation import SemanticGoal
from ddm_mcts.robotics.physical_agent import PhysicalAgent
from ddm_mcts.robotics.semantic_scene import OBJECT_RGB, SEMANTIC_PLANE_Z, panda_semantic_reach
from ddm_mcts.robotics.vlm_backend import Qwen3VLBackend
from ddm_mcts.robotics.vlm_perception import VLMSemanticPerception
from ddm_mcts.search.mcts import MCTSConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="robotics_runs")
    args = parser.parse_args()
    world = panda_semantic_reach(args.model, use_tcp=True)
    backend = Qwen3VLBackend()
    perception = VLMSemanticPerception(backend, SEMANTIC_PLANE_Z, OBJECT_RGB, debug=True)
    with PhysicalAgent(
        world,
        MujocoCameraObservationProvider(world, width=640, height=480),
        perception,
        ApproachReachTask(SemanticGoal("cylinder")),
        RoboticsPlanner(world, config=MCTSConfig(60, seed=0)),
        diagnostics=world.target_entities,
    ) as agent:
        records = []
        for goal in ("cylinder", "cube", "sphere"):
            agent.reset(ApproachReachTask(SemanticGoal(goal)))
            folder = agent.run(args.output)
            summary = json.loads((folder / "summary.json").read_text())
            truth = next(e.position for e in world.target_entities() if e.label == goal)
            observations = [json.loads(line) for line in (folder / "observations.jsonl").read_text().splitlines()]
            record = {
                "goal": goal,
                "success": summary["success"],
                "actions": summary["decisions"],
                "tcp_to_object_center_m": dist(world.get_state().position, truth),
                "approach": summary["execution_diagnostics"],
                "max_localization_error_m": max(row["perception_errors_m"][goal] for row in observations),
                "planning_seconds": summary["total_planning_seconds"],
                "observations": len(observations),
                "vlm": backend.diagnostics(),
                "run": str(folder),
            }
            (folder / "vlm_acceptance.json").write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
            print(json.dumps(record), flush=True)
    return 0 if all(r["success"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())

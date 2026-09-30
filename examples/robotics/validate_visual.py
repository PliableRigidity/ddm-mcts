"""Small Phase 3 acceptance audit; six real closed-loop runs, no benchmark harness."""

import argparse
import json
from math import dist
from pathlib import Path

from ddm_mcts.robotics.camera import MujocoCameraObservationProvider
from ddm_mcts.robotics.color_perception import ColorPlanePerception
from ddm_mcts.robotics.core import RoboticsPlanner
from ddm_mcts.robotics.observation import SemanticGoal
from ddm_mcts.robotics.physical_agent import PhysicalAgent
from ddm_mcts.robotics.representation import SemanticReachTask
from ddm_mcts.robotics.visual_scene import TARGET_PLANE_Z, VisualTarget, panda_visual_reach
from ddm_mcts.search.mcts import MCTSConfig

PLACEMENTS = (
    ((0.6045, 0.08), (0.5045, -0.08)),
    ((0.5945, -0.06), (0.5145, -0.08)),
    ((0.6245, 0.04), (0.5045, -0.10)),
)


def validate(model, output="robotics_runs", placements=PLACEMENTS):
    records = []
    for index, positions in enumerate(placements):
        targets = tuple(
            VisualTarget(label, (*position, TARGET_PLANE_Z), rgba)
            for label, position, rgba in zip(("red", "blue"), positions, ((1, 0, 0, 1), (0, 0, 1, 1)), strict=True)
        )
        world = panda_visual_reach(model, targets=targets)
        with PhysicalAgent(
            world,
            MujocoCameraObservationProvider(world),
            ColorPlanePerception(TARGET_PLANE_Z),
            SemanticReachTask(SemanticGoal("red")),
            RoboticsPlanner(world, config=MCTSConfig(60, seed=0), horizon=3),
            diagnostics=world.target_entities,
        ) as agent:
            # Only the goal changes: same world, camera, perception, policy,
            # MCTS configuration and controller. Reset restores episode state.
            for goal in ("red", "blue"):
                agent.reset(SemanticReachTask(SemanticGoal(goal)))
                folder = agent.run(output)
                summary = json.loads((folder / "summary.json").read_text())
                observations = [json.loads(line) for line in (folder / "observations.jsonl").read_text().splitlines()]
                truth = next(entity.position for entity in world.target_entities() if entity.label == goal)
                estimate = next(
                    entity["position"] for entity in summary["final_perception"]["perceived_state"]["entities"] if entity["label"] == goal
                )
                record = {
                    "placement": index,
                    "goal": goal,
                    "ground_truth": truth,
                    "perceived_position": estimate,
                    "max_perception_error_m": max(row["perception_errors_m"][goal] for row in observations),
                    "true_final_reach_error_m": dist(world.get_state().position, truth),
                    "success": summary["success"],
                    "actions": summary["decisions"],
                    "observations": len(observations),
                    "planning_seconds": summary["total_planning_seconds"],
                    "run": str(folder),
                }
                (folder / "acceptance.json").write_text(json.dumps(record, indent=2) + "\n")
                records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, default=Path("robotics_runs"))
    args = parser.parse_args()
    records = validate(args.model, args.output)
    print(json.dumps(records, indent=2))
    return 0 if all(record["success"] and record["true_final_reach_error_m"] < 0.015 for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())

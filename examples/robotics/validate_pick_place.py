"""Real contact-physics acceptance; viewers close automatically after each plan."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ddm_mcts.robotics.manipulation_scene import panda_pickup_scene
from ddm_mcts.robotics.pick_place import PickPlaceAgent, rearrangement_plan
from ddm_mcts.robotics.placement import NextTo, PlacementTarget


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--sequential-trials", type=int, default=2)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--viewer-speed", type=float, default=4)
    parser.add_argument("--output", default="robotics_runs")
    args = parser.parse_args(argv)
    if min(args.trials, args.sequential_trials) < 1:
        parser.error("positive trial counts required")
    runs = []
    for mode, count in (("cube", args.trials), ("cylinder", args.trials), ("sequential", args.sequential_trials)):
        for trial in range(count):
            world = panda_pickup_scene(args.model)
            agent = PickPlaceAgent(world)

            def execute(visual=None, mode=mode, agent=agent):
                if mode == "sequential":
                    return agent.run_manipulation_plan(rearrangement_plan(), args.output, visualization=visual)
                destination = PlacementTarget((0.60, -0.10, 0.37)) if mode == "cube" else NextTo("cube")
                return agent.pick_and_place(mode, destination, args.output, visualization=visual)

            if args.viewer:
                from ddm_mcts.robotics.manipulation_visual import PickupInspection

                with PickupInspection(world, args.viewer_speed) as visual:
                    result = execute(visual)
            else:
                result = execute()
            metrics = []
            for row in result.trace["operations"]:
                pickup = row.get("pickup_attempts", [{}])[-1].get("trace", {})
                metrics.append(
                    {
                        "target": row["target"],
                        "success": row["success"],
                        "pregrasp_errors": pickup.get("pregrasp_errors"),
                        "grasp_verification": pickup.get("grasp_verification"),
                        "lift_verification": pickup.get("lift_verification"),
                        "transport": row["transport"],
                        "desired_pose": row.get("placement_target"),
                        "actual_pose": row.get("final_object", {}).get("pose"),
                        "placement": row.get("placement_verification"),
                        "relation": row.get("relation"),
                        "recovery": row["recovery"],
                        "physics_substeps": row["physics_substeps"],
                        "intended_contacts": row["intended_contacts"],
                        "forbidden_contacts": row["forbidden_contacts"],
                        "maximum_penetration_m": row["maximum_penetration_m"],
                        "duration_seconds": row["duration_seconds"],
                        "failure": row.get("failure"),
                    }
                )
            runs.append(
                {
                    "mode": mode,
                    "trial": trial + 1,
                    "success": result.success,
                    "folder": str(result.folder),
                    "state_continuity": result.trace["state_continuity"],
                    "reset_count": result.trace["reset_count"],
                    "operations": metrics,
                }
            )
    root = Path(args.output).resolve()
    folder = root / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_placement_validation_" + uuid4().hex[:12])
    folder.mkdir(exist_ok=False)
    (folder / "validation.json").write_text(json.dumps(runs, indent=2))
    print(json.dumps(runs, indent=2))
    print("Validation:", folder, flush=True)
    return 0 if all(r["success"] for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())

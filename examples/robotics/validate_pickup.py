"""Small real-physics acceptance utility; GUI closes after each bounded trial."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ddm_mcts.robotics.manipulation import ManipulationAgent, PickupTask
from ddm_mcts.robotics.manipulation_scene import panda_pickup_scene


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--viewer-speed", type=float, default=4)
    parser.add_argument("--output", default="robotics_runs")
    args = parser.parse_args(argv)
    if args.trials < 1:
        parser.error("positive trial count required")
    runs = []
    for label in ("cube", "cylinder"):
        for trial in range(args.trials):
            world = panda_pickup_scene(args.model)
            agent = ManipulationAgent(world)
            if args.viewer:
                from ddm_mcts.robotics.manipulation_visual import PickupInspection

                with PickupInspection(world, args.viewer_speed) as visual:
                    result = agent.run(PickupTask(label), args.output, visualization=visual)
            else:
                result = agent.run(PickupTask(label), args.output)
            t = result.trace
            runs.append(
                {
                    "target": label,
                    "trial": trial + 1,
                    "success": result.success,
                    "folder": str(result.folder),
                    "pregrasp_errors": t.get("pregrasp_errors"),
                    "grasp_pose_errors": t.get("grasp_pose_errors"),
                    "grasp_verification": t.get("grasp_verification"),
                    "lift_verification": t.get("lift_verification"),
                    "finger_width": t["final_gripper"]["actual_width"],
                    "minimum_contact_distance_m": t["minimum_contact_distance_m"],
                    "physics_steps_monitored": t["physics_steps_monitored"],
                    "elapsed_seconds": t["elapsed_seconds"],
                    "reason": t.get("reason"),
                }
            )
    root = Path(args.output).resolve()
    folder = root / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_pickup_validation_" + uuid4().hex[:12])
    folder.mkdir(exist_ok=False)
    (folder / "validation.json").write_text(json.dumps(runs, indent=2))
    print(json.dumps(runs, indent=2))
    print("Validation:", folder)
    return 0 if all(r["success"] for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Integrated language acceptance, retaining all runs and one shared local VLM."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ddm_mcts.robotics.manipulation_agent import PhysicalAIManipulationAgent
from ddm_mcts.robotics.manipulation_observation import ManipulationCamera, ManipulationPerception
from ddm_mcts.robotics.manipulation_scene import panda_pickup_scene
from ddm_mcts.robotics.vlm_backend import Qwen3VLBackend

SINGLE = "Pick up the cylinder and place it next to the cube."
MULTI = "Move the cylinder next to the cube, then move the cube to the target location."


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vlm", action="store_true")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--viewer-speed", type=float, default=8)
    parser.add_argument("--output", default="robotics_runs")
    args = parser.parse_args(argv)
    if args.trials < 1:
        parser.error("positive trial count required")
    root = Path(args.output).resolve() / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_agent_validation_" + uuid4().hex[:12])
    root.mkdir(parents=True, exist_ok=False)
    backend = Qwen3VLBackend() if args.vlm else None
    rows = []
    for instruction, count in ((SINGLE, 1), (MULTI, args.trials)):
        for trial in range(count):
            world = panda_pickup_scene(args.model)  # separate acceptance trials, never between operations
            observation = ManipulationCamera(world) if args.vlm else None
            agent = PhysicalAIManipulationAgent(world, observation=observation, perception=ManipulationPerception(world, backend))
            try:
                if args.viewer:
                    from ddm_mcts.robotics.manipulation_visual import PickupInspection

                    with PickupInspection(world, args.viewer_speed) as viewer:
                        result = agent.run_instruction(instruction, root, visualization=viewer)
                        agent.close()
                else:
                    result = agent.run_instruction(instruction, root)
                rows.append(
                    {
                        "instruction": instruction,
                        "trial": trial + 1,
                        "success": result.success,
                        "folder": str(result.folder),
                        "trace": result.trace,
                    }
                )
            finally:
                agent.close()
            summary = {"runs": rows, "success": all(r["success"] for r in rows), "model": backend.diagnostics() if backend else None}
            (root / "validation.json").write_text(json.dumps(summary, indent=2))
    print(f"VALIDATION: {root}; successes {sum(r['success'] for r in rows)}/{len(rows)}", flush=True)
    return 0 if all(r["success"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

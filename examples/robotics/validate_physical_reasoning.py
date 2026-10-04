"""Independent real MuJoCo trials; every outcome is retained in ignored run logs."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ddm_mcts.robotics.manipulation_observation import ManipulationCamera, ManipulationPerception
from ddm_mcts.robotics.physical_reasoning import CompositionalPhysicalAgent
from ddm_mcts.robotics.reasoning_scene import panda_reasoning_scene

INSTRUCTIONS = {
    "stack": "Put the cube on the cylinder.",
    "push": "Push the cylinder over.",
    "headline": "Pick up the cube and place it on the cylinder. Once that is done, wait for 2 seconds and topple the tower.",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--scenario", choices=INSTRUCTIONS, required=True)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--vlm", action="store_true")
    args = parser.parse_args()
    folder = Path("robotics_runs") / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_v5_validation_" + uuid4().hex[:12])
    folder.mkdir()
    backend = None
    if args.vlm:
        from ddm_mcts.robotics.vlm_backend import Qwen3VLBackend

        backend = Qwen3VLBackend()
    report = {"scenario": args.scenario, "trials": [], "successes": 0}
    for index in range(args.trials):
        world = panda_reasoning_scene(args.model)
        camera = ManipulationCamera(world) if backend else None
        perception = ManipulationPerception(world, backend) if backend else None
        agent = CompositionalPhysicalAgent(world, observation=camera, perception=perception)
        try:
            result = agent.run_goal_instruction(INSTRUCTIONS[args.scenario], folder)
            report["trials"].append(
                {
                    "index": index,
                    "success": result.success,
                    "folder": str(result.folder),
                    "physics": result.trace["live_physics"],
                    "search_substeps": result.trace["speculative_substeps"],
                    "goals": result.trace["goal_results"],
                    "final_world": result.trace["final_world"],
                    "duration": result.trace["duration_seconds"],
                    "failure": result.trace.get("failure"),
                }
            )
            report["successes"] += result.success
            if backend:
                report["vlm"] = backend.diagnostics()
            (folder / "validation.json").write_text(json.dumps(report, indent=2, default=str))
        finally:
            if camera:
                camera.close()
    print(json.dumps({"successes": report["successes"], "trials": args.trials, "manifest": str(folder / "validation.json")}))
    return 0 if report["successes"] == args.trials else 1


if __name__ == "__main__":
    raise SystemExit(main())

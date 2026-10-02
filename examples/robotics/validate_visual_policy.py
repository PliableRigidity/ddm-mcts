"""Small real-Qwen acceptance utility, with one shared model session."""

import argparse
import json
from pathlib import Path
from uuid import uuid4


def validate(model_path, goals=("cylinder", "cube", "sphere"), *, viewer=False):
    from ddm_mcts.robotics.approach import ApproachReachTask
    from ddm_mcts.robotics.camera import MujocoCameraObservationProvider
    from ddm_mcts.robotics.core import RoboticsPlanner
    from ddm_mcts.robotics.observation import SemanticGoal
    from ddm_mcts.robotics.physical_agent import PhysicalAgent
    from ddm_mcts.robotics.qwen_visual_policy import QwenVisualPolicy
    from ddm_mcts.robotics.semantic_scene import OBJECT_RGB, SEMANTIC_PLANE_Z, panda_semantic_reach
    from ddm_mcts.robotics.visual import VisualInspection
    from ddm_mcts.robotics.vlm_backend import Qwen3VLBackend, VLMConfig
    from ddm_mcts.robotics.vlm_perception import VLMSemanticPerception
    from ddm_mcts.search.mcts import MCTSConfig

    backend = Qwen3VLBackend(VLMConfig(device="cuda:0"))
    records = []
    for label in goals:
        print(f"Starting real visual-policy goal: {label}", flush=True)
        world = panda_semantic_reach(model_path)
        planner = RoboticsPlanner(world, config=MCTSConfig(60, c_puct=0.05, seed=0))
        policy = QwenVisualPolicy(backend)
        with PhysicalAgent(
            world,
            MujocoCameraObservationProvider(world, width=640, height=480),
            VLMSemanticPerception(backend, SEMANTIC_PLANE_Z, OBJECT_RGB),
            ApproachReachTask(SemanticGoal(f"Reach the {label} object.")),
            planner,
            diagnostics=world.target_entities,
            visual_policy=policy,
        ) as agent:
            if viewer:
                import time

                agent.prepare()
                with VisualInspection(world, planner, speed=3) as visual:
                    folder = agent.run(visualization=visual)
                    agent.close()
                    time.sleep(2)  # Automated smoke only; normal CLI holds until user closes.
            else:
                folder = agent.run()
        summary = json.loads((folder / "summary.json").read_text())
        steps = [json.loads(line) for line in (folder / "steps.jsonl").read_text().splitlines()]
        row = {
            "goal": label,
            "run": str(folder),
            "success": summary["success"],
            "actions": [r["action"]["name"] for r in steps],
            "model_top1": [r["visual_decision"]["model_top1"] for r in steps],
            "overrides": sum(not r["visual_decision"]["selected_equals_model_top1"] for r in steps),
            "visual_policy": policy.diagnostics(),
            "tcp": summary["execution_diagnostics"],
            "mcts_search_seconds": sum(r["mcts_search_seconds"] for r in steps),
            "execution_seconds": sum(r["execution_seconds"] for r in steps),
            "minimum_selected_clearance_m": min(r["execution_diagnostics"]["minimum_gripper_target_distance_m"] for r in steps),
        }
        records.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("visual_policy", "actions", "model_top1")}), flush=True)
    folder = Path("robotics_runs") / f"visual_policy_acceptance_{uuid4().hex[:12]}"
    folder.mkdir(exist_ok=False)
    (folder / "acceptance.json").write_text(json.dumps({"records": records, "backend": backend.diagnostics()}, indent=2) + "\n")
    print(f"Acceptance summary: {folder}", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--goal", choices=("cylinder", "cube", "sphere"), action="append")
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    rows = validate(args.model, args.goal or ("cylinder", "cube", "sphere"), viewer=args.viewer)
    return 0 if all(r["success"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Real image + goal + action scoring, without planning or physical execution."""

import argparse
import json
from pathlib import Path
from uuid import uuid4


def main():
    from ddm_mcts.robotics.camera import MujocoCameraObservationProvider
    from ddm_mcts.robotics.observation import SemanticGoal
    from ddm_mcts.robotics.qwen_visual_policy import QwenVisualDecisionModel
    from ddm_mcts.robotics.semantic_scene import panda_semantic_reach
    from ddm_mcts.robotics.vlm_backend import Qwen3VLBackend, VLMConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--goal", default="Reach the cylindrical object.")
    parser.add_argument("--vlm-device", default="auto")
    args = parser.parse_args()
    world = panda_semantic_reach(args.model)
    backend = Qwen3VLBackend(VLMConfig(device=args.vlm_device))
    model = QwenVisualDecisionModel(backend)
    with MujocoCameraObservationProvider(world, width=640, height=480) as camera:
        observation = camera.observe()
        model.bind_context(observation.camera, observation.robot.position)
        try:
            scores = model.score(observation.rgb, SemanticGoal(args.goal), sorted(world.actions, key=lambda a: a.name))
        except ValueError:
            print(json.dumps(backend.diagnostics(), indent=2), flush=True)
            raise
    result = {
        "goal": args.goal,
        "priors": {a.name: p for a, p in scores.items()},
        "model_top1": max(scores, key=scores.get).name,
        "backend": backend.diagnostics(),
        "presentations": model.records,
    }
    folder = Path("robotics_runs") / f"visual_decision_sanity_{uuid4().hex[:12]}"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**result, "output": str(folder)}, indent=2), flush=True)


if __name__ == "__main__":
    main()

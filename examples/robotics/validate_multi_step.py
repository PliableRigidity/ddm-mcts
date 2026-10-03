"""Real cached-Qwen multi-step acceptance, with an optional bounded viewer smoke.

Normal CLI viewer runs hold the final scene until the user closes it. This utility
closes its one viewer session automatically after the full task, for validation.
"""

import argparse
import json


def make_agent(model_path, *, visual=False, backend=None):
    from ddm_mcts.robotics import PhysicalAgent, RoboticsPlanner, SemanticGoal
    from ddm_mcts.robotics.approach import ApproachReachTask
    from ddm_mcts.robotics.camera import MujocoCameraObservationProvider
    from ddm_mcts.robotics.qwen_visual_policy import QwenVisualPolicy
    from ddm_mcts.robotics.semantic_scene import OBJECT_RGB, SEMANTIC_PLANE_Z, panda_semantic_reach
    from ddm_mcts.robotics.vlm_backend import Qwen3VLBackend, VLMConfig
    from ddm_mcts.robotics.vlm_perception import VLMSemanticPerception
    from ddm_mcts.search.mcts import MCTSConfig

    backend = backend or Qwen3VLBackend(VLMConfig(device="cuda:0"))
    world = panda_semantic_reach(model_path)
    return PhysicalAgent(
        world,
        MujocoCameraObservationProvider(world, width=640, height=480),
        VLMSemanticPerception(backend, SEMANTIC_PLANE_Z, OBJECT_RGB),
        ApproachReachTask(SemanticGoal("cylinder")),
        RoboticsPlanner(world, config=MCTSConfig(60, c_puct=0.05 if visual else 1.4, seed=0)),
        diagnostics=world.target_entities,
        visual_policy=QwenVisualPolicy(backend, alpha=0.5) if visual else None,
    )


def validate(model_path, instruction, *, visual=False, viewer=False, output="robotics_runs"):
    from ddm_mcts.robotics import parse_physical_task

    task = parse_physical_task(instruction)
    with make_agent(model_path, visual=visual) as agent:
        if viewer:
            from ddm_mcts.robotics.visual import VisualInspection

            with VisualInspection(agent.environment, agent.planner, speed=3) as inspection:
                result = agent.run_task(task, output, visualization=inspection)
                agent.close()  # Release offscreen GL before the viewer context.
        else:
            result = agent.run_task(task, output)
    print(json.dumps({"output": str(result.folder), "success": result.success}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--instruction", default="Approach the cylinder, then the sphere, then the cube.")
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--output", default="robotics_runs")
    args = parser.parse_args()
    return 0 if validate(args.model, args.instruction, visual=args.visual, viewer=args.viewer, output=args.output).success else 1


if __name__ == "__main__":
    raise SystemExit(main())

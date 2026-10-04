"""Additive compositional-goal CLI; frozen V4 modes keep their existing paths."""

import json

from ddm_mcts.search.mcts import MCTSConfig

from .manipulation_observation import ManipulationCamera, ManipulationPerception
from .physical_goals import parse_physical_goals
from .physical_reasoning import CompositionalPhysicalAgent
from .reasoning_scene import panda_reasoning_scene


def run_physical_reasoning(args, parser):
    if not args.model or not args.instruction:
        parser.error("physical-reason requires --model/PANDA_MODEL and --instruction")
    try:
        parse_physical_goals(args.instruction)
    except ValueError as exc:
        parser.error(str(exc))
    if args.perception not in (None, "ground-truth", "vlm") or args.decision_policy != "structured" or args.policy != "uniform":
        parser.error("physical-reason supports deterministic or semantic VLM grounding and uniform high-level priors")
    world = panda_reasoning_scene(args.model)
    camera = perception = None
    if args.perception == "vlm":
        from .vlm_backend import Qwen3VLBackend, VLMConfig

        backend = Qwen3VLBackend(VLMConfig(args.vlm_model, args.vlm_device, args.vlm_max_new_tokens))
        camera = ManipulationCamera(world, max(args.width, 640), max(args.height, 480))
        perception = ManipulationPerception(world, backend)
    agent = CompositionalPhysicalAgent(
        world,
        observation=camera,
        perception=perception,
        search_config=MCTSConfig(args.simulations, c_puct=args.c_puct if args.c_puct is not None else 0.05, seed=args.seed),
    )
    try:
        if args.viewer:
            from .manipulation_visual import PickupInspection

            with PickupInspection(world, args.viewer_speed) as visual:
                result = agent.run_goal_instruction(args.instruction, args.output, visualization=visual)
                print(json.dumps(result.trace, indent=2) if args.diagnostics else f"Success: {result.success}; logs: {result.folder}")
                print("Final scene remains open. Close viewer or press Ctrl+C.", flush=True)
                visual.wait_until_closed()
        else:
            result = agent.run_goal_instruction(args.instruction, args.output)
            print(json.dumps(result.trace, indent=2) if args.diagnostics else f"Success: {result.success}; logs: {result.folder}")
    finally:
        if camera:
            camera.close()
    return 0 if result.success else 1

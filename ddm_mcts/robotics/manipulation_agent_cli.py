"""Language manipulation CLI composition; existing task modes are untouched."""

import json
from math import isfinite

from ddm_mcts.search.mcts import MCTSConfig

from .manipulation_agent import PhysicalAIManipulationAgent
from .manipulation_observation import ManipulationCamera, ManipulationPerception
from .manipulation_scene import panda_pickup_scene
from .manipulation_task import parse_manipulation_task


def run_manipulation(args, parser):
    if not args.model or not args.instruction:
        parser.error("manipulate requires --model/PANDA_MODEL and --instruction")
    if args.goal or args.target or args.place_position or args.place_next_to:
        parser.error("manipulate destinations are supplied through the bounded instruction")
    if args.decision_policy != "structured" or args.policy != "uniform":
        parser.error("manipulation CLI uses uniform high-level priors; Cartesian visual/DDM policies remain in reach modes")
    if args.perception not in (None, "ground-truth", "vlm"):
        parser.error("manipulate supports explicit known-scene ground-truth or VLM semantic grounding")
    if not isfinite(args.viewer_speed) or args.viewer_speed <= 0 or args.simulations < 2:
        parser.error("positive viewer speed and at least two search simulations required")
    if args.c_puct is not None and (not isfinite(args.c_puct) or args.c_puct < 0):
        parser.error("--c-puct must be finite and nonnegative")
    try:
        task = parse_manipulation_task(args.instruction)
    except ValueError as exc:
        parser.error(str(exc))
    world = panda_pickup_scene(args.model)
    backend = None
    observation = None
    if args.perception == "vlm":
        from .vlm_backend import Qwen3VLBackend, VLMConfig

        backend = Qwen3VLBackend(VLMConfig(args.vlm_model, args.vlm_device, args.vlm_max_new_tokens))
        observation = ManipulationCamera(world, max(args.width, 640), max(args.height, 480))
    agent = PhysicalAIManipulationAgent(
        world,
        observation=observation,
        perception=ManipulationPerception(world, backend),
        search_config=MCTSConfig(args.simulations, c_puct=args.c_puct if args.c_puct is not None else 0.05, seed=args.seed),
    )
    try:
        if args.viewer:
            from .manipulation_visual import PickupInspection

            with PickupInspection(world, args.viewer_speed) as visual:
                result = agent.run_task(task, args.output, visualization=visual)
                agent.close()
                print(json.dumps(result.trace, indent=2) if args.diagnostics else f"Success: {result.success}; logs: {result.folder}")
                print("Final scene remains open. Close viewer or press Ctrl+C.", flush=True)
                visual.wait_until_closed()
        else:
            result = agent.run_task(task, args.output)
            print(json.dumps(result.trace, indent=2) if args.diagnostics else f"Success: {result.success}; logs: {result.folder}")
    except KeyboardInterrupt:
        return 130
    finally:
        agent.close()
    return 0 if result.success else 1

"""Panda reach command with optional visual inspection; independent of the historical experiment CLI."""

import argparse
import json
import os

from ddm_mcts.policies.mica_policy import MicaPolicy
from ddm_mcts.policies.random_policy import UniformPolicy
from ddm_mcts.policies.text_laya_policy import TextLayaPolicy
from ddm_mcts.search.mcts import MCTSConfig

from .core import RoboticsPlanner
from .reach import ReachTask, panda_reach
from .run import run_episode


def state_text(state):
    observation = state.observation
    return f"Hand XYZ: {observation.position}; target XYZ: {observation.goal}; steps: {observation.steps}"


def build_parser():
    parser = argparse.ArgumentParser(description="Physical MCTS Panda reach (headless by default)")
    parser.add_argument("--model", default=os.environ.get("PANDA_MODEL"), help="Menagerie Panda scene.xml or panda.xml")
    parser.add_argument("--target", nargs=3, type=float, help="XYZ required for the original reach task")
    parser.add_argument("--policy", choices=("uniform", "laya", "mica"), default="uniform")
    parser.add_argument("--simulations", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--increment", type=float, default=0.02)
    parser.add_argument("--epsilon", type=float, default=0.012)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--physics-steps", type=int, default=150)
    parser.add_argument("--planar", action="store_true", help="second configuration: X/Y actions only")
    parser.add_argument("--output", default="robotics_runs")
    parser.add_argument("--viewer", action="store_true", help="observe selected actions in the MuJoCo viewer")
    parser.add_argument("--viewer-speed", type=float, default=1.0, help="execution speed multiplier (0.5 is slower; default 1)")
    parser.add_argument("--task", choices=("reach", "visual-reach", "semantic-reach"), default="reach")
    parser.add_argument("--goal", default=None, help="red/blue for visual-reach; language goal for semantic-reach")
    parser.add_argument("--observation", choices=("ground-truth", "camera"), help="visual-reach defaults to camera")
    parser.add_argument("--perception", choices=("ground-truth", "color", "vlm"), help="defaults to match task/observation mode")
    parser.add_argument("--vlm-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--vlm-device", default="auto")
    parser.add_argument("--vlm-max-new-tokens", type=int, default=256)
    parser.add_argument("--vlm-refresh", type=int, default=0, help="semantic inference interval; 0 caches until goal/reset/camera changes")
    parser.add_argument("--vlm-debug", action="store_true", help="save live input/grounding images and semantic diagnostics")
    parser.add_argument("--standoff", type=float, default=0.05, help="semantic-reach clearance beyond object bounding radius (meters)")
    parser.add_argument("--object-radius", type=float, default=0.04, help="semantic-reach conservative bounding-radius prior (meters)")
    parser.add_argument("--camera", default="inspection")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--save-images", action="store_true", help="save one RGB PPM per live observation")
    parser.add_argument("--diagnostics", action="store_true", help="log true state and target-position errors separately")
    return parser


def main(argv=None):
    from math import isfinite

    parser = build_parser()
    args = parser.parse_args(argv)
    args.goal = args.goal or ("Reach the cylindrical object." if args.task == "semantic-reach" else "red")
    if args.task == "visual-reach" and args.goal not in ("red", "blue"):
        parser.error("visual-reach goal must be red or blue")
    if args.task != "semantic-reach" and args.perception == "vlm":
        parser.error("VLM perception requires --task semantic-reach")
    if not isfinite(args.viewer_speed) or args.viewer_speed <= 0:
        parser.error("--viewer-speed must be finite and positive")
    if not args.model:
        parser.error("provide --model or PANDA_MODEL; see docs/robotics/README.md")
    if args.task == "reach":
        if args.target is None:
            parser.error("--target X Y Z is required for --task reach")
        if args.observation == "camera" or args.perception in ("color", "vlm"):
            parser.error("camera/color input requires --task visual-reach")
        world = panda_reach(
            args.model,
            ReachTask(tuple(args.target), args.epsilon, args.max_steps),
            increment=args.increment,
            axes=(0, 1) if args.planar else (0, 1, 2),
            physics_steps=args.physics_steps,
        )
    else:
        from .visual_scene import panda_visual_reach

        if args.target is not None:
            parser.error("visual-reach resolves --goal from observation; omit --target")
        factory = panda_visual_reach
        if args.task == "semantic-reach":
            from .semantic_scene import panda_semantic_reach

            factory = panda_semantic_reach
        world = factory(
            args.model,
            increment=args.increment,
            axes=(0, 1) if args.planar else (0, 1, 2),
            physics_steps=args.physics_steps,
            **({"use_tcp": True} if args.task == "semantic-reach" else {}),
        )
    policy = UniformPolicy()
    if args.policy != "uniform":
        cls = TextLayaPolicy if args.policy == "laya" else MicaPolicy
        policy = cls(state_text, objective="Move the hand toward the target XYZ.", action_text=lambda a: f"{a.name}: {a.displacement}")
    planner = RoboticsPlanner(world, policy, MCTSConfig(args.simulations, seed=args.seed), args.horizon)
    agent = None
    if args.task != "reach" or args.observation is not None or args.perception is not None:
        from .observation import GroundTruthObservationProvider, SemanticGoal
        from .perception import GroundTruthPerception
        from .physical_agent import PhysicalAgent
        from .representation import CoordinateReachTask, SemanticReachTask

        mode = args.observation or ("camera" if args.task != "reach" else "ground-truth")
        perception_mode = args.perception or ("vlm" if args.task == "semantic-reach" else ("color" if mode == "camera" else "ground-truth"))
        if args.task == "semantic-reach" and (mode, perception_mode) != ("camera", "vlm"):
            parser.error("semantic-reach requires camera/vlm; use the Python API for explicit ground-truth baselines")
        if (mode, perception_mode) not in (("camera", "color"), ("camera", "vlm"), ("ground-truth", "ground-truth")):
            parser.error("use camera/color or ground-truth/ground-truth")
        entity_provider = getattr(world, "target_entities", None)
        if mode == "camera":
            from .camera import MujocoCameraObservationProvider
            from .color_perception import ColorPlanePerception
            from .visual_scene import TARGET_PLANE_Z

            width, height = (max(args.width, 640), max(args.height, 480)) if perception_mode == "vlm" else (args.width, args.height)
            observation_provider = MujocoCameraObservationProvider(world, args.camera, width, height)
            perception_provider = ColorPlanePerception(TARGET_PLANE_Z)
            if perception_mode == "vlm":
                from .semantic_scene import OBJECT_RGB, SEMANTIC_PLANE_Z
                from .vlm_backend import Qwen3VLBackend, VLMConfig
                from .vlm_perception import VLMSemanticPerception

                perception_provider = VLMSemanticPerception(
                    Qwen3VLBackend(VLMConfig(args.vlm_model, args.vlm_device, args.vlm_max_new_tokens)),
                    SEMANTIC_PLANE_Z,
                    OBJECT_RGB,
                    refresh=args.vlm_refresh,
                    debug=args.vlm_debug,
                )
        else:
            observation_provider = GroundTruthObservationProvider(world, entity_provider)
            perception_provider = GroundTruthPerception()
        task = (
            SemanticReachTask(SemanticGoal(args.goal), args.epsilon, args.max_steps)
            if args.task != "reach"
            else CoordinateReachTask(epsilon=args.epsilon, max_steps=args.max_steps)
        )
        if args.task == "semantic-reach":
            from .approach import ApproachReachTask

            task = ApproachReachTask(SemanticGoal(args.goal), args.epsilon, args.max_steps, args.standoff, args.object_radius)
        agent = PhysicalAgent(
            world,
            observation_provider,
            perception_provider,
            task,
            planner,
            diagnostics=entity_provider if args.diagnostics else None,
            save_images=args.save_images,
        )
    if args.viewer:
        from .visual import ViewerClosed, VisualInspection

        try:
            if agent is not None:
                agent.prepare()  # Resolve the live target before placing the viewer marker.
            with VisualInspection(world, planner, args.viewer_speed) as observer:
                try:
                    folder = (
                        run_episode(world, planner, args.output, configuration=vars(args), observer=observer)
                        if agent is None
                        else agent.run(args.output, visualization=observer, configuration=vars(args))
                    )
                    if agent is not None:
                        agent.close()  # Release offscreen GL while the viewer context is still alive.
                        execution_diagnostics = agent.execution_diagnostics()
                        if execution_diagnostics:
                            print(json.dumps({"semantic_approach": execution_diagnostics}, indent=2), flush=True)
                    print(json.dumps({"output": str(folder), "success": world.task.is_success(world.get_state())}), flush=True)
                    observer.wait_until_closed()
                finally:
                    if agent is not None:
                        agent.close()
        except ViewerClosed:
            print("Viewer closed; execution stopped. Run diagnostics retained.", flush=True)
            return 1
        except KeyboardInterrupt:
            print("Interrupted; viewer closed and run diagnostics retained.", flush=True)
            return 130
        finally:
            if agent is not None:
                agent.close()
        return 0 if world.task.is_success(world.get_state()) else 1
    if agent is not None:
        with agent:
            folder = agent.run(args.output, configuration=vars(args))
        print(json.dumps({"output": str(folder), "success": world.task.is_success(world.get_state())}))
        return 0 if world.task.is_success(world.get_state()) else 1
    folder = run_episode(world, planner, args.output, configuration=vars(args))
    print(json.dumps({"output": str(folder), "success": world.task.is_success(world.get_state())}))
    return 0 if world.task.is_success(world.get_state()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

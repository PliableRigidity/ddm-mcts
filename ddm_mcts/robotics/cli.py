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
    parser.add_argument("--task", choices=("reach", "visual-reach"), default="reach")
    parser.add_argument("--goal", choices=("red", "blue"), default="red", help="semantic target for visual-reach")
    parser.add_argument("--observation", choices=("ground-truth", "camera"), help="visual-reach defaults to camera")
    parser.add_argument("--perception", choices=("ground-truth", "color"), help="defaults to match observation mode")
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
    if not isfinite(args.viewer_speed) or args.viewer_speed <= 0:
        parser.error("--viewer-speed must be finite and positive")
    if not args.model:
        parser.error("provide --model or PANDA_MODEL; see docs/robotics/README.md")
    if args.task == "reach":
        if args.target is None:
            parser.error("--target X Y Z is required for --task reach")
        if args.observation == "camera" or args.perception == "color":
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
        world = panda_visual_reach(
            args.model, increment=args.increment, axes=(0, 1) if args.planar else (0, 1, 2), physics_steps=args.physics_steps
        )
    policy = UniformPolicy()
    if args.policy != "uniform":
        cls = TextLayaPolicy if args.policy == "laya" else MicaPolicy
        policy = cls(state_text, objective="Move the hand toward the target XYZ.", action_text=lambda a: f"{a.name}: {a.displacement}")
    planner = RoboticsPlanner(world, policy, MCTSConfig(args.simulations, seed=args.seed), args.horizon)
    agent = None
    if args.task == "visual-reach" or args.observation is not None or args.perception is not None:
        from .observation import GroundTruthObservationProvider, SemanticGoal
        from .perception import GroundTruthPerception
        from .physical_agent import PhysicalAgent
        from .representation import CoordinateReachTask, SemanticReachTask

        mode = args.observation or ("camera" if args.task == "visual-reach" else "ground-truth")
        perception_mode = args.perception or ("color" if mode == "camera" else "ground-truth")
        if (mode, perception_mode) not in (("camera", "color"), ("ground-truth", "ground-truth")):
            parser.error("use camera/color or ground-truth/ground-truth")
        entity_provider = getattr(world, "target_entities", None)
        if mode == "camera":
            from .camera import MujocoCameraObservationProvider
            from .color_perception import ColorPlanePerception
            from .visual_scene import TARGET_PLANE_Z

            observation_provider = MujocoCameraObservationProvider(world, args.camera, args.width, args.height)
            perception_provider = ColorPlanePerception(TARGET_PLANE_Z)
        else:
            observation_provider = GroundTruthObservationProvider(world, entity_provider)
            perception_provider = GroundTruthPerception()
        task = (
            SemanticReachTask(SemanticGoal(args.goal), args.epsilon, args.max_steps)
            if args.task == "visual-reach"
            else CoordinateReachTask(epsilon=args.epsilon, max_steps=args.max_steps)
        )
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

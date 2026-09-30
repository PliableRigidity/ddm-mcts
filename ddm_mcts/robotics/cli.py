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
    parser.add_argument("--target", nargs=3, type=float, required=True)
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
    return parser


def main(argv=None):
    from math import isfinite

    parser = build_parser()
    args = parser.parse_args(argv)
    if not isfinite(args.viewer_speed) or args.viewer_speed <= 0:
        parser.error("--viewer-speed must be finite and positive")
    if not args.model:
        parser.error("provide --model or PANDA_MODEL; see docs/robotics/README.md")
    world = panda_reach(
        args.model,
        ReachTask(tuple(args.target), args.epsilon, args.max_steps),
        increment=args.increment,
        axes=(0, 1) if args.planar else (0, 1, 2),
        physics_steps=args.physics_steps,
    )
    policy = UniformPolicy()
    if args.policy != "uniform":
        cls = TextLayaPolicy if args.policy == "laya" else MicaPolicy
        policy = cls(state_text, objective="Move the hand toward the target XYZ.", action_text=lambda a: f"{a.name}: {a.displacement}")
    planner = RoboticsPlanner(world, policy, MCTSConfig(args.simulations, seed=args.seed), args.horizon)
    if args.viewer:
        from .visual import ViewerClosed, VisualInspection

        try:
            with VisualInspection(world, planner, args.viewer_speed) as observer:
                folder = run_episode(world, planner, args.output, configuration=vars(args), observer=observer)
                print(json.dumps({"output": str(folder), "success": world.task.is_success(world.get_state())}), flush=True)
                observer.wait_until_closed()
        except ViewerClosed:
            print("Viewer closed; execution stopped. Run diagnostics retained.", flush=True)
            return 1
        except KeyboardInterrupt:
            print("Interrupted; viewer closed and run diagnostics retained.", flush=True)
            return 130
        return 0 if world.task.is_success(world.get_state()) else 1
    folder = run_episode(world, planner, args.output, configuration=vars(args))
    print(json.dumps({"output": str(folder), "success": world.task.is_success(world.get_state())}))
    return 0 if world.task.is_success(world.get_state()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

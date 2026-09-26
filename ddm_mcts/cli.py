from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence

from ddm_mcts.agents import LayaAgent, LayaMCTSAgent, MCTSAgent, RandomAgent
from ddm_mcts.environments import ConnectFour
from ddm_mcts.evaluation import Arena
from ddm_mcts.evaluation.v4_experiment import run_v4_experiment
from ddm_mcts.policies import LayaPolicy, LayaUnavailableError, MicaUnavailableError


def _agent(name: str, env: ConnectFour, simulations: int, seed: int, args: argparse.Namespace):
    if name == "random":
        return RandomAgent(env, seed)
    if name == "mcts":
        return MCTSAgent(env, simulations, seed)
    policy = LayaPolicy(device=args.laya_device, model=args.laya_model)
    if name == "laya":
        return LayaAgent(env, policy, seed)
    if name == "laya-mcts":
        return LayaMCTSAgent(
            env,
            policy,
            simulations,
            seed,
            adaptive=args.adaptive,
            min_simulations=args.min_simulations,
            max_simulations=args.max_simulations,
        )
    raise ValueError(name)


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations per move")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--laya-device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--laya-model", choices=("english", "multilingual", "typed-decisions"))
    parser.add_argument("--adaptive", action="store_true", help="map Laya entropy to a search budget")
    parser.add_argument("--min-simulations", type=int, default=10)
    parser.add_argument("--max-simulations", type=int, default=500)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ddm-mcts", description="DDM-guided MCTS experiments on Connect Four")
    sub = parser.add_subparsers(dest="command", required=True)
    play = sub.add_parser("play", help="play Connect Four against an agent")
    play.add_argument("--agent", choices=("random", "mcts", "laya", "laya-mcts"), default="mcts")
    play.add_argument("--human-player", type=int, choices=(1, 2), default=1)
    _add_shared(play)
    benchmark = sub.add_parser("benchmark", help="benchmark two agents, alternating first player")
    benchmark.add_argument("--agent-a", choices=("random", "mcts", "laya", "laya-mcts"), required=True)
    benchmark.add_argument("--agent-b", choices=("random", "mcts", "laya", "laya-mcts"), required=True)
    benchmark.add_argument("--games", type=int, default=10)
    benchmark.add_argument("--budgets", type=int, nargs="+", help="run one experiment per MCTS budget")
    benchmark.add_argument("--output", help="write .json/.csv (single budget only)")
    benchmark.add_argument("--verbose", action="store_true")
    _add_shared(benchmark)
    experiment = sub.add_parser("experiment", help="run the complete automated diagnostic suite")
    experiment.add_argument("--games", type=int, default=10, help="paired games per search budget")
    experiment.add_argument("--budgets", type=int, nargs="+", help="override mode-specific budgets")
    experiment.add_argument("--seed", type=int, default=0)
    modes = experiment.add_mutually_exclusive_group()
    modes.add_argument("--quick", action="store_true", help="small sanity run (default)")
    modes.add_argument("--standard", action="store_true", help="moderate multi-scenario run")
    modes.add_argument("--extended", action="store_true", help="larger, long-running experiment")
    experiment.add_argument("--results-root", default="results")
    experiment.add_argument("--laya-device", choices=("cpu", "cuda", "mps"))
    experiment.add_argument("--laya-model", choices=("english", "multilingual", "typed-decisions"))
    experiment.add_argument("--skip-laya", action="store_true", help="run only policy-independent diagnostics")
    experiment.add_argument("--ddm", choices=("laya", "mica", "all"), default="all")
    experiment.add_argument("--mica-endpoint", default="http://127.0.0.1:8010/v1/systemone")
    experiment.add_argument("--mica-model", default="mica-v0.1-4b")
    experiment.add_argument("--order-stability-check", choices=("on", "off"), default="on")
    experiment.add_argument("--order-samples", type=int, default=2, help="total option-order samples including normal order")
    return parser


def _play(args: argparse.Namespace) -> None:
    env = ConnectFour()
    agent = _agent(args.agent, env, args.simulations, args.seed, args)
    state = env.initial_state()
    while not env.is_terminal(state):
        print(env.render(state))
        if env.current_player(state) == args.human_player:
            try:
                action = int(input("Column: "))
                state = env.step(state, action)
            except (ValueError, EOFError) as exc:
                print(f"Invalid move: {exc}")
        else:
            decision = agent.decide(state)
            print(f"{agent.name} chooses {decision.action}")
            state = env.step(state, decision.action)
    print(env.render(state))


def _benchmark(args: argparse.Namespace) -> None:
    budgets = args.budgets or [args.simulations]
    if args.output and len(budgets) != 1:
        raise SystemExit("--output currently requires one budget")
    for budget in budgets:
        env = ConnectFour()
        a = _agent(args.agent_a, env, budget, args.seed, args)
        b = _agent(args.agent_b, env, budget, args.seed + 1, args)
        result = Arena(env).run(a, b, args.games, seed=args.seed, verbose=args.verbose, metadata={"simulations": budget})
        print(json.dumps(result.to_dict(), indent=2))
        if args.output:
            result.export(args.output)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING)
    try:
        if args.command == "play":
            _play(args)
        elif args.command == "benchmark":
            _benchmark(args)
        else:
            mode = "extended" if args.extended else "standard" if args.standard else "quick"
            run_v4_experiment(
                mode=mode,
                ddm=args.ddm,
                budgets=args.budgets,
                seed=args.seed,
                results_root=args.results_root,
                laya_device=args.laya_device,
                laya_model=args.laya_model,
                mica_endpoint=args.mica_endpoint,
                mica_model=args.mica_model,
                skip_laya=args.skip_laya,
                order_stability_check=args.order_stability_check == "on",
                order_samples=args.order_samples,
            )
    except (LayaUnavailableError, MicaUnavailableError) as exc:
        parser.exit(2, f"DDM unavailable: {exc}\n")


if __name__ == "__main__":
    main()

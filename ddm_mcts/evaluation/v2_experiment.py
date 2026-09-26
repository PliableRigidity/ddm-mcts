from __future__ import annotations

import csv
import json
import logging
import math
import random
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any

from ddm_mcts.agents import AdaptiveLayaMCTSAgent, AdaptiveSearchConfig, LayaAgent, LayaMCTSAgent, LayaRootMCTSAgent
from ddm_mcts.environments import ConnectFour, DelayedReward, DeliveryRouting, GridNavigation, InventoryManagement, JobScheduling
from ddm_mcts.environments.base import Environment
from ddm_mcts.environments.delayed_reward import delayed_reward_scenarios
from ddm_mcts.environments.delivery_routing import routing_scenarios
from ddm_mcts.environments.grid_navigation import grid_scenarios
from ddm_mcts.environments.job_scheduling import scheduling_scenarios
from ddm_mcts.policies import LayaPolicy, LayaUnavailableError, MixedPolicy, TextLayaPolicy
from ddm_mcts.search import MCTS, MCTSConfig

from .experiment import FixedPriorPolicy, _play

LOGGER = logging.getLogger(__name__)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(slots=True)
class Scenario:
    domain: str
    title: str
    scenario_id: str
    environment: Environment[Any, Any]
    state: Any
    objective: str
    metric: str
    stochastic: bool = False
    known_actions: frozenset[Any] | None = None
    known_value: float | None = None


def regret(optimal: float | None, achieved: float) -> float | None:
    """Regret for higher-is-better objectives, robust to floating-point noise."""
    if optimal is None:
        return None
    return max(0.0, optimal - achieved)


def optimal_action_analysis(environment: Environment[Any, Any], state: Any) -> tuple[frozenset[Any], float, dict[Any, float]]:
    """Exactly solve a small deterministic state and retain every tied optimum."""
    @cache
    def value(current: Any) -> float:
        if environment.is_terminal(current):
            return environment.get_reward(current, 1)
        return max(value(environment.step(current, action)) for action in environment.legal_actions(current))

    action_values = {action: value(environment.step(state, action)) for action in environment.legal_actions(state)}
    best = max(action_values.values())
    optimal = frozenset(action for action, candidate in action_values.items() if math.isclose(candidate, best, rel_tol=1e-9, abs_tol=1e-9))
    return optimal, best, action_values


def option_order_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare semantic-action probabilities after presentation-order changes."""
    if not rows:
        return {"status": "unavailable", "samples": 0}
    changes, top_matches, correlations = [], [], []
    by_case: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault((row["domain"], row["scenario_id"]), []).append(row)
    for case_rows in by_case.values():
        by_order: dict[str, dict[Any, dict[str, Any]]] = {}
        for row in case_rows:
            by_order.setdefault(row["order"], {})[row["action"]] = row
        normal = by_order.get("normal")
        if not normal:
            continue
        normal_top = min(normal.values(), key=lambda item: item["rank"])["action"]
        for order, measured in by_order.items():
            if order == "normal" or set(measured) != set(normal):
                continue
            changes.extend(abs(float(measured[action]["probability"]) - float(normal[action]["probability"])) for action in normal)
            top_matches.append(min(measured.values(), key=lambda item: item["rank"])["action"] == normal_top)
            left = [normal[action]["rank"] for action in normal]
            right = [measured[action]["rank"] for action in normal]
            n = len(left)
            correlations.append(1.0 if n < 2 else 1 - 6 * sum((a - b) ** 2 for a, b in zip(left, right, strict=True)) / (n * (n * n - 1)))
    return {
        "status": "complete",
        "samples": len(changes),
        "top_action_consistency": statistics.fmean(top_matches) if top_matches else None,
        "mean_absolute_probability_change": statistics.fmean(changes) if changes else None,
        "mean_rank_correlation": statistics.fmean(correlations) if correlations else None,
        "mean_probability_by_position": {
            str(position): statistics.fmean(float(row["probability"]) for row in rows if row["presentation_position"] == position)
            for position in sorted({row["presentation_position"] for row in rows})
        },
        "sensitivity_observed": bool(changes and (statistics.fmean(changes) > 0.05 or statistics.fmean(top_matches) < 0.8)),
    }


def _connect_scenarios() -> list[Scenario]:
    env = ConnectFour()
    definitions = [
        ("vertical_0", [0, 6, 0, 6, 0, 5], {0}),
        ("vertical_2", [2, 6, 2, 6, 2, 5], {2}),
        ("vertical_4", [4, 0, 4, 0, 4, 1], {4}),
        ("vertical_6", [6, 0, 6, 0, 6, 1], {6}),
        ("horizontal", [0, 6, 1, 6, 2, 5], {3}),
        ("two_wins", [1, 6, 2, 6, 3, 5], {0, 4}),
        ("block_horizontal", [6, 0, 6, 1, 5, 2], {3}),
        ("block_vertical", [0, 2, 1, 2, 4, 2], {2}),
        ("block_existing", [0, 3, 1, 3, 2, 3], {3}),
        ("diagonal_win", [0, 1, 1, 2, 4, 2, 2, 3, 4, 3, 5, 3], {3}),
    ]
    return [Scenario("connect_four", "Connect Four", name, env, _play(env, moves), "Choose a known immediate win or forced block.", "tactical success", known_actions=frozenset(actions), known_value=1.0) for name, moves, actions in definitions]


def _all_scenarios(mode: str, seed: int) -> list[Scenario]:
    limit = 2 if mode == "quick" else 10
    scenarios = _connect_scenarios()[:limit]
    for name, state in list(grid_scenarios().items())[:limit]:
        env = GridNavigation(state)
        scenarios.append(Scenario("grid_navigation", "Grid Navigation", name, env, state, "Reach the goal with fewer steps.", "terminal reward"))
    for name, jobs in list(scheduling_scenarios().items())[:limit]:
        env = JobScheduling(jobs)
        scenarios.append(Scenario("job_scheduling", "Job Scheduling", name, env, env.initial_state(), "Maximize job value after lateness penalties.", "schedule reward"))
    for name, distances in list(routing_scenarios().items())[:limit]:
        env = DeliveryRouting(distances)
        scenarios.append(Scenario("delivery_routing", "Delivery Routing", name, env, env.initial_state(), "Minimize total route cost.", "negative travel cost"))
    inventory_count = 3 if mode == "quick" else 10
    for index in range(inventory_count):
        env = InventoryManagement(seed + index)
        scenarios.append(Scenario("inventory", "Inventory Management", f"demand_seed_{seed + index}", env, env.initial_state(), "Minimize order, holding, and stockout cost across seeded demand.", "negative total cost", stochastic=True))
    for name, transitions in list(delayed_reward_scenarios().items())[:limit]:
        env = DelayedReward(transitions)
        scenarios.append(Scenario("delayed_reward", "Delayed Reward", name, env, env.initial_state(), "Maximize eventual rather than immediate reward.", "cumulative reward"))
    return scenarios


def _backend(device: str | None, model: str | None):
    try:
        from laya import Router
        router = Router(preload=False, **({"device": device} if device else {}))
        if model:
            def predict(state: Any, questions: dict[str, Any]) -> dict[str, Any]:
                return router.predict(state, questions, model=model)
            return predict, None
        return router.predict, None
    except Exception as exc:
        return None, str(exc)


def _policy(scenario: Scenario, predictor: Any):
    if scenario.domain == "connect_four":
        return LayaPolicy(predictor=predictor)
    return TextLayaPolicy(scenario.environment.render, objective=scenario.objective, action_text=lambda action: f"Choose {action}", predictor=predictor)


def _episode(scenario: Scenario, agent: Any) -> tuple[float, list[Any]]:
    state, actions = scenario.state, []
    while not scenario.environment.is_terminal(state):
        decision = agent.decide(state)
        actions.append(decision.action)
        state = scenario.environment.step(state, decision.action)
    return scenario.environment.get_reward(state, 1), actions


def _adaptive_episode(scenario: Scenario, agent: AdaptiveLayaMCTSAgent) -> tuple[float, list[Any], dict[str, Any], int]:
    state, actions, first_metadata, total_simulations = scenario.state, [], {}, 0
    while not scenario.environment.is_terminal(state):
        decision = agent.decide(state)
        if not actions:
            first_metadata = dict(decision.metadata)
        else:
            first_metadata["stability_changed_alpha"] = bool(first_metadata.get("stability_changed_alpha") or decision.metadata.get("stability_changed_alpha"))
            first_metadata["escalated"] = bool(first_metadata.get("escalated") or decision.metadata.get("escalated"))
            first_metadata["triggered_additional_search"] = bool(first_metadata.get("triggered_additional_search") or decision.metadata.get("triggered_additional_search"))
        actions.append(decision.action)
        total_simulations += decision.simulations
        state = scenario.environment.step(state, decision.action)
    return scenario.environment.get_reward(state, 1), actions, first_metadata, total_simulations


def _stats_delta(after: dict[str, Any], before: dict[str, Any], key: str) -> float:
    return float(after.get(key, 0)) - float(before.get(key, 0))


def _measure(
    scenario: Scenario,
    base_policy: Any,
    alpha: float,
    budget: int,
    seed: int,
    guidance: str,
    optimal_actions: frozenset[Any] | None,
    optimal_value: float | None,
) -> dict[str, Any]:
    before = base_policy.diagnostics()
    if guidance == "root":
        agent = LayaRootMCTSAgent(scenario.environment, base_policy, budget, seed, alpha=alpha)
    elif guidance == "full":
        agent = LayaMCTSAgent(scenario.environment, base_policy, budget, seed, alpha=alpha)
    else:
        agent = LayaAgent(scenario.environment, MixedPolicy(base_policy, alpha), seed)
    started = time.perf_counter()
    if scenario.domain == "connect_four":
        decision = agent.decide(scenario.state)
        actions = [decision.action]
        objective = float(optimal_actions is not None and decision.action in optimal_actions)
    else:
        objective, actions = _episode(scenario, agent)
    latency = time.perf_counter() - started
    after = base_policy.diagnostics()
    measured_regret = regret(optimal_value, objective)
    return {
        "domain": scenario.domain,
        "scenario_id": scenario.scenario_id,
        "guidance": guidance,
        "alpha": alpha,
        "budget": budget,
        "seed": seed,
        "objective": objective,
        "optimal_objective": optimal_value,
        "regret": measured_regret,
        "normalized_regret": measured_regret / max(1.0, abs(optimal_value)) if measured_regret is not None and optimal_value is not None else None,
        "first_action": actions[0],
        "optimal_actions": sorted(optimal_actions, key=str) if optimal_actions is not None else None,
        "first_action_optimal": actions[0] in optimal_actions if optimal_actions is not None else None,
        "simulations_per_decision": budget if guidance != "laya" else 0,
        "latency_seconds": latency,
        "laya_calls": _stats_delta(after, before, "actual_inference_calls"),
        "laya_requests": _stats_delta(after, before, "total_policy_requests"),
        "cache_hits": _stats_delta(after, before, "cache_hits"),
        "cache_misses": _stats_delta(after, before, "cache_misses"),
    }


def _order_rows(scenario: Scenario, policy: Any, seed: int) -> list[dict[str, Any]]:
    legal = list(scenario.environment.legal_actions(scenario.state))
    rng = random.Random(seed)
    orders = [("normal", legal), ("reversed", list(reversed(legal)))]
    for index in range(3):
        shuffled = legal.copy()
        rng.shuffle(shuffled)
        orders.append((f"shuffle_{index + 1}", shuffled))
    rows = []
    for order_name, order in orders:
        probabilities = policy.probabilities(scenario.state, order)
        ranking = sorted(probabilities, key=probabilities.get, reverse=True)
        for action, probability in probabilities.items():
            rows.append({"domain": scenario.domain, "scenario_id": scenario.scenario_id, "order": order_name, "action": action, "presentation_position": order.index(action), "probability": probability, "rank": ranking.index(action) + 1})
    return rows


def _aggregate(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in raw:
        key = (row["domain"], row["guidance"], row["alpha"], row["budget"])
        groups.setdefault(key, []).append(row)
    result = []
    for (domain, guidance, alpha, budget), rows in groups.items():
        objectives = [float(row["objective"]) for row in rows]
        regrets = [float(row["regret"]) for row in rows if row["regret"] is not None]
        result.append({
            "environment": domain,
            "guidance": guidance,
            "alpha": alpha,
            "budget": budget,
            "mean_objective": statistics.fmean(objectives),
            "objective_stddev": statistics.stdev(objectives) if len(objectives) > 1 else 0.0,
            "mean_regret": statistics.fmean(regrets) if regrets else None,
            "mean_normalized_regret": statistics.fmean(float(row["normalized_regret"]) for row in rows if row["normalized_regret"] is not None) if regrets else None,
            "regret_stddev": statistics.stdev(regrets) if len(regrets) > 1 else 0.0 if regrets else None,
            "trials": len(rows),
            "mean_laya_calls": statistics.fmean(float(row["laya_calls"]) for row in rows),
            "mean_laya_requests": statistics.fmean(float(row["laya_requests"]) for row in rows),
            "mean_cache_hits": statistics.fmean(float(row["cache_hits"]) for row in rows),
            "mean_cache_misses": statistics.fmean(float(row["cache_misses"]) for row in rows),
            "mean_latency_seconds": statistics.fmean(float(row["latency_seconds"]) for row in rows),
            "mean_simulations_per_decision": statistics.fmean(float(row["simulations_per_decision"]) for row in rows),
        })
    return sorted(result, key=lambda row: (row["environment"], row["guidance"], row["alpha"], row["budget"]))


def _mock_alpha(seed: int) -> list[dict[str, Any]]:
    env = ConnectFour()
    state = _play(env, [0, 1, 0, 1, 0, 2])
    rows = []
    for quality, policy in (("good", FixedPriorPolicy(0, 0.9)), ("bad", FixedPriorPolicy(6, 0.9))):
        for alpha in ALPHAS:
            for budget in (1, 2, 5, 10, 25, 50):
                successes = 0
                for trial in range(20):
                    result = MCTS(env, MixedPolicy(policy, alpha), MCTSConfig(simulations=budget, seed=seed + trial)).search(state)
                    successes += result.action == 0
                rows.append({"prior_quality": quality, "alpha": alpha, "budget": budget, "trials": 20, "success_rate": successes / 20})
    return rows


def _adaptive_ablation(scenarios: list[Scenario], predictor: Any, mode: str, seed: int, order_samples: int) -> list[dict[str, Any]]:
    if predictor is None:
        return []
    rows = []
    per_domain = 1 if mode == "quick" else 2
    selected: list[Scenario] = []
    for domain in {scenario.domain for scenario in scenarios}:
        selected.extend([scenario for scenario in scenarios if scenario.domain == domain][:per_domain])
    minimum, maximum = ((5, 10) if mode == "quick" else (10, 25) if mode == "standard" else (10, 50))
    trials = 1 if mode == "quick" else 3
    for scenario in selected:
        if scenario.known_actions is not None:
            optimal_actions, optimal_value = scenario.known_actions, scenario.known_value
        elif scenario.stochastic:
            optimal_actions, optimal_value = None, None
        else:
            optimal_actions, optimal_value, _ = optimal_action_analysis(scenario.environment, scenario.state)
        paired: dict[int, dict[bool, dict[str, Any]]] = {}
        for trial in range(trials):
            paired[trial] = {}
            for check in (False, True):
                policy = _policy(scenario, predictor)
                config = AdaptiveSearchConfig(minimum, maximum, order_stability_check=check, order_samples=order_samples, seed=seed + trial)
                agent = AdaptiveLayaMCTSAgent(scenario.environment, policy, config)
                before = policy.diagnostics()
                started = time.perf_counter()
                if scenario.domain == "connect_four":
                    decision = agent.decide(scenario.state)
                    objective, actions = float(optimal_actions is not None and decision.action in optimal_actions), [decision.action]
                    metadata, total_simulations = decision.metadata, decision.simulations
                else:
                    objective, actions, metadata, total_simulations = _adaptive_episode(scenario, agent)
                elapsed = time.perf_counter() - started
                after = policy.diagnostics()
                row = {
                    "domain": scenario.domain,
                    "scenario_id": scenario.scenario_id,
                    "trial": trial,
                    "order_stability_check": check,
                    "objective": objective,
                    "optimal_objective": optimal_value,
                    "regret": regret(optimal_value, objective),
                    "first_action": actions[0],
                    "first_action_optimal": actions[0] in optimal_actions if optimal_actions is not None else None,
                    "simulations": total_simulations,
                    "alpha": metadata.get("alpha"),
                    "stability_changed_alpha": metadata.get("stability_changed_alpha", False),
                    "triggered_additional_search": metadata.get("triggered_additional_search", False),
                    "laya_requests": _stats_delta(after, before, "total_policy_requests"),
                    "laya_calls": _stats_delta(after, before, "actual_inference_calls"),
                    "cache_hits": _stats_delta(after, before, "cache_hits"),
                    "latency_seconds": elapsed,
                    "decision_trace": metadata,
                }
                paired[trial][check] = row
        for _trial, pair in paired.items():
            without, with_check = pair[False], pair[True]
            with_check["prevented_bad_laya_action"] = bool(with_check["first_action_optimal"] and not without["first_action_optimal"])
            with_check["unnecessarily_increased_compute"] = bool(
                with_check["simulations"] is not None
                and without["simulations"] is not None
                and with_check["simulations"] > without["simulations"]
                and with_check["objective"] <= without["objective"]
            )
            without["prevented_bad_laya_action"] = False
            without["unnecessarily_increased_compute"] = False
            rows.extend((without, with_check))
    return rows


def _pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row["mean_regret"] is not None]
    result = []
    for row in candidates:
        dominated = any(
            other["environment"] == row["environment"]
            and other["guidance"] == row["guidance"]
            and other["mean_regret"] <= row["mean_regret"]
            and other["mean_latency_seconds"] <= row["mean_latency_seconds"]
            and (other["mean_regret"] < row["mean_regret"] or other["mean_latency_seconds"] < row["mean_latency_seconds"])
            for other in candidates
        )
        if not dominated:
            result.append(row)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def _report(results: dict[str, Any]) -> str:
    sweep, order = results["trust_sweep"], results["option_order"]
    root = [row for row in sweep if row["guidance"] == "root"]
    partial_wins = sorted({row["environment"] for row in root if row["alpha"] in (0.25, 0.5, 0.75) and row["mean_regret"] is not None and row["mean_regret"] < min((other["mean_regret"] for other in root if other["environment"] == row["environment"] and other["alpha"] == 1 and other["budget"] == row["budget"] and other["mean_regret"] is not None), default=float("inf"))})
    lines = [
        "# DDM-MCTS Multi-Domain Experiment Report", "", "# V2 — How Much Should Search Trust the DDM?", "",
        "`alpha` mixes the DDM prior with a uniform legal-action prior. `alpha=0` ignores Laya and `alpha=1` uses its full distribution. This is separate from `c_puct`, which controls PUCT exploration strength.", "",
        "## Evaluation correction", "", "Deterministic scenarios now retain every tied optimal first action. Regret is `optimal objective - achieved objective` for the project's higher-is-better metrics. Seeded inventory results are repeated but deliberately do not claim an exact stochastic optimum.", "",
        "## Trust sweep", "", "Model calls are incremental within the shared-cache sweep. Requests show how often a configuration consulted the policy, including cache hits.", "", "| Environment | Guidance | Alpha | Budget | Objective mean ± sd | Regret mean | Trials | Laya requests | Incremental calls | Latency (s) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sweep:
        regret_text = "n/a" if row["mean_regret"] is None else f"{row['mean_regret']:.3f}"
        lines.append(f"| {row['environment']} | {row['guidance']} | {row['alpha']:.2f} | {row['budget']} | {row['mean_objective']:.3f} ± {row['objective_stddev']:.3f} | {regret_text} | {row['trials']} | {row['mean_laya_requests']:.2f} | {row['mean_laya_calls']:.2f} | {row['mean_latency_seconds']:.4f} |")
    lines += ["", "## Root-Only vs Full-Tree Guidance", "", "This table uses only scenario IDs included in the deliberately smaller full-tree subset.", "", "| Environment | Alpha | Budget | Root objective | Full objective | Root requests | Full requests | Root latency | Full latency |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    raw = results["raw_trials"]
    full_keys = sorted({(row["domain"], row["scenario_id"], row["alpha"], row["budget"]) for row in raw if row["guidance"] == "full"})
    paired_groups: dict[tuple[str, float, int], dict[str, list[dict[str, Any]]]] = {}
    for domain, scenario_id, alpha, budget in full_keys:
        group = paired_groups.setdefault((domain, alpha, budget), {"root": [], "full": []})
        for row in raw:
            if row["domain"] == domain and row["scenario_id"] == scenario_id and row["alpha"] == alpha and row["budget"] == budget:
                group[row["guidance"]].append(row)
    for (domain, alpha, budget), group in sorted(paired_groups.items()):
        if not group["root"] or not group["full"]:
            continue
        def group_mean(rows: list[dict[str, Any]], key: str) -> float:
            return statistics.fmean(float(row[key]) for row in rows)
        lines.append(f"| {domain} | {alpha:.2f} | {budget} | {group_mean(group['root'], 'objective'):.3f} | {group_mean(group['full'], 'objective'):.3f} | {group_mean(group['root'], 'laya_requests'):.2f} | {group_mean(group['full'], 'laya_requests'):.2f} | {group_mean(group['root'], 'latency_seconds'):.4f}s | {group_mean(group['full'], 'latency_seconds'):.4f}s |")
    lines += ["", "## Option-Order Sensitivity", ""]
    if order.get("status") == "unavailable":
        lines.append("Unavailable because Laya did not run. No zero-valued sensitivity result is inferred.")
    else:
        lines += [f"Top-action consistency: **{order['top_action_consistency']:.1%}**. Mean absolute probability change: **{order['mean_absolute_probability_change']:.3f}**. Mean rank correlation: **{order['mean_rank_correlation']:.3f}**.", "", f"Mean probability by presentation position: `{json.dumps(order['mean_probability_by_position'])}`.", ""]
        lines.append("Option-order sensitivity was observed under the configured threshold." if order.get("sensitivity_observed") else "No strong option-order sensitivity was established by this run; more samples may still be useful.")
    lines += ["", "## Adaptive Order-Stability Ablation", ""]
    adaptive = results.get("adaptive_ablation", [])
    if not adaptive:
        lines.append("Unavailable because Laya/order checking was disabled or unavailable.")
    else:
        lines += ["| Controller | Objective | Regret | Simulations | Laya requests | Laya calls | Latency |", "|---|---:|---:|---:|---:|---:|---:|"]
        for enabled, label in ((False, "Without stability check"), (True, "With stability check")):
            selected = [row for row in adaptive if row["order_stability_check"] is enabled]
            regrets = [float(row["regret"]) for row in selected if row["regret"] is not None]
            lines.append(f"| {label} | {statistics.fmean(float(row['objective']) for row in selected):.3f} | {statistics.fmean(regrets) if regrets else float('nan'):.3f} | {statistics.fmean(float(row['simulations']) for row in selected):.2f} | {statistics.fmean(float(row['laya_requests']) for row in selected):.2f} | {statistics.fmean(float(row['laya_calls']) for row in selected):.2f} | {statistics.fmean(float(row['latency_seconds']) for row in selected):.4f}s |")
        checked = [row for row in adaptive if row["order_stability_check"]]
        unchecked = [row for row in adaptive if not row["order_stability_check"]]
        extra_requests = statistics.fmean(float(row["laya_requests"]) for row in checked) - statistics.fmean(float(row["laya_requests"]) for row in unchecked)
        extra_calls = statistics.fmean(float(row["laya_calls"]) for row in checked) - statistics.fmean(float(row["laya_calls"]) for row in unchecked)
        lines += ["", f"Stability changed alpha in **{sum(row['stability_changed_alpha'] for row in checked) / len(checked):.1%}** of checked trials and triggered additional search in **{sum(row['triggered_additional_search'] for row in checked) / len(checked):.1%}**. Average stability-check overhead was **{extra_requests:.2f} policy requests** and **{extra_calls:.2f} actual inference calls** per trial. It prevented a bad first action in **{sum(row['prevented_bad_laya_action'] for row in checked)}** cases and unnecessarily increased compute under the configured definition in **{sum(row['unnecessarily_increased_compute'] for row in checked)}** cases."]
        example = next((row for row in checked if row["decision_trace"].get("order_comparisons")), None)
        if example:
            lines += ["", "### Example adaptive decision trace", "", "```json", json.dumps(example["decision_trace"], indent=2, default=str), "```"]
    mock = results["mock_prior_alpha_sweep"]
    bad_low = max((row for row in mock if row["prior_quality"] == "bad" and row["budget"] == 1), key=lambda row: row["success_rate"])
    bad_high = max((row for row in mock if row["prior_quality"] == "bad" and row["budget"] == 50), key=lambda row: row["success_rate"])
    good_low = max((row for row in mock if row["prior_quality"] == "good" and row["budget"] == 1), key=lambda row: row["success_rate"])
    lines += ["", "## Search Mechanism Validation", "", f"With the bad mock prior at budget 1, the best measured alpha was **{bad_low['alpha']:.2f}** ({bad_low['success_rate']:.1%} success); at budget 50 it was **{bad_high['alpha']:.2f}** ({bad_high['success_rate']:.1%}). With the good prior at budget 1, the best measured alpha was **{good_low['alpha']:.2f}** ({good_low['success_rate']:.1%}). These are measured ties/bests, not assumed monotonic effects.", "", "## Pareto-Style Quality/Cost Tradeoffs", "", "These are non-dominated only among tested configurations and within the same guidance mode, not universal optima. Latency is the observed shared-cache sweep latency.", "", "| Environment | Guidance | Alpha | Budget | Regret | Latency |", "|---|---|---:|---:|---:|---:|"]
    for row in results["pareto"]:
        lines.append(f"| {row['environment']} | {row['guidance']} | {row['alpha']:.2f} | {row['budget']} | {row['mean_regret']:.3f} | {row['mean_latency_seconds']:.4f}s |")
    lines += ["", "## V2 Questions", "", f"1. Partial trust beat full trust at matched budgets in: **{', '.join(partial_wins) or 'none measured'}**.", "2. Low-budget effects can be compared directly in `trust_sweep.csv`; this report does not infer a universal pattern from a small run.", "3. Higher budgets reduce prior dependence when alpha curves converge; inspect per-environment rows.", "4. Bad-prior/alpha interaction is measured in the mock sweep, not assumed.", "5. Good-prior/alpha interaction is measured in the same controlled sweep.", "6. Root-only sufficiency is evaluated against the smaller full-tree subset.", "7. Full-tree value must be weighed against its measured Laya calls and latency.", "8. Option-order sensitivity is summarized above without automatically labeling it bias.", "9. The Pareto table lists useful measured quality/cost tradeoffs.", "", "## Warnings", ""]
    lines += [f"- {warning}" for warning in results["warnings"]]
    return "\n".join(lines) + "\n"


def _run_directory(root: Path) -> Path:
    base = f"v2_trust_{datetime.now():%Y-%m-%d_%H%M}"
    path, index = root / base, 2
    while path.exists():
        path = root / f"{base}_{index}"
        index += 1
    return path


def run_v2_experiment(*, mode: str = "quick", budgets: list[int] | None = None, seed: int = 0, results_root: str | Path = "results", laya_device: str | None = None, laya_model: str | None = None, skip_laya: bool = False, order_stability_check: bool = True, order_samples: int = 2) -> Path:
    defaults = {"quick": [1, 5], "standard": [1, 2, 5, 10, 25, 50], "extended": [1, 2, 5, 10, 25, 50, 100, 250, 500]}
    budgets = budgets or defaults[mode]
    trials = 2 if mode == "quick" else 3 if mode == "standard" else 5
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    output = _run_directory(root)
    output.mkdir()
    handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    logging.getLogger().addHandler(handler)
    logging.captureWarnings(True)
    log_lines = [f"V2 run started {datetime.now().isoformat()}", f"mode={mode} budgets={budgets} trials={trials} seed={seed}"]
    checks = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True, check=False, timeout=180)
    log_lines.append(f"tests_return_code={checks.returncode} {checks.stdout.strip()} {checks.stderr.strip()}")
    print("DDM-MCTS V2 Trust Sweep\n", flush=True)
    print(f"[1/8] System checks ............... {'PASS' if checks.returncode == 0 else 'FAIL'}", flush=True)
    predictor, laya_error = (None, "disabled by --skip-laya") if skip_laya else _backend(laya_device, laya_model)
    scenarios = _all_scenarios(mode, seed)
    raw, order_rows = [], []
    by_domain: dict[str, list[Scenario]] = {}
    for scenario in scenarios:
        by_domain.setdefault(scenario.domain, []).append(scenario)
    domain_order = ("connect_four", "grid_navigation", "job_scheduling", "delivery_routing", "inventory", "delayed_reward")
    for domain_index, domain in enumerate(domain_order, start=2):
        domain_raw = []
        for scenario_index, scenario in enumerate(by_domain[domain]):
            if scenario.known_actions is not None:
                optimal_actions, optimal_value = scenario.known_actions, scenario.known_value
            elif scenario.stochastic:
                optimal_actions, optimal_value = None, None
            else:
                optimal_actions, optimal_value, _ = optimal_action_analysis(scenario.environment, scenario.state)
            policy = _policy(scenario, predictor)
            if predictor is not None and scenario_index < (1 if mode == "quick" else 2):
                try:
                    order_rows.extend(_order_rows(scenario, policy, seed + scenario_index))
                except LayaUnavailableError as exc:
                    laya_error, predictor = str(exc), None
            for alpha in ALPHAS:
                if alpha > 0 and predictor is None:
                    continue
                for budget in budgets:
                    for trial in range(trials):
                        row = _measure(scenario, policy, alpha, budget, seed + trial, "root", optimal_actions, optimal_value)
                        raw.append(row)
                        domain_raw.append(row)
            # Full-tree Laya is deliberately a small subset.
            if scenario_index < (1 if mode == "quick" else 2):
                full_alphas = (0.0, 1.0) if mode == "quick" else (0.0, 0.5, 1.0)
                full_budgets = sorted({budgets[0], budgets[-1]})
                for alpha in full_alphas:
                    if alpha > 0 and predictor is None:
                        continue
                    for budget in full_budgets:
                        row = _measure(scenario, policy, alpha, budget, seed, "full", optimal_actions, optimal_value)
                        raw.append(row)
                        domain_raw.append(row)
        folder = output / domain
        folder.mkdir()
        (folder / "results.json").write_text(json.dumps(domain_raw, indent=2, default=str), encoding="utf-8")
        _write_csv(folder / "decisions.csv", domain_raw)
        domain_orders = [row for row in order_rows if row["domain"] == domain]
        _write_csv(folder / "policy_analysis.csv", domain_orders)
        log_lines.append(f"domain={domain} raw_trials={len(domain_raw)} option_order_rows={len(domain_orders)}")
        print(f"[{domain_index}/8] {by_domain[domain][0].title:<27} DONE", flush=True)
    sweep = _aggregate(raw)
    mock = _mock_alpha(seed)
    adaptive = _adaptive_ablation(scenarios, predictor, mode, seed, order_samples) if order_stability_check else []
    order_summary = option_order_metrics(order_rows)
    warnings = ["Results are best among tested configurations only; scenario and trial counts remain limited.", "Inventory uses repeated reproducible demand seeds and reports no exact optimal action or regret."]
    if laya_error:
        warnings.append(f"Laya unavailable or disabled: {laya_error}")
    results = {"configuration": {"mode": mode, "budgets": budgets, "alphas": ALPHAS, "trials": trials, "seed": seed, "order_stability_check": order_stability_check, "order_samples": order_samples}, "system_checks": {"return_code": checks.returncode, "output": (checks.stdout + checks.stderr).strip()}, "trust_sweep": sweep, "raw_trials": raw, "mock_prior_alpha_sweep": mock, "option_order": order_summary, "option_order_raw": order_rows, "adaptive_ablation": adaptive, "pareto": _pareto(sweep), "warnings": warnings}
    (output / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    _write_csv(output / "trust_sweep.csv", sweep)
    _write_csv(output / "adaptive_ablation.csv", [{key: json.dumps(value, default=str) if isinstance(value, (dict, list)) else value for key, value in row.items()} for row in adaptive])
    (output / "REPORT.md").write_text(_report(results), encoding="utf-8")
    print("[8/8] Building V2 report .......... DONE", flush=True)
    logging.getLogger().removeHandler(handler)
    logging.captureWarnings(False)
    handler.close()
    with (output / "run.log").open("a", encoding="utf-8") as log_handle:
        log_handle.write("\n".join(log_lines) + "\n")
    latest, staging = root / "latest", root / ".latest_staging"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(output, staging)
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(staging, latest)
    shutil.rmtree(staging)
    print("\nExperiment complete.\n\nOpen:\n\nresults/latest/REPORT.md", flush=True)
    return output

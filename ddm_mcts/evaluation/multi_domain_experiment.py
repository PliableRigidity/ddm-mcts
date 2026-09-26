from __future__ import annotations

import csv
import json
import logging
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any, Generic, TypeVar

from ddm_mcts.agents import LayaAgent, LayaMCTSAgent, LayaRootMCTSAgent, MCTSAgent
from ddm_mcts.environments import ConnectFour, DelayedReward, DeliveryRouting, GridNavigation, InventoryManagement, JobScheduling
from ddm_mcts.environments.base import Environment
from ddm_mcts.environments.delivery_routing import routing_scenarios
from ddm_mcts.environments.grid_navigation import grid_scenarios
from ddm_mcts.environments.job_scheduling import scheduling_scenarios
from ddm_mcts.policies import LayaPolicy, LayaUnavailableError, TextLayaPolicy

from .experiment import _mock_prior_curves, _play

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DomainSpec(Generic[StateT, ActionT]):
    key: str
    title: str
    purpose: str
    objective: str
    metric: str
    environment: Environment[StateT, ActionT]
    scenarios: list[tuple[str, StateT]]


def _domain_specs(mode: str, seed: int) -> list[DomainSpec[Any, Any]]:
    grid_items = list(grid_scenarios().items())
    schedule_items = list(scheduling_scenarios().items())
    route_items = list(routing_scenarios().items())
    if mode == "quick":
        grid_items, schedule_items, route_items = grid_items[:1], schedule_items[:1], route_items[:1]
    grid_env = GridNavigation(grid_items[0][1])
    scheduling_env = JobScheduling(schedule_items[0][1])
    routing_env = DeliveryRouting(route_items[0][1])
    inventory_scenarios = [(f"seed_{seed + i}", InventoryManagement(seed + i).initial_state()) for i in range(1 if mode == "quick" else 3)]
    return [
        DomainSpec("grid_navigation", "Grid Navigation", "Multi-step spatial planning around obstacles.", "Choose movements that reach the goal in as few steps as possible.", "terminal reward (success minus step cost)", grid_env, grid_items),
        DomainSpec("job_scheduling", "Job Scheduling", "Resource allocation and deadline trade-offs.", "Choose the next job to maximize value after lateness penalties.", "total schedule reward", scheduling_env, [(name, JobScheduling(jobs).initial_state()) for name, jobs in schedule_items]),
        DomainSpec("delivery_routing", "Delivery Routing", "Global route optimization rather than nearest-neighbor greediness.", "Visit every destination and return to the depot with minimum cost.", "negative travel cost (higher is better)", routing_env, [(name, DeliveryRouting(distances).initial_state()) for name, distances in route_items]),
        DomainSpec("inventory", "Inventory Management", "Seeded decision making under uncertain demand.", "Choose order quantities that minimize order, holding, and stockout costs.", "negative total cost (higher is better)", InventoryManagement(seed), inventory_scenarios),
        DomainSpec("delayed_reward", "Delayed Reward", "Deliberation when immediate-looking rewards are deceptive.", "Choose a branch that maximizes eventual cumulative reward.", "final cumulative reward", DelayedReward(), [("deceptive_immediate_reward", DelayedReward().initial_state())]),
    ]


def _scenario_environment(spec: DomainSpec[Any, Any], scenario_name: str, state: Any) -> Environment[Any, Any]:
    if spec.key == "grid_navigation":
        return GridNavigation(state)
    if spec.key == "job_scheduling":
        return JobScheduling(state.remaining)
    if spec.key == "delivery_routing":
        return DeliveryRouting(routing_scenarios()[scenario_name])
    if spec.key == "inventory":
        return InventoryManagement(state.seed, state.horizon)
    return spec.environment


def _optimal(environment: Environment[Any, Any], state: Any) -> tuple[Any, float]:
    @cache
    def value(current: Any) -> float:
        if environment.is_terminal(current):
            return environment.get_reward(current, 1)
        return max(value(environment.step(current, action)) for action in environment.legal_actions(current))

    choices = [(action, value(environment.step(state, action))) for action in environment.legal_actions(state)]
    return max(choices, key=lambda item: item[1])


def _policy_stats(policy: Any) -> dict[str, int | float]:
    return policy.diagnostics() if hasattr(policy, "diagnostics") else {}


def _delta(after: dict[str, int | float], before: dict[str, int | float], key: str) -> int | float:
    return after.get(key, 0) - before.get(key, 0)


def _episode(environment: Environment[Any, Any], agent: Any, state: Any) -> tuple[float, list[Any], list[dict[str, Any]]]:
    actions, details = [], []
    while not environment.is_terminal(state):
        decision = agent.decide(state)
        actions.append(decision.action)
        details.append({"state": environment.render(state), "action": decision.action, "search": decision.statistics})
        state = environment.step(state, decision.action)
    return environment.get_reward(state, 1), actions, details


def _text_policy(environment: Environment[Any, Any], objective: str, device: str | None, model: str | None, predictor: Any = None) -> TextLayaPolicy[Any, Any]:
    return TextLayaPolicy(environment.render, objective=objective, action_text=lambda action: f"Choose action {action}", device=device, model=model, predictor=predictor)


def _shared_predictor(policy: Any, model: str | None):
    if model and policy._router is not None:
        def predict(state: Any, questions: dict[str, Any]) -> dict[str, Any]:
            return policy._router.predict(state, questions, model=model)
        return predict
    return policy._predictor


def _evaluate_domain(spec: DomainSpec[Any, Any], budgets: list[int], seed: int, device: str | None, model: str | None, skip_laya: bool) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    laya_error: str | None = "disabled by --skip-laya" if skip_laya else None
    for scenario_index, (scenario_name, initial_state) in enumerate(spec.scenarios):
        environment = _scenario_environment(spec, scenario_name, initial_state)
        optimal_action, optimal_value = _optimal(environment, initial_state)
        policy = _text_policy(environment, spec.objective, device, model)
        for budget in budgets:
            agents: list[tuple[str, Any, Any]] = [("mcts", MCTSAgent(environment, budget, seed + scenario_index), policy)]
            if laya_error is None:
                try:
                    # Probe once. This also makes option-order evidence available and gives all agents the same cached root evaluation.
                    probabilities = policy.probabilities(initial_state, environment.legal_actions(initial_state))
                    for action, probability in probabilities.items():
                        policy_rows.append({"scenario": scenario_name, "budget": budget, "order": "normal", "action": action, "presentation_position": list(environment.legal_actions(initial_state)).index(action), "probability": probability, "rank": sorted(probabilities, key=probabilities.get, reverse=True).index(action) + 1})
                    if budget == budgets[0]:
                        rng = random.Random(seed + scenario_index)
                        for shuffle_index in range(3):
                            ordered = list(environment.legal_actions(initial_state))
                            rng.shuffle(ordered)
                            shuffled = policy.probabilities(initial_state, ordered)
                            for action, probability in shuffled.items():
                                policy_rows.append({"scenario": scenario_name, "budget": budget, "order": f"shuffle_{shuffle_index + 1}", "action": action, "presentation_position": ordered.index(action), "probability": probability, "rank": sorted(shuffled, key=shuffled.get, reverse=True).index(action) + 1})
                    backend = _shared_predictor(policy, model)
                    direct_policy = _text_policy(environment, spec.objective, device, None, backend)
                    root_policy = _text_policy(environment, spec.objective, device, None, backend)
                    full_policy = _text_policy(environment, spec.objective, device, None, backend)
                    agents += [
                        ("laya", LayaAgent(environment, direct_policy, seed), direct_policy),
                        ("laya_root_mcts", LayaRootMCTSAgent(environment, root_policy, budget, seed), root_policy),
                        ("laya_full_mcts", LayaMCTSAgent(environment, full_policy, budget, seed), full_policy),
                    ]
                except LayaUnavailableError as exc:
                    laya_error = str(exc)
            for agent_name, agent, tracked_policy in agents:
                before = _policy_stats(tracked_policy)
                started = time.perf_counter()
                reward, actions, details = _episode(environment, agent, initial_state)
                elapsed = time.perf_counter() - started
                after = _policy_stats(tracked_policy)
                decisions.append({
                    "environment": spec.key,
                    "scenario": scenario_name,
                    "budget": budget,
                    "agent": agent_name,
                    "objective_value": reward,
                    "optimal_value": optimal_value,
                    "first_action": actions[0],
                    "optimal_first_action": optimal_action,
                    "optimal_first_action_correct": actions[0] == optimal_action,
                    "actions": actions,
                    "decision_latency_seconds": elapsed,
                    "laya_policy_requests": _delta(after, before, "total_policy_requests"),
                    "laya_inference_calls": _delta(after, before, "actual_inference_calls"),
                    "laya_cache_hits": _delta(after, before, "cache_hits"),
                    "laya_inference_seconds": _delta(after, before, "total_inference_seconds"),
                    "trace": details,
                })
    return {"status": "complete", "title": spec.title, "purpose": spec.purpose, "objective": spec.objective, "metric": spec.metric, "laya_error": laya_error, "decisions": decisions, "policy_analysis": policy_rows}


def _connect_four(budgets: list[int], seed: int, device: str | None, model: str | None, skip_laya: bool) -> dict[str, Any]:
    environment = ConnectFour()
    positions = [("immediate_win", _play(environment, [0, 1, 0, 1, 0, 2]), 0), ("immediate_block", _play(environment, [0, 3, 1, 3, 2, 3]), 3), ("vertical_win", _play(environment, [4, 0, 4, 1, 4, 2]), 4)]
    policy = LayaPolicy(device=device, model=model)
    laya_error = "disabled by --skip-laya" if skip_laya else None
    rows, policy_rows = [], []
    for position_name, state, correct in positions:
        for budget in budgets:
            agents: list[tuple[str, Any, Any]] = [("mcts", MCTSAgent(environment, budget, seed), policy)]
            if laya_error is None:
                try:
                    legal = environment.legal_actions(state)
                    policy.probabilities(state, legal)
                    if budget == budgets[0]:
                        orders = [("normal", list(legal))]
                        rng = random.Random(seed)
                        for index in range(3):
                            order = list(legal)
                            rng.shuffle(order)
                            orders.append((f"shuffle_{index + 1}", order))
                        for order_name, order in orders:
                            measured = policy.probabilities(state, order)
                            for action, probability in measured.items():
                                policy_rows.append({"scenario": position_name, "order": order_name, "action": action, "presentation_position": order.index(action), "probability": probability, "rank": sorted(measured, key=measured.get, reverse=True).index(action) + 1})
                    backend = _shared_predictor(policy, model)
                    direct_policy = LayaPolicy(predictor=backend)
                    root_policy = LayaPolicy(predictor=backend)
                    full_policy = LayaPolicy(predictor=backend)
                    agents += [("laya", LayaAgent(environment, direct_policy, seed), direct_policy), ("laya_root_mcts", LayaRootMCTSAgent(environment, root_policy, budget, seed), root_policy), ("laya_full_mcts", LayaMCTSAgent(environment, full_policy, budget, seed), full_policy)]
                except LayaUnavailableError as exc:
                    laya_error = str(exc)
            for name, agent, tracked_policy in agents:
                before = _policy_stats(tracked_policy)
                started = time.perf_counter()
                decision = agent.decide(state)
                elapsed = time.perf_counter() - started
                after = _policy_stats(tracked_policy)
                rows.append({"environment": "connect_four", "scenario": position_name, "budget": budget, "agent": name, "objective_value": int(decision.action == correct), "metric": "tactical accuracy", "first_action": decision.action, "optimal_first_action": correct, "optimal_first_action_correct": decision.action == correct, "decision_latency_seconds": elapsed, "laya_policy_requests": _delta(after, before, "total_policy_requests"), "laya_inference_calls": _delta(after, before, "actual_inference_calls"), "laya_cache_hits": _delta(after, before, "cache_hits"), "laya_inference_seconds": _delta(after, before, "total_inference_seconds"), "search": decision.statistics})
    return {"status": "complete", "title": "Connect Four", "purpose": "Adversarial tactical decisions and threat response.", "objective": "Choose the known winning or blocking move.", "metric": "tactical accuracy", "laya_error": laya_error, "decisions": rows, "policy_analysis": policy_rows}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    simple = [{key: json.dumps(value, default=str) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()} for row in rows]
    fields = list(dict.fromkeys(key for row in simple for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(simple)


def _mean(rows: list[dict[str, Any]], agent: str, key: str) -> float | None:
    values = [float(row[key]) for row in rows if row["agent"] == agent]
    return sum(values) / len(values) if values else None


def _report(results: dict[str, Any]) -> str:
    domains = results["domains"]
    agents = ("laya", "mcts", "laya_root_mcts", "laya_full_mcts")
    labels = {"laya": "Laya", "mcts": "MCTS", "laya_root_mcts": "Laya Root", "laya_full_mcts": "Laya Full"}
    lines = ["# DDM-MCTS Multi-Domain Experiment Report", "", "## What was tested", "", "- **Laya:** chooses directly from its action probabilities.", "- **MCTS:** uniform-prior PUCT with explicit environment simulation.", "- **Laya Root + MCTS:** Laya supplies priors once at each real decision; descendant nodes are uniform.", "- **Laya Full + MCTS:** Laya supplies priors throughout the expanded tree.", "", "Results are preliminary diagnostics, not evidence of general or statistically significant superiority.", "", "## Overall Results", "", "Each cell shows the environment's stated objective metric; all stored objective values are oriented so higher is better.", "", "| Environment and metric | Laya | MCTS | Laya Root | Laya Full |", "|---|---:|---:|---:|---:|"]
    for _key, domain in domains.items():
        cells = []
        for agent in agents:
            value = _mean(domain.get("decisions", []), agent, "objective_value")
            cells.append("unavailable" if value is None else f"{value:.3f}")
        lines.append(f"| {domain['title']} — {domain['metric']} | " + " | ".join(cells) + " |")
    for _key, domain in domains.items():
        if domain.get("status") == "failed":
            lines += ["", f"## {domain['title']}", "", f"**FAILED:** `{domain.get('error', 'unknown error')}`. Other environments continued."]
            continue
        lines += ["", f"## {domain['title']}", "", f"**Decision:** {domain['objective']}", "", f"**State and simulation:** {domain['purpose']} Laya receives the environment's concise text rendering and legal actions. MCTS uses the real `step` transition until a terminal objective is available.", "", f"**Metric:** {domain['metric']}.", ""]
        if domain.get("laya_error"):
            lines.append(f"Laya results unavailable: `{domain['laya_error']}`. Non-Laya results still ran.")
        lines += ["", "| Agent | Mean objective | Optimal first-action accuracy | Mean latency (s) |", "|---|---:|---:|---:|"]
        for agent in agents:
            rows = [row for row in domain.get("decisions", []) if row["agent"] == agent]
            if not rows:
                lines.append(f"| {labels[agent]} | unavailable | unavailable | unavailable |")
            else:
                accuracy = sum(row["optimal_first_action_correct"] for row in rows) / len(rows)
                lines.append(f"| {labels[agent]} | {_mean(rows, agent, 'objective_value'):.3f} | {accuracy:.1%} | {_mean(rows, agent, 'decision_latency_seconds'):.4f} |")
    lines += ["", "# Laya Root vs Laya Full", "", "| Environment | Root performance | Full performance | Root inference calls | Full inference calls | Root cache hits | Full cache hits | Root latency | Full latency |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for domain in domains.values():
        rows = domain.get("decisions", [])
        root_perf, full_perf = _mean(rows, "laya_root_mcts", "objective_value"), _mean(rows, "laya_full_mcts", "objective_value")
        if root_perf is None:
            lines.append(f"| {domain['title']} | unavailable | unavailable | — | — | — | — | — | — |")
        else:
            root_req, full_req = _mean(rows, "laya_root_mcts", "laya_inference_calls"), _mean(rows, "laya_full_mcts", "laya_inference_calls")
            root_hits, full_hits = _mean(rows, "laya_root_mcts", "laya_cache_hits"), _mean(rows, "laya_full_mcts", "laya_cache_hits")
            root_lat, full_lat = _mean(rows, "laya_root_mcts", "decision_latency_seconds"), _mean(rows, "laya_full_mcts", "decision_latency_seconds")
            lines.append(f"| {domain['title']} | {root_perf:.3f} | {full_perf:.3f} | {root_req:.1f} | {full_req:.1f} | {root_hits:.1f} | {full_hits:.1f} | {root_lat:.4f}s | {full_lat:.4f}s |")
    lines += ["", "# Cross-Domain Findings", ""]
    helped_root, helped_full, costly = [], [], []
    for domain in domains.values():
        rows = domain.get("decisions", [])
        uniform, root, full = (_mean(rows, name, "objective_value") for name in ("mcts", "laya_root_mcts", "laya_full_mcts"))
        if root is not None and uniform is not None and root > uniform:
            helped_root.append(domain["title"])
        if full is not None and uniform is not None and full > uniform:
            helped_full.append(domain["title"])
        if full is not None and _mean(rows, "laya_full_mcts", "decision_latency_seconds") > 2 * _mean(rows, "mcts", "decision_latency_seconds"):
            costly.append(domain["title"])
    lines += [f"- Root-only guidance exceeded uniform MCTS in this run: {', '.join(helped_root) or 'none measured'}.", f"- Full-tree guidance exceeded uniform MCTS in this run: {', '.join(helped_full) or 'none measured'}.", f"- Full-tree latency exceeded twice uniform MCTS latency in: {', '.join(costly) or 'none measured'}.", "- These observations combine small deterministic scenarios and seeded inventory trials. More seeds and episodes are needed before generalizing.", "", "## Option-Order Sensitivity", "", "The per-domain `policy_analysis.csv` files contain action, presentation position, probability, and rank for normal and shuffled orders. This report records sensitivity evidence without labeling it bias; systematic effects require more samples.", "", "## Search Mechanism Validation", "", f"Bad-prior recovery and good-prior efficiency curves are stored in `results.json`. At the largest bad-prior budget, recovery rate was **{results['mechanism_validation']['bad_prior'][-1]['recovery_rate']:.1%}**.", "", "## Important Warnings", ""]
    lines += [f"- {warning}" for warning in results["warnings"]]
    lines += ["", "Open each environment folder for raw decisions and option-order measurements."]
    return "\n".join(lines) + "\n"


def _run_name(root: Path) -> Path:
    base = f"multi_domain_{datetime.now():%Y-%m-%d_%H%M}"
    candidate = root / base
    suffix = 2
    while candidate.exists():
        candidate = root / f"{base}_{suffix}"
        suffix += 1
    return candidate


def run_multi_domain_experiment(*, mode: str = "quick", games: int = 1, budgets: list[int] | None = None, seed: int = 0, results_root: str | Path = "results", laya_device: str | None = None, laya_model: str | None = None, skip_laya: bool = False) -> Path:
    del games  # Scenarios, rather than head-to-head games, are the common unit across these domains.
    defaults = {"quick": [10], "standard": [10, 25, 50, 100], "extended": [10, 25, 50, 100, 250]}
    budgets = budgets or defaults[mode]
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    output = _run_name(root)
    output.mkdir()
    handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    logging.getLogger().addHandler(handler)
    print("DDM-MCTS Multi-Domain Experiment\n", flush=True)
    checks = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True, check=False, timeout=180)
    print(f"[1/8] System checks ............... {'PASS' if checks.returncode == 0 else 'FAIL'}", flush=True)
    domains: dict[str, Any] = {}
    warnings = ["Small scenario counts make all cross-domain comparisons preliminary.", "Inventory demand is sampled reproducibly from each state seed and timestep; multiple seeds represent uncertainty."]
    tasks: list[tuple[str, Any]] = [("connect_four", None)] + [(spec.key, spec) for spec in _domain_specs(mode, seed)]
    titles = {"connect_four": "Connect Four", **{spec.key: spec.title for _, spec in tasks[1:]}}
    for index, (key, spec) in enumerate(tasks, start=2):
        try:
            result = _connect_four(budgets, seed, laya_device, laya_model, skip_laya) if key == "connect_four" else _evaluate_domain(spec, budgets, seed, laya_device, laya_model, skip_laya)
            domains[key] = result
            folder = output / key
            folder.mkdir()
            (folder / "results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            _write_csv(folder / "decisions.csv", result["decisions"])
            _write_csv(folder / "policy_analysis.csv", result["policy_analysis"])
            status = "DONE"
        except Exception as exc:
            LOGGER.exception("Environment %s failed", key)
            domains[key] = {"status": "failed", "title": titles[key], "metric": "unavailable", "error": str(exc), "decisions": [], "policy_analysis": []}
            warnings.append(f"{titles[key]} failed: {exc}")
            status = "FAILED"
        print(f"[{index}/8] {titles[key]:<27} {status}", flush=True)
    environment = ConnectFour()
    bad, good = _mock_prior_curves(environment, seed)
    results = {"configuration": {"mode": mode, "budgets": budgets, "seed": seed, "laya_device": laya_device or "auto", "laya_model": laya_model or "auto", "skip_laya": skip_laya}, "system_checks": {"return_code": checks.returncode, "output": (checks.stdout + checks.stderr).strip()}, "domains": domains, "mechanism_validation": {"bad_prior": bad, "good_prior": good}, "warnings": warnings}
    (output / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (output / "REPORT.md").write_text(_report(results), encoding="utf-8")
    print("[8/8] Building report ............ DONE", flush=True)
    handler.flush()
    logging.getLogger().removeHandler(handler)
    handler.close()
    latest = root / "latest"
    latest_staging = root / ".latest_staging"
    if latest_staging.exists():
        shutil.rmtree(latest_staging)
    shutil.copytree(output, latest_staging)
    if latest.exists():
        shutil.rmtree(latest)
    latest_staging.rename(latest)
    print("\nExperiment complete.\n\nOpen:\n\nresults/latest/REPORT.md", flush=True)
    return output

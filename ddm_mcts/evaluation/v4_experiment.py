from __future__ import annotations

import json
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ddm_mcts.agents import AdaptiveLayaMCTSAgent, AdaptiveSearchConfig, LayaAgent, LayaRootMCTSAgent
from ddm_mcts.policies import (
    MicaPolicy,
    MicaUnavailableError,
    MixedPolicy,
    PermutationAveragedPolicy,
    TextLayaPolicy,
    UniformPolicy,
)

from .v2_experiment import (
    Scenario,
    _all_scenarios,
    _backend,
    _episode,
    _order_rows,
    _write_csv,
    optimal_action_analysis,
    option_order_metrics,
    regret,
)

BACKENDS = ("laya", "mica")


def _run_directory(root: Path) -> Path:
    base = f"v4_multi_ddm_{datetime.now():%Y-%m-%d_%H%M}"
    candidate, suffix = root / base, 2
    while candidate.exists():
        candidate = root / f"{base}_{suffix}"
        suffix += 1
    return candidate


def _text_policy(
    backend: str,
    scenario: Scenario,
    *,
    predictor: Any = None,
    mica_endpoint: str,
    mica_model: str,
) -> Any:
    common = {
        "state_text": scenario.environment.render,
        "objective": scenario.objective,
        "action_text": lambda action: f"Choose {action}",
    }
    if backend == "laya":
        return TextLayaPolicy(**common, predictor=predictor)
    return MicaPolicy(**common, endpoint=mica_endpoint, model=mica_model)


def _optimum(scenario: Scenario) -> tuple[frozenset[Any] | None, float | None]:
    if scenario.known_actions is not None:
        return scenario.known_actions, scenario.known_value
    if scenario.stochastic:
        return None, None
    actions, value, _ = optimal_action_analysis(scenario.environment, scenario.state)
    return actions, value


def _evaluate(
    scenario: Scenario,
    policy: Any,
    *,
    backend: str,
    samples: int,
    alpha: float,
    budget: int,
    seed: int,
    direct: bool,
) -> dict[str, Any]:
    optimal_actions, optimal_value = _optimum(scenario)
    before = getattr(policy, "diagnostics", lambda: {})()
    agent = (
        LayaAgent(scenario.environment, MixedPolicy(policy, alpha), seed)
        if direct
        else LayaRootMCTSAgent(scenario.environment, policy, budget, seed, alpha=alpha)
    )
    started = time.perf_counter()
    if scenario.domain == "connect_four":
        decision = agent.decide(scenario.state)
        objective, actions, simulations = (
            float(optimal_actions is not None and decision.action in optimal_actions),
            [decision.action],
            decision.simulations,
        )
    else:
        objective, actions = _episode(scenario, agent)
        simulations = 0 if direct else budget * len(actions)
    latency = time.perf_counter() - started
    after = getattr(policy, "diagnostics", lambda: {})()
    ddm_latency = float(after.get("total_inference_seconds", 0)) - float(before.get("total_inference_seconds", 0))
    measured_regret = regret(optimal_value, objective)
    return {
        "backend": backend,
        "domain": scenario.domain,
        "scenario_id": scenario.scenario_id,
        "method": "direct" if direct else "root_mcts",
        "permutation_samples": samples,
        "alpha": alpha,
        "budget": 0 if direct else budget,
        "objective": objective,
        "optimal_objective": optimal_value,
        "regret": measured_regret,
        "first_action": actions[0],
        "ddm_action": actions[0] if direct else None,
        "final_mcts_action": None if direct else actions[0],
        "first_action_optimal": actions[0] in optimal_actions if optimal_actions is not None else None,
        "simulations": simulations,
        "backend_requests": float(after.get("total_policy_requests", 0)) - float(before.get("total_policy_requests", 0)),
        "backend_calls": float(after.get("actual_inference_calls", 0)) - float(before.get("actual_inference_calls", 0)),
        "cache_hits": float(after.get("cache_hits", 0)) - float(before.get("cache_hits", 0)),
        "ddm_latency_seconds": ddm_latency,
        "search_latency_seconds": max(0.0, latency - ddm_latency),
        "latency_seconds": latency,
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[name] for name in ("backend", "domain", "method", "permutation_samples", "alpha", "budget"))
        groups.setdefault(key, []).append(row)
    result = []
    for key, selected in groups.items():
        regrets = [float(row["regret"]) for row in selected if row["regret"] is not None]
        result.append({
            **dict(zip(("backend", "domain", "method", "permutation_samples", "alpha", "budget"), key, strict=True)),
            "mean_objective": statistics.fmean(float(row["objective"]) for row in selected),
            "mean_regret": statistics.fmean(regrets) if regrets else None,
            "mean_simulations": statistics.fmean(float(row["simulations"]) for row in selected),
            "mean_backend_requests": statistics.fmean(float(row["backend_requests"]) for row in selected),
            "mean_backend_calls": statistics.fmean(float(row["backend_calls"]) for row in selected),
            "mean_latency_seconds": statistics.fmean(float(row["latency_seconds"]) for row in selected),
            "trials": len(selected),
        })
    return sorted(result, key=lambda row: tuple(str(row[name]) for name in ("backend", "domain", "method", "permutation_samples", "alpha", "budget")))


def _averaging_summary(order_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for backend in BACKENDS:
        for samples in (1, 2, 3):
            selected = [row for row in order_rows if row["backend"] == backend and row["permutation_samples"] == samples]
            metrics = option_order_metrics(selected)
            output.append({"backend": backend, "permutation_samples": samples, **metrics})
    return output


def _report(results: dict[str, Any]) -> str:
    config, comparison, stability = results["configuration"], results["ddm_comparison"], results["permutation_averaging"]
    lines = [
        "# DDM-MCTS Experiment Report",
        "",
        "# V4 — DDM Comparison and Robust Priors",
        "",
        "Laya and Mica receive the same rendered state, objective, semantic actions, presentation orders, alpha values, budgets, scenarios, and seeds. Mica is queried only through its direct `choice` probability interface; no prose or chain-of-thought is requested.",
        "",
        f"Mode: **{config['mode']}**. Backends requested: **{', '.join(config['requested_backends'])}**. Budgets: `{config['budgets']}`. Conservative alphas: `{config['alphas']}`.",
        "",
        "### Backend status",
        "",
    ]
    for backend, metadata in results["backend_metadata"].items():
        lines.append(f"- **{backend}**: {metadata['status']} — {metadata.get('detail', '')}")
    lines += [
        "",
        "### Direct and root-search comparison",
        "",
        "| Backend | Domain | Method | K | Alpha | Budget | Objective | Regret | Calls | Simulations | Latency |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        regret_text = "n/a" if row["mean_regret"] is None else f"{row['mean_regret']:.3f}"
        lines.append(f"| {row['backend']} | {row['domain']} | {row['method']} | {row['permutation_samples']} | {row['alpha']:.2f} | {row['budget']} | {row['mean_objective']:.3f} | {regret_text} | {row['mean_backend_calls']:.2f} | {row['mean_simulations']:.1f} | {row['mean_latency_seconds']:.4f}s |")
    lines += ["", "### Option-order sensitivity and permutation averaging", "", "| Backend | K | Top-action consistency | Mean absolute probability change | Rank correlation |", "|---|---:|---:|---:|---:|"]
    for row in stability:
        if row.get("status") == "unavailable" or row.get("top_action_consistency") is None:
            lines.append(f"| {row['backend']} | {row['permutation_samples']} | unavailable | unavailable | unavailable |")
        else:
            lines.append(f"| {row['backend']} | {row['permutation_samples']} | {row['top_action_consistency']:.1%} | {row['mean_absolute_probability_change']:.3f} | {row['mean_rank_correlation']:.3f} |")
    lines += [
        "",
        "Permutation averaging is a stability intervention, not evidence of correctness. Every MCTS configuration still performs its mandatory search budget.",
        "",
        "### V4 questions",
        "",
        "1. Which DDM is stronger directly? Compare `method=direct` rows at K=1.",
        "2. Which is more order-sensitive? Compare K=1 stability metrics.",
        "3. Does K=2 or K=3 reduce sensitivity? Compare each backend across K.",
        "4. Does averaging improve objective or regret? Compare matched direct and root-MCTS rows.",
        "5. Does search correct model errors? Use `raw_trials.json` to compare direct and root first actions.",
        "6. Is added inference worth it? Compare backend calls and latency against quality.",
        "7. Detection versus averaging remains an intervention question: V3 adaptive stability detection is preserved, while V4 adds preventative averaging; neither certifies correctness.",
        "",
        "The manageable adaptive subset is recorded in `adaptive_results.csv`, comparing reordered-call detection with a K=2 averaged prior.",
        "",
        "### Runtime guidance",
        "",
        f"This run took **{results['runtime_seconds']:.1f}s**. A rough standard-run estimate from quick-mode work is **{results['standard_runtime_estimate_seconds']:.0f}s**, excluding model startup and cold-download time. Run quick mode first and keep each model service loaded once.",
        "",
        "### Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in results["warnings"])
    return "\n".join(lines) + "\n"


def run_v4_experiment(
    *,
    mode: str = "quick",
    ddm: str = "all",
    budgets: list[int] | None = None,
    seed: int = 0,
    results_root: str | Path = "results",
    laya_device: str | None = None,
    laya_model: str | None = None,
    mica_endpoint: str = "http://127.0.0.1:8010/v1/systemone",
    mica_model: str = "mica-v0.1-4b",
    skip_laya: bool = False,
    order_stability_check: bool = True,
    order_samples: int = 2,
) -> Path:
    del order_stability_check, order_samples  # V3 controls remain accepted; V4 measures all K values explicitly.
    started = time.perf_counter()
    budgets = budgets or ({"quick": [1, 5], "standard": [1, 5, 10, 25, 50], "extended": [1, 2, 5, 10, 25, 50, 100, 250]}[mode])
    requested = list(BACKENDS if ddm == "all" else (ddm,))
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    output = _run_directory(root)
    output.mkdir()
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True, timeout=180, check=False)
    scenarios = _all_scenarios(mode, seed)
    # Keep inference-bounded quick mode; standard remains small rather than exhaustive.
    per_domain = 1 if mode == "quick" else 2 if mode == "standard" else 3
    selected = []
    for domain in dict.fromkeys(scenario.domain for scenario in scenarios):
        selected.extend([scenario for scenario in scenarios if scenario.domain == domain][:per_domain])
    load_started = time.perf_counter()
    predictor, laya_error = (None, "disabled") if skip_laya else _backend(laya_device, laya_model)
    laya_load_seconds = time.perf_counter() - load_started
    backend_metadata: dict[str, dict[str, Any]] = {}
    raw: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    adaptive_rows: list[dict[str, Any]] = []
    warnings = ["Small scenario samples support engineering comparison, not broad scientific claims."]
    print("DDM-MCTS V4 Multi-DDM Experiment", flush=True)
    print(f"[1/4] Tests ....................... {'PASS' if tests.returncode == 0 else 'FAIL'}", flush=True)
    for backend in requested:
        if backend == "laya" and predictor is None:
            backend_metadata[backend] = {"status": "unavailable", "load_success": False, "load_seconds": laya_load_seconds, "detail": laya_error or "Laya failed to initialize"}
            warnings.append(f"Laya unavailable: {laya_error}")
            continue
        backend_metadata[backend] = {
            "status": "available",
            "load_success": True,
            "load_seconds": laya_load_seconds if backend == "laya" else 0.0,
            "device": laya_device or "auto" if backend == "laya" else "server-managed",
            "dtype": "runtime-managed" if backend == "laya" else "server/checkpoint-managed",
            "detail": (f"model={laya_model or 'default'}, device={laya_device or 'auto'}" if backend == "laya" else f"model={mica_model}, endpoint={mica_endpoint}"),
        }
        try:
            for scenario_index, scenario in enumerate(selected):
                base = _text_policy(backend, scenario, predictor=predictor, mica_endpoint=mica_endpoint, mica_model=mica_model)
                averaged = {samples: PermutationAveragedPolicy(base, samples=samples, seed=seed + scenario_index) for samples in (1, 2, 3)}
                for samples, policy in averaged.items():
                    measured = _order_rows(scenario, policy, seed + scenario_index)
                    for row in measured:
                        row.update({"backend": backend, "permutation_samples": samples})
                    order_rows.extend(measured)
                    raw.append(_evaluate(scenario, policy, backend=backend, samples=samples, alpha=1.0, budget=0, seed=seed, direct=True))
                    for alpha in (0.25, 0.5):
                        for budget in budgets:
                            raw.append(_evaluate(scenario, policy, backend=backend, samples=samples, alpha=alpha, budget=budget, seed=seed, direct=False))
                if scenario_index == 0:
                    optimal_actions, optimal_value = _optimum(scenario)
                    for intervention, adaptive_policy, stability_check in (
                        ("detection", base, True),
                        ("averaging_k2", averaged[2], False),
                    ):
                        before = adaptive_policy.diagnostics()
                        adaptive = AdaptiveLayaMCTSAgent(
                            scenario.environment,
                            adaptive_policy,
                            AdaptiveSearchConfig(
                                minimum_simulations=min(10, max(budgets)),
                                maximum_simulations=max(10, max(budgets)),
                                order_stability_check=stability_check,
                                order_samples=2,
                                seed=seed,
                            ),
                        )
                        adaptive_started = time.perf_counter()
                        decision = adaptive.decide(scenario.state)
                        adaptive_latency = time.perf_counter() - adaptive_started
                        after = adaptive_policy.diagnostics()
                        objective = float(optimal_actions is not None and decision.action in optimal_actions)
                        adaptive_rows.append({
                            "backend": backend,
                            "domain": scenario.domain,
                            "scenario_id": scenario.scenario_id,
                            "intervention": intervention,
                            "objective": objective,
                            "regret": regret(optimal_value, objective),
                            "simulations": decision.simulations,
                            "ddm_requests": float(after.get("total_policy_requests", 0)) - float(before.get("total_policy_requests", 0)),
                            "actual_inference_calls": float(after.get("actual_inference_calls", 0)) - float(before.get("actual_inference_calls", 0)),
                            "latency_seconds": adaptive_latency,
                            "final_action": decision.action,
                            "trace": decision.metadata,
                        })
        except MicaUnavailableError as exc:
            raw = [row for row in raw if row["backend"] != backend]
            order_rows = [row for row in order_rows if row["backend"] != backend]
            backend_metadata[backend] = {"status": "unavailable", "load_success": False, "load_seconds": 0.0, "detail": str(exc), "model": mica_model, "endpoint": mica_endpoint, "device": "server-managed", "dtype": "server/checkpoint-managed"}
            warnings.append(str(exc))
        print(f"[2/4] {backend:<27} {backend_metadata[backend]['status'].upper()}", flush=True)
    # The alpha-zero baseline is evaluated once, with the identical MCTS implementation.
    for scenario in selected:
        uniform = UniformPolicy()
        for budget in budgets:
            raw.append(_evaluate(scenario, uniform, backend="uniform", samples=1, alpha=0.0, budget=budget, seed=seed, direct=False))
    direct_lookup = {
        (row["backend"], row["domain"], row["scenario_id"], row["permutation_samples"]): row
        for row in raw if row["method"] == "direct"
    }
    for row in raw:
        if row["method"] != "root_mcts" or row["backend"] == "uniform":
            continue
        direct = direct_lookup[(row["backend"], row["domain"], row["scenario_id"], row["permutation_samples"])]
        row["ddm_action"] = direct["first_action"]
        initial, final = direct["first_action_optimal"], row["first_action_optimal"]
        row["correction_outcome"] = (
            "ddm_correct_mcts_retained" if initial and final else
            "ddm_correct_mcts_broke" if initial and final is False else
            "ddm_wrong_mcts_corrected" if initial is False and final else
            "both_wrong" if initial is False and final is False else "ground_truth_unavailable"
        )
    comparison = _aggregate(raw)
    averaging = _averaging_summary(order_rows)
    for backend in requested:
        calls = [row for row in raw if row["backend"] == backend]
        backend_metadata[backend]["inference_errors"] = 0 if backend_metadata[backend]["status"] == "available" else 1
        backend_metadata[backend]["average_incremental_inference_latency_seconds"] = (
            statistics.fmean(float(row["ddm_latency_seconds"]) / float(row["backend_calls"]) for row in calls if row["backend_calls"])
            if any(row["backend_calls"] for row in calls) else None
        )
    elapsed = time.perf_counter() - started
    quick_units = max(1, len(selected) * max(1, len([item for item in requested if backend_metadata.get(item, {}).get("status") == "available"])))
    estimate = elapsed * (12 / quick_units) * (5 / max(1, len(budgets))) if mode == "quick" else elapsed
    results = {
        "configuration": {"version": 4, "mode": mode, "requested_backends": requested, "budgets": budgets, "alphas": [0.25, 0.5], "permutation_samples": [1, 2, 3], "seed": seed},
        "system_checks": {"return_code": tests.returncode, "output": (tests.stdout + tests.stderr).strip()},
        "backend_metadata": backend_metadata,
        "ddm_comparison": comparison,
        "raw_trials": raw,
        "order_sensitivity": order_rows,
        "permutation_averaging": averaging,
        "adaptive_results": adaptive_rows,
        "correction_analysis": {
            label: sum(row.get("correction_outcome") == label for row in raw)
            for label in ("ddm_correct_mcts_retained", "ddm_correct_mcts_broke", "ddm_wrong_mcts_corrected", "both_wrong")
        },
        "runtime_seconds": elapsed,
        "standard_runtime_estimate_seconds": estimate,
        "warnings": warnings,
    }
    (output / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (output / "raw_trials.json").write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")
    _write_csv(output / "ddm_comparison.csv", comparison)
    _write_csv(output / "trust_sweep.csv", comparison)
    _write_csv(output / "order_sensitivity.csv", order_rows)
    _write_csv(output / "permutation_averaging.csv", averaging)
    _write_csv(output / "adaptive_results.csv", [
        {key: json.dumps(value, default=str) if isinstance(value, (dict, list)) else value for key, value in row.items()}
        for row in adaptive_rows
    ])
    (output / "REPORT.md").write_text(_report(results), encoding="utf-8")
    (output / "run.log").write_text(f"tests={tests.returncode}\nruntime_seconds={elapsed}\n" + tests.stdout + tests.stderr, encoding="utf-8")
    print("[3/4] Analysis files .............. DONE", flush=True)
    latest, staging = root / "latest", root / ".latest_staging"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(output, staging)
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(staging, latest)
    shutil.rmtree(staging)
    print("[4/4] V4 report ................... DONE", flush=True)
    return output

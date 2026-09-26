from __future__ import annotations

import csv
import json
import logging
import math
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ddm_mcts.agents import LayaAgent, LayaMCTSAgent, MCTSAgent, RandomAgent
from ddm_mcts.environments import ConnectFour, ConnectFourState
from ddm_mcts.policies import LayaPolicy, LayaUnavailableError, Policy, UniformPolicy
from ddm_mcts.search import MCTS, MCTSConfig

from .arena import Arena


@dataclass(frozen=True, slots=True)
class DiagnosticPosition:
    name: str
    description: str
    state: ConnectFourState
    correct_action: int | None = None


class FixedPriorPolicy(Policy[ConnectFourState, int]):
    def __init__(self, favored_action: int, weight: float = 0.97) -> None:
        self.favored_action = favored_action
        self.weight = weight

    def probabilities(self, state: ConnectFourState, legal_actions: tuple[int, ...]) -> dict[int, float]:
        del state
        if self.favored_action not in legal_actions:
            return {action: 1 / len(legal_actions) for action in legal_actions}
        other = (1 - self.weight) / max(1, len(legal_actions) - 1)
        return {action: self.weight if action == self.favored_action else other for action in legal_actions}


def _play(env: ConnectFour, actions: list[int]) -> ConnectFourState:
    state = env.initial_state()
    for action in actions:
        state = env.step(state, action)
    return state


def diagnostic_positions(env: ConnectFour) -> list[DiagnosticPosition]:
    """Deterministic positions with independently understandable tactical labels."""
    near_end = _play(
        env,
        [3, 2, 4, 3, 2, 4, 5, 1, 0, 6, 1, 0, 6, 5, 3, 2, 4, 3, 2, 4, 5, 1, 0, 6],
    )
    return [
        DiagnosticPosition("empty_board", "Opening position; no uniquely correct move is asserted.", env.initial_state()),
        DiagnosticPosition("immediate_win", "Player 1 wins immediately by playing column 0.", _play(env, [0, 1, 0, 1, 0, 2]), 0),
        DiagnosticPosition("immediate_block", "Player 1 must block player 2 in column 3.", _play(env, [0, 3, 1, 3, 2, 3]), 3),
        DiagnosticPosition("tactical_block", "A second horizontal threat requires column 3.", _play(env, [6, 0, 6, 1, 5, 2]), 3),
        DiagnosticPosition("vertical_win", "Player 1 wins vertically in column 4.", _play(env, [4, 0, 4, 1, 4, 2]), 4),
        DiagnosticPosition("near_endgame", "A deterministic crowded non-terminal position.", near_end),
    ]


def _entropy(probabilities: dict[int, float]) -> float:
    return -sum(p * math.log(p) for p in probabilities.values() if p > 0)


def _search_record(
    env: ConnectFour,
    state: ConnectFourState,
    policy: Policy[ConnectFourState, int],
    budget: int,
    seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = MCTS(env, policy, MCTSConfig(simulations=budget, seed=seed)).search(state)
    return {
        "selected_action": result.action,
        "simulations": result.simulations,
        "latency_seconds": time.perf_counter() - started,
        "root_statistics": result.root_statistics(),
    }


def _board_rows(state: ConnectFourState) -> list[list[str]]:
    symbols = {0: ".", 1: "X", 2: "O"}
    return [[symbols[state.at(row, col)] for col in range(7)] for row in range(6)]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _unit_tests() -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return {
            "status": "pass" if completed.returncode == 0 else "fail",
            "return_code": completed.returncode,
            "output": (completed.stdout + completed.stderr).strip(),
            "latency_seconds": time.perf_counter() - started,
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"status": "skipped", "error": str(exc), "latency_seconds": time.perf_counter() - started}


def _position_diagnostics(
    env: ConnectFour,
    positions: list[DiagnosticPosition],
    laya: LayaPolicy,
    budget: int,
    seed: int,
) -> tuple[list[dict[str, Any]], str | None]:
    records: list[dict[str, Any]] = []
    laya_error: str | None = None
    for index, position in enumerate(positions):
        legal = env.legal_actions(position.state)
        base = {
            "name": position.name,
            "description": position.description,
            "board": _board_rows(position.state),
            "board_values": list(position.state.board),
            "current_player": position.state.player,
            "legal_actions": list(legal),
            "correct_action": position.correct_action,
        }
        vanilla = _search_record(env, position.state, UniformPolicy(), budget, seed + index)
        base["vanilla_mcts"] = vanilla
        if laya_error is None:
            try:
                before = laya.diagnostics()
                started = time.perf_counter()
                priors = laya.probabilities(position.state, legal)
                policy_latency = time.perf_counter() - started
                raw = laya.last_raw_output
                inference_error = laya.last_error
                # Repeat exactly once to verify a measurable cache hit.
                laya.probabilities(position.state, legal)
                after = laya.diagnostics()
                guided = _search_record(env, position.state, laya, budget, seed + index)
                ranking = sorted(priors, key=priors.get, reverse=True)
                top = ranking[0]
                base["laya"] = {
                    "raw_output": raw,
                    "probabilities": priors,
                    "ranking": ranking,
                    "entropy": _entropy(priors),
                    "top_action": top,
                    "top_probability": priors[top],
                    "correct_action_probability": priors.get(position.correct_action) if position.correct_action is not None else None,
                    "correct_action_rank": ranking.index(position.correct_action) + 1 if position.correct_action in ranking else None,
                    "request_latency_seconds": policy_latency,
                    "cache_hits_added_by_repeat": after["cache_hits"] - before["cache_hits"],
                    "fallback_used": inference_error is not None,
                    "inference_error": inference_error,
                }
                base["laya_mcts"] = guided
                base["comparison"] = {
                    "laya_correct": top == position.correct_action if position.correct_action is not None else None,
                    "vanilla_correct": vanilla["selected_action"] == position.correct_action if position.correct_action is not None else None,
                    "guided_correct": guided["selected_action"] == position.correct_action if position.correct_action is not None else None,
                    "laya_and_vanilla_agree": top == vanilla["selected_action"],
                    "laya_changed_search_result": guided["selected_action"] != vanilla["selected_action"],
                    "mcts_overrode_laya": guided["selected_action"] != top,
                    "beneficial_override": (
                        guided["selected_action"] == position.correct_action and top != position.correct_action
                        if position.correct_action is not None else None
                    ),
                }
            except LayaUnavailableError as exc:
                laya_error = str(exc)
                base["laya"] = {"status": "skipped", "error": laya_error}
                base["laya_mcts"] = {"status": "skipped", "error": laya_error}
        else:
            base["laya"] = {"status": "skipped", "error": laya_error}
            base["laya_mcts"] = {"status": "skipped", "error": laya_error}
        records.append(base)
    return records, laya_error


def _mock_prior_curves(env: ConnectFour, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    state = _play(env, [0, 1, 0, 1, 0, 2])
    correct, bad = 0, 6
    bad_rows = []
    for budget in (1, 5, 10, 25, 50, 100, 250, 500):
        actions = [
            _search_record(env, state, FixedPriorPolicy(bad, 0.9), budget, seed + trial)["selected_action"]
            for trial in range(20)
        ]
        correct_count = actions.count(correct)
        bad_rows.append({"budget": budget, "trials": len(actions), "bad_prior_action": bad, "first_final_action": actions[0], "correct_count": correct_count, "recovery_rate": correct_count / len(actions), "correct": correct_count > len(actions) / 2})
    good_rows = []
    for budget in (1, 2, 5, 10, 25, 50):
        uniform_correct = guided_correct = 0
        trials = 20
        for trial in range(trials):
            uniform = _search_record(env, state, UniformPolicy(), budget, seed + trial)
            guided = _search_record(env, state, FixedPriorPolicy(correct), budget, seed + trial)
            uniform_correct += uniform["selected_action"] == correct
            guided_correct += guided["selected_action"] == correct
        good_rows.append({"budget": budget, "trials": trials, "uniform_correct": uniform_correct, "good_prior_correct": guided_correct})
    return bad_rows, good_rows


def _matchup(
    env: ConnectFour,
    name_a: str,
    name_b: str,
    laya: LayaPolicy,
    games: int,
    budget: int,
    seed: int,
) -> dict[str, Any]:
    def make(name: str, offset: int):
        if name == "random":
            return RandomAgent(env, seed + offset)
        if name == "mcts":
            return MCTSAgent(env, budget, seed + offset)
        if name == "laya":
            return LayaAgent(env, laya, seed + offset)
        return LayaMCTSAgent(env, laya, budget, seed + offset)

    result = Arena(env).run(
        make(name_a, 0),
        make(name_b, 1),
        games,
        seed=seed,
        metadata={"simulations": budget, "game_seeds": [seed + i for i in range(games)]},
    )
    return result.to_dict()


def _policy_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in records if "probabilities" in row.get("laya", {})]
    labeled = [row for row in usable if row["correct_action"] is not None]
    if not usable:
        return {"status": "skipped", "evaluated_states": 0}
    entropies = [row["laya"]["entropy"] for row in usable]
    maxima = [row["laya"]["top_probability"] for row in usable]
    return {
        "status": "complete",
        "evaluated_states": len(usable),
        "mean_entropy": statistics.fmean(entropies),
        "median_entropy": statistics.median(entropies),
        "mean_maximum_action_probability": statistics.fmean(maxima),
        "frequency_top_over_0_5": sum(value > 0.5 for value in maxima) / len(maxima),
        "frequency_top_over_0_75": sum(value > 0.75 for value in maxima) / len(maxima),
        "frequency_top_over_0_9": sum(value > 0.9 for value in maxima) / len(maxima),
        "tactical_top_1_accuracy": sum(row["laya"]["top_action"] == row["correct_action"] for row in labeled) / len(labeled) if labeled else None,
        "mean_correct_action_probability": statistics.fmean(row["laya"]["correct_action_probability"] for row in labeled) if labeled else None,
        "mean_correct_action_rank": statistics.fmean(row["laya"]["correct_action_rank"] for row in labeled) if labeled else None,
    }


def _markdown(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    health = summary["system_health"]
    policy = summary["policy_confidence"]
    bad = summary["bad_prior_recovery"]
    good = summary["good_prior_efficiency"]
    laya_ok = summary["laya_available"]
    labeled = [row for row in records if row["correct_action"] is not None]
    mock_good_helps = sum(r["good_prior_correct"] for r in good) > sum(r["uniform_correct"] for r in good)
    bad_recovers = any(row["correct"] for row in bad if row["budget"] >= 50)
    guided_wins = [row for row in summary["budget_sweep"] if row.get("guided_available") and row["guided_win_rate"] > 0.5]
    if not bad_recovers:
        interpretation = "Guided MCTS did not reliably overcome the deliberately bad prior at larger tested budgets. Inspect search configuration before interpreting Laya results."
    elif mock_good_helps and laya_ok and not guided_wins:
        interpretation = "Useful mock priors improved low-budget decisions and bad priors were recoverable, so the guided-search mechanism appears functional. Laya did not show a head-to-head advantage in this preliminary run."
    elif mock_good_helps and guided_wins:
        interpretation = "Guided search worked with controlled priors, and Laya-guided search showed an advantage at some tested budgets. More games and seeds are required before drawing a strong conclusion."
    elif not laya_ok:
        interpretation = "The policy-independent diagnostics ran, but Laya was unavailable. No conclusion about Laya's Connect Four usefulness can be drawn from this run."
    else:
        interpretation = "Observed results were mixed. This run is diagnostic and too small to establish a reliable performance difference."
    lines = [
        "# ddm-mcts Experiment Report", "", "## Executive Summary", "", interpretation,
        "", "This is a preliminary diagnostic run, not a statistical-significance claim.",
        "", "## System Health", "",
        f"- Unit tests: **{health['unit_tests']['status'].upper()}** — `{health['unit_tests'].get('output', '')}`",
        f"- Environment checks: **{health['environment_status']}**",
        f"- MCTS smoke test: **{health['mcts_status']}**",
        f"- Laya available: **{'yes' if laya_ok else 'no'}**",
        f"- Requested Laya device/model: `{health['laya_configuration']}`",
        "", "The existing vanilla agent is uniform-prior PUCT. Laya-guided search uses the same selection equation with Laya priors, isolating prior influence.",
        "", "## Laya Policy Diagnostics", "",
    ]
    if policy.get("status") == "complete":
        lines += [
            f"Evaluated {policy['evaluated_states']} positions. Mean entropy: **{policy['mean_entropy']:.3f}**; median entropy: **{policy['median_entropy']:.3f}**; mean top probability: **{policy['mean_maximum_action_probability']:.1%}**.",
            f"Tactical top-1 accuracy: **{policy['tactical_top_1_accuracy']:.1%}**. These are policy-confidence and tactical-accuracy diagnostics, not a calibration measurement.",
        ]
    else:
        lines.append(f"Skipped: {summary.get('laya_error', 'Laya unavailable')}.")
    lines += ["", "## Tactical Tests", "", "| Test | Laya | Uniform MCTS | Laya + MCTS | Correct action |", "|---|---:|---:|---:|---:|"]
    for row in labeled:
        laya = row["laya"].get("top_action", "SKIP")
        guided = row["laya_mcts"].get("selected_action", "SKIP")
        lines.append(f"| {row['name']} | {laya} | {row['vanilla_mcts']['selected_action']} | {guided} | {row['correct_action']} |")
    lines += ["", "### Position details", ""]
    for row in labeled:
        lines += [f"#### {row['name'].replace('_', ' ').title()}", "", "| " + " | ".join(str(i) for i in range(7)) + " |", "|" + "---|" * 7]
        lines += ["| " + " | ".join(board_row) + " |" for board_row in row["board"]]
        lines += ["", f"Current player: **{row['current_player']}**. Known action: **column {row['correct_action']}**.", ""]
        if "probabilities" in row.get("laya", {}):
            probabilities = ", ".join(f"column {action}: {probability:.1%}" for action, probability in row["laya"]["probabilities"].items())
            comparison = row["comparison"]
            if comparison["laya_correct"] and comparison["guided_correct"]:
                explanation = "Laya ranked the labeled move first and guided search retained it."
            elif comparison["beneficial_override"]:
                explanation = "Search overruled Laya and changed the decision to the labeled move."
            elif comparison["laya_correct"] and not comparison["guided_correct"]:
                explanation = "Laya ranked the labeled move first, but search changed it to a different move."
            else:
                explanation = "Laya did not rank the labeled move first in this run."
            lines += [f"Laya priors: {probabilities}.", "", explanation, ""]
        else:
            lines += [f"Laya: **SKIPPED**. Uniform MCTS selected column {row['vanilla_mcts']['selected_action']}.", ""]
    lines += ["", "## Search Budget Results", "", "| Budget | Games | Uniform MCTS wins | Guided wins | Draws |", "|---:|---:|---:|---:|---:|"]
    for row in summary["budget_sweep"]:
        lines.append(f"| {row['budget']} | {row['games']} | {row.get('uniform_wins', '—')} | {row.get('guided_wins', '—')} | {row.get('draws', '—')} |")
    lines += ["", "## Policy Confidence", "", "```json", json.dumps(policy, indent=2), "```", "", "## When MCTS Overruled Laya", ""]
    overrides = summary["mcts_overrides"]
    if overrides:
        lines += ["| Position | Laya top (p) | Search action | Outcome |", "|---|---|---:|---|"]
        for row in overrides:
            lines.append(f"| {row['position']} | {row['laya_action']} ({row['laya_probability']:.1%}) | {row['search_action']} | {row['classification']} |")
    else:
        lines.append("No measured Laya/search disagreements, or Laya was unavailable.")
    lines += ["", "## Prior Recovery Test", "", "| Budget | Trials | Bad prior action | Correct recoveries | Recovery rate |", "|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['budget']} | {r['trials']} | {r['bad_prior_action']} | {r['correct_count']} | {r['recovery_rate']:.1%} |" for r in bad]
    lines += ["", "## Good Prior Test", "", "| Budget | Trials | Uniform correct | Good-prior correct |", "|---:|---:|---:|---:|"]
    lines += [f"| {r['budget']} | {r['trials']} | {r['uniform_correct']} | {r['good_prior_correct']} |" for r in good]
    cache = summary["cache_stats"]
    latency = summary["latency"]
    lines += [
        "", "## Performance", "", f"Cache/inference counters: `{json.dumps(cache)}`", "",
        f"Mean vanilla position-search latency: **{latency['mean_vanilla_search_seconds']:.4f}s**.",
    ]
    if latency.get("mean_guided_search_seconds") is not None:
        lines.append(f"Mean Laya-guided position-search latency: **{latency['mean_guided_search_seconds']:.4f}s**. Measured Laya inference time is reported separately; remaining time is only an approximate residual because timers overlap nested work.")
    warnings = summary["warnings"]
    lines += ["", "## Important Warnings", ""] + [f"- {warning}" for warning in warnings]
    lines += ["", "## Interpretation", "", interpretation, "", "Inspect `positions/`, `benchmarks/`, `budgets/`, and `diagnostics/` for raw machine-readable evidence."]
    return "\n".join(lines) + "\n"


def run_experiment(
    *,
    games: int = 10,
    budgets: list[int] | None = None,
    seed: int = 0,
    extended: bool = False,
    results_root: str | Path = "results",
    laya_device: str | None = None,
    laya_model: str | None = None,
    skip_laya: bool = False,
) -> Path:
    budgets = budgets or [10, 25, 50, 100, 250]
    if extended:
        budgets = sorted(set([*budgets, 500, 1000]))
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    output = Path(results_root) / f"experiment_{stamp}"
    for child in ("positions", "benchmarks", "budgets", "diagnostics"):
        (output / child).mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output / "run.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logging.captureWarnings(True)
    log_lines: list[str] = []

    def progress(message: str) -> None:
        print(message, flush=True)
        log_lines.append(f"{datetime.now().isoformat()} {message}")

    progress("[1/8] Running unit tests...")
    tests = _unit_tests()
    progress(tests["status"].upper())
    env = ConnectFour()
    positions = diagnostic_positions(env)
    environment_ok = all(not env.is_terminal(p.state) and p.correct_action in env.legal_actions(p.state) for p in positions if p.correct_action is not None)
    smoke = _search_record(env, positions[1].state, UniformPolicy(), 100, seed)

    progress("[2/8] Testing deterministic positions and Laya availability...")
    def skipped_predictor(state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        del state, questions
        raise LayaUnavailableError("Laya was disabled with --skip-laya")

    laya = LayaPolicy(
        device=laya_device,
        model=laya_model,
        predictor=skipped_predictor if skip_laya else None,
    )
    position_records, laya_error = _position_diagnostics(env, positions, laya, 100, seed)
    for record in position_records:
        _write_json(output / "positions" / f"{record['name']}.json", record)
        if record["correct_action"] is not None:
            result = record.get("comparison", {})
            status = "PASS" if (record["vanilla_mcts"]["selected_action"] == record["correct_action"] and (laya_error or result.get("guided_correct"))) else "WARN"
            progress(f"{status}  {record['name']}")

    progress("[3/8] Testing bad-prior recovery...")
    bad_curve, good_curve = _mock_prior_curves(env, seed)
    _write_csv(output / "diagnostics" / "bad_prior_recovery.csv", bad_curve)
    progress("PASS" if any(row["correct"] for row in bad_curve[4:]) else "WARN")

    progress("[4/8] Testing good-prior efficiency...")
    _write_csv(output / "diagnostics" / "good_prior_efficiency.csv", good_curve)
    progress("PASS" if sum(r["good_prior_correct"] for r in good_curve) > sum(r["uniform_correct"] for r in good_curve) else "WARN")

    progress("[5/8] Running matchup diagnostics...")
    benchmark_results: dict[str, Any] = {}
    matchups = [("mcts", "random")]
    if laya_error is None:
        matchups += [("laya", "random"), ("laya-mcts", "random"), ("laya", "mcts"), ("laya", "laya-mcts"), ("mcts", "laya-mcts")]
    diagnostic_games = min(games, 4)
    for index, (a, b) in enumerate(matchups):
        try:
            result = _matchup(env, a, b, laya, diagnostic_games, 100, seed + index * 1000)
            benchmark_results[f"{a}_vs_{b}"] = result
            _write_json(output / "benchmarks" / f"{a}_vs_{b}.json", result)
            progress(f"Complete  {a} vs {b}")
        except LayaUnavailableError as exc:
            laya_error = str(exc)
            progress(f"SKIP  {a} vs {b}: {exc}")

    progress("[6/8] Running paired budget sweep...")
    budget_rows = []
    for index, budget in enumerate(budgets):
        if laya_error is None:
            result = _matchup(env, "mcts", "laya-mcts", laya, games, budget, seed + index * 10000)
            a, b = result["agent_a"], result["agent_b"]
            row = {"budget": budget, "games": games, "uniform_wins": a["wins"], "guided_wins": b["wins"], "draws": a["draws"], "uniform_win_rate": a["win_rate"], "guided_win_rate": b["win_rate"], "uniform_average_latency_ms": a["average_decision_latency_ms"], "guided_average_latency_ms": b["average_decision_latency_ms"], "guided_available": True, "game_seeds": result["metadata"]["game_seeds"]}
            _write_json(output / "budgets" / f"budget_{budget}.json", result)
        else:
            baseline = _matchup(env, "mcts", "random", laya, games, budget, seed + index * 10000)
            row = {"budget": budget, "games": games, "guided_available": False, "baseline_mcts_vs_random": baseline, "game_seeds": baseline["metadata"]["game_seeds"]}
            _write_json(output / "budgets" / f"budget_{budget}.json", row)
        budget_rows.append(row)
        progress(f"Budget {budget} complete")

    progress("[7/8] Analysing policy, overrides, latency, and cache...")
    policy_summary = _policy_summary(position_records)
    policy_rows = []
    overrides = []
    for row in position_records:
        if "probabilities" in row.get("laya", {}):
            for action, probability in row["laya"]["probabilities"].items():
                policy_rows.append({"position": row["name"], "action": action, "probability": probability, "rank": row["laya"]["ranking"].index(action) + 1, "correct_action": row["correct_action"]})
            if row["comparison"]["mcts_overrode_laya"]:
                if row["correct_action"] is None:
                    classification = "unknown (no labeled answer)"
                elif row["comparison"]["beneficial_override"]:
                    classification = "corrected Laya"
                elif row["laya"]["top_action"] == row["correct_action"]:
                    classification = "changed correct Laya choice"
                else:
                    classification = "changed decision; neither matched label"
                overrides.append({"position": row["name"], "state": row["board_values"], "laya_action": row["laya"]["top_action"], "laya_probability": row["laya"]["top_probability"], "search_action": row["laya_mcts"]["selected_action"], "root_statistics": row["laya_mcts"]["root_statistics"], "budget": row["laya_mcts"]["simulations"], "classification": classification})
    _write_csv(output / "diagnostics" / "laya_policy.csv", policy_rows)
    _write_csv(output / "diagnostics" / "search_comparison.csv", [{"position": r["name"], **r.get("comparison", {})} for r in position_records])
    cache_stats = laya.diagnostics()
    _write_json(output / "diagnostics" / "cache_stats.json", cache_stats)
    vanilla_times = [r["vanilla_mcts"]["latency_seconds"] for r in position_records]
    guided_times = [r["laya_mcts"]["latency_seconds"] for r in position_records if "latency_seconds" in r.get("laya_mcts", {})]
    latency = {"mean_vanilla_search_seconds": statistics.fmean(vanilla_times), "mean_guided_search_seconds": statistics.fmean(guided_times) if guided_times else None, "measured_total_laya_inference_seconds": cache_stats["total_inference_seconds"], "measured_average_laya_inference_seconds": cache_stats["average_inference_seconds"], "measured_average_cached_lookup_seconds": cache_stats["average_cache_lookup_seconds"]}
    _write_csv(output / "diagnostics" / "latency.csv", [latency])
    warnings = ["Game counts are preliminary and insufficient for claims of statistical significance.", "Laya confidence on Connect Four has not been calibrated; entropy and probabilities are diagnostic only."]
    if laya_error:
        warnings.append(f"Laya diagnostics were skipped or stopped: {laya_error}")
    warnings.append("Any checkpoint temperature warning emitted by Laya is preserved in run.log/stdout; treat raw confidence cautiously.")
    summary = {
        "configuration": {"games": games, "budgets": budgets, "seed": seed, "extended": extended, "laya_device": laya_device, "laya_model": laya_model, "skip_laya": skip_laya},
        "system_health": {"unit_tests": tests, "environment_status": "pass" if environment_ok else "fail", "mcts_status": "pass" if smoke["selected_action"] == 0 else "fail", "laya_configuration": f"device={laya_device or 'auto'}, model={laya_model or 'auto'}"},
        "laya_available": laya_error is None,
        "laya_error": laya_error,
        "policy_confidence": policy_summary,
        "tactical_positions": position_records,
        "mcts_overrides": overrides,
        "bad_prior_recovery": bad_curve,
        "good_prior_efficiency": good_curve,
        "benchmarks": benchmark_results,
        "budget_sweep": budget_rows,
        "cache_stats": cache_stats,
        "latency": latency,
        "warnings": warnings,
    }
    progress("[8/8] Writing report...")
    _write_json(output / "summary.json", summary)
    (output / "summary.md").write_text(_markdown(summary, position_records), encoding="utf-8")
    progress("Experiment complete.")
    print(f"Report: {output / 'summary.md'}", flush=True)
    file_handler.flush()
    logging.getLogger().removeHandler(file_handler)
    file_handler.close()
    with (output / "run.log").open("a", encoding="utf-8") as handle:
        handle.write("\n".join(log_lines) + "\n")
    return output

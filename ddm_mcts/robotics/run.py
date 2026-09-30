"""Unique structured toolkit run artifacts, never historical experiment outputs."""

import json
import time
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


def _record(value):
    """Serialize common toolkit values; custom types retain a readable representation."""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (str, int, float, bool, list, tuple, dict)) or value is None:
        return value
    return repr(value)


def run_episode(world, planner, output_root="robotics_runs", *, configuration=None, observer=None):
    root = Path(output_root).resolve()
    historical = Path(__file__).resolve().parents[2] / "results"
    if root == historical or historical in root.parents:
        raise ValueError("robotics logs cannot use historical results directory")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:12]
    folder = root / run_id
    folder.mkdir(parents=True, exist_ok=False)
    backend = getattr(world, "backend", None)
    controller = getattr(world, "controller", None)
    config = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "backend": type(backend).__name__ if backend is not None else type(world).__name__,
        "physics_steps": getattr(backend, "physics_steps", None),
        "controller": _record(getattr(controller, "config", None)),
        "actions": [_record(action) for action in world.legal_actions()],
        "gravity_compensated_bodies": getattr(getattr(backend, "model", None), "ngravcomp", None),
        "task": _record(world.task),
        "policy": type(planner.search.policy).__name__,
        "search": asdict(planner.search.config),
        "horizon": planner.adapter.horizon,
        "initial_state": _record(world.get_state()),
        "configuration": configuration or {},
        "agent": getattr(planner, "run_configuration", None),
    }
    if hasattr(planner, "start_run"):
        planner.start_run(folder)
    (folder / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    decisions = 0
    error = None
    total_planning_seconds = 0.0
    try:
        with (folder / "steps.jsonl").open("x", encoding="utf-8") as stream:
            while True:
                if hasattr(planner, "prepare"):
                    perceived = planner.prepare()
                    if observer is not None and hasattr(observer, "perceived"):
                        observer.perceived(perceived)
                if world.task.is_terminal(world.get_state()):
                    break
                if observer is not None:
                    observer.before_plan(world.get_state(), decisions + 1)
                started = time.perf_counter()
                result = planner.plan()
                planning_seconds = time.perf_counter() - started
                total_planning_seconds += planning_seconds
                if observer is not None:
                    observer.after_plan(result.action, planning_seconds)
                execution_started = time.perf_counter()
                state = world.step(result.action) if observer is None else observer.execute(world, result.action)
                execution_seconds = time.perf_counter() - execution_started
                logged_state = planner.logged_state(state) if hasattr(planner, "logged_state") else state
                record = {
                    "step": decisions + 1,
                    "action": _record(result.action),
                    "state": _record(logged_state),
                    "objective": world.task.evaluate(state),
                    "planning_seconds": planning_seconds,
                    "execution_seconds": execution_seconds,
                    "decision_context": getattr(planner, "decision_context", None),
                    "simulations": result.simulations,
                    "root_statistics": [{**row, "action": str(row["action"])} for row in result.root_statistics()],
                }
                stream.write(json.dumps(record, default=str) + "\n")
                stream.flush()
                decisions += 1
                if observer is not None:
                    observer.after_step(state)
    except (Exception, KeyboardInterrupt) as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary = {
            "success": error is None and world.task.is_success(world.get_state()),
            "decisions": decisions,
            "objective": world.task.evaluate(world.get_state()),
            "error": error,
            "final_state": _record(
                planner.logged_state(world.get_state())
                if hasattr(planner, "logged_state") and planner.state is not None
                else world.get_state()
            ),
            "final_perception": getattr(planner, "decision_context", None),
            "total_planning_seconds": total_planning_seconds,
            "policy_diagnostics": (
                planner.policy_diagnostics()
                if hasattr(planner, "policy_diagnostics")
                else getattr(planner.search.policy, "diagnostics", lambda: {})()
            ),
        }
        (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (folder / "SUMMARY.md").write_text(
            f"# Robotics run {run_id}\n\nSuccess: {summary['success']}\n\nDecisions: {decisions}\n\nObjective: {summary['objective']}\n\nError: {error}\n",
            encoding="utf-8",
        )
    if observer is not None:
        observer.finish(world.get_state(), summary)
    return folder

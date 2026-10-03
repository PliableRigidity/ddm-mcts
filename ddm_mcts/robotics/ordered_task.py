"""Bounded language tasks and ordered execution over the existing PhysicalAgent.

No robot resets, extra planners, or model inference live in this module. Scene
geometry is used explicitly for verification, never to supply visual inference.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import dist, isfinite
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from .approach import ApproachReachTask
from .observation import SemanticGoal


@dataclass(frozen=True)
class ApproachGoal:
    id: str
    target: str
    epsilon: float = 0.012
    max_actions: int = 30
    standoff: float = 0.05
    object_radius: float = 0.04

    def __post_init__(self):
        if (
            not self.id.strip()
            or self.target not in ("cylinder", "sphere", "cube")
            or isinstance(self.max_actions, bool)
            or not isinstance(self.max_actions, int)
            or self.max_actions < 1
            or not isfinite(self.epsilon)
            or self.epsilon <= 0
        ):
            raise ValueError("approach goal requires an ID, supported shape and positive action budget")
        ApproachReachTask(SemanticGoal(self.target), self.epsilon, self.max_actions, self.standoff, self.object_radius)

    def task(self, completed_actions=0):
        # ReachTask counts cumulative robot actions. Allocate a fresh budget while
        # preserving that counter and every physical integration state variable.
        return ApproachReachTask(
            SemanticGoal(self.target),
            self.epsilon,
            completed_actions + self.max_actions,
            self.standoff,
            self.object_radius,
        )


@dataclass(frozen=True)
class PhysicalTask:
    instruction: str
    subgoals: tuple[ApproachGoal, ...]

    def __post_init__(self):
        if not self.instruction.strip() or not self.subgoals or len({g.id for g in self.subgoals}) != len(self.subgoals):
            raise ValueError("ordered task requires an instruction and uniquely identified subgoals")


def parse_physical_task(instruction: str, **criteria) -> PhysicalTask:
    """Parse approach/visit/go-to lists of cube, cylinder and sphere, in order.

    This is deliberately not unrestricted NLP. Unknown words, negation and other
    operations are rejected rather than silently dropping parts of an instruction.
    Duplicate targets are allowed and receive distinct stable occurrence IDs.
    """
    text = instruction.strip().lower().rstrip(".! ")
    match = re.fullmatch(r"(?:approach|visit|go to)\s+(.+)", text)
    if match is None:
        raise ValueError("supported instruction: Approach/Visit/Go to a comma/then/and ordered shape list")
    parts = re.split(r"\s*(?:,\s*(?:and\s+)?(?:then\s+)?|\band\s+(?:then\s+)?|\bthen\s+)\s*", match[1])
    targets = []
    for part in parts:
        item = re.fullmatch(r"(?:(?:approach|visit|go to)\s+)?(?:the\s+)?(cube|cylinder|sphere|cylindrical|spherical)(?:\s+object)?", part)
        if item is None:
            raise ValueError(f"unsupported ordered target: {part!r}; use cube, cylinder or sphere")
        targets.append({"cylindrical": "cylinder", "spherical": "sphere"}.get(item[1], item[1]))
    return PhysicalTask(instruction, tuple(ApproachGoal(f"goal-{i + 1}", target, **criteria) for i, target in enumerate(targets)))


@dataclass(frozen=True)
class GoalVerification:
    success: bool
    target: str
    perceived_target: str | None
    tcp_error_m: float | None
    true_waypoint_error_m: float | None
    threshold_m: float
    approach_waypoint: tuple[float, float, float] | None
    final_clearances_m: dict[str, float]
    minimum_transit_clearance_m: float | None
    reason: str


class ApproachVerifier:
    """Verify physical TCP proximity to the FINAL outside-object target.

    Signed gripper/scene distances, if the simulator exposes them, are explicit
    privileged evaluation. No answer is fed back into perception or Qwen prompts.
    Without that capability, clearance is recorded as unavailable, not invented.
    """

    def clearances(self, environment):
        geometry = getattr(environment, "geometry_diagnostics", None)
        entities = getattr(environment, "target_entities", None)
        if geometry is None or entities is None:
            return {}
        return {e.label: value for e in entities() if (value := geometry(e.label).get("minimum_gripper_target_distance_m")) is not None}

    def verify(self, agent, goal, minimum_transit_clearance=None):
        state = agent.state
        label = state.target().label if state is not None else None
        waypoint = agent.task.diagnostics(state)["pre_contact_target"] if state is not None else None
        error = dist(agent.environment.get_state().position, waypoint) if waypoint is not None else None
        truth_provider = getattr(agent.environment, "target_entities", None)
        truth_error = None
        truth_ok = True
        if truth_provider is not None and waypoint is not None:
            truth = next((e for e in truth_provider() if e.label == goal.target), None)
            if truth is None:
                truth_ok = False
            else:
                diagnostics = agent.task.diagnostics(state)
                allowance = diagnostics["object_bounding_radius_m"] + diagnostics["standoff_m"]
                actual_waypoint = tuple(c + v * allowance for c, v in zip(truth.position, diagnostics["approach_vector"], strict=True))
                truth_error = dist(agent.environment.get_state().position, actual_waypoint)
                truth_ok = isfinite(truth_error) and truth_error <= goal.epsilon
        clearances = self.clearances(agent.environment)
        semantic_ok = label == goal.target
        tcp_ok = error is not None and isfinite(error) and error <= goal.epsilon
        clearance_ok = all(isfinite(v) and v > 0 for v in clearances.values()) and (
            minimum_transit_clearance is None or (isfinite(minimum_transit_clearance) and minimum_transit_clearance > 0)
        )
        success = semantic_ok and tcp_ok and truth_ok and clearance_ok
        reason = (
            "verified"
            if success
            else (
                "semantic target mismatch"
                if not semantic_ok
                else "TCP outside final waypoint tolerance"
                if not tcp_ok
                else "TCP outside requested object's true approach tolerance"
                if not truth_ok
                else "invalid geometry clearance"
            )
        )
        return GoalVerification(
            success, goal.target, label, error, truth_error, goal.epsilon, waypoint, clearances, minimum_transit_clearance, reason
        )


def _physical_state(environment):
    record = asdict(environment.get_state())
    record.pop("goal", None)  # The objective can change; physical state cannot.
    return record


def _accounting(agent):
    perception = agent.perception
    visual = agent.visual_policy
    backends = []
    for backend in (getattr(perception, "backend", None), getattr(getattr(visual, "model", None), "backend", None)):
        if backend is not None and all(backend is not b for b in backends):
            backends.append(backend)
    return {
        "logical_perception_requests": agent.perception_requests,
        "observation_seconds": agent.total_observation_seconds,
        "perception_seconds": agent.total_perception_seconds,
        "physical_perception_inference_calls": getattr(perception, "inference_calls", 0),
        "perception_cache_hits": getattr(perception, "cache_hits", 0),
        "physical_perception_inference_seconds": getattr(perception, "inference_seconds", 0),
        "logical_visual_requests": getattr(visual, "calls", 0),
        "visual_cache_hits": getattr(visual, "hits", 0),
        "visual_cache_misses": getattr(visual, "misses", 0),
        "physical_visual_inference_calls": len(getattr(visual, "inference_times", ())),
        "physical_visual_inference_seconds": sum(getattr(visual, "inference_times", ())),
        "model_load_count": sum(b.load_count for b in backends),
        "model_load_seconds": sum(b.load_seconds for b in backends),
        "physical_model_inference_calls": sum(b.calls for b in backends),
        "total_model_inference_seconds": sum(sum(b.inference_seconds) for b in backends),
    }


class _ExecutionObserver:
    """Scope clearance checks to live execution, and reuse one optional viewer."""

    def __init__(self, verifier, agent, visualization):
        self.verifier, self.agent, self.visualization = verifier, agent, visualization
        self.minimum = None
        self.substeps = 0

    def __getattr__(self, name):
        return getattr(self.visualization, name, lambda *args: None)

    def execute(self, world, action):
        def capture(count=True):
            values = self.verifier.clearances(world).values()
            if values:
                value = min(values)
                self.minimum = value if self.minimum is None else min(self.minimum, value)
                if not isfinite(value) or value <= 0:
                    raise RuntimeError("nonpositive gripper/scene clearance during selected execution")
            if count:
                self.substeps += 1

        backend = getattr(world, "backend", None)
        if backend is None or not hasattr(backend, "observe_steps"):
            return world.step(action) if self.visualization is None else self.visualization.execute(world, action)
        capture(count=False)
        with backend.observe_steps(capture):
            return world.step(action) if self.visualization is None else self.visualization.execute(world, action)

    def finish(self, *args):
        # A subgoal ending must not print "final scene" or close/hold the viewer.
        pass


@dataclass(frozen=True)
class TaskResult:
    success: bool
    folder: Path
    trace: dict


class OrderedTaskRunner:
    """Coordinate subgoals using the same observer, perception, MCTS and robot."""

    def __init__(self, agent, verifier=None):
        self.agent = agent
        self.verifier = verifier or ApproachVerifier()

    def run(self, task: PhysicalTask, output_root="robotics_runs", *, visualization=None, configuration=None):
        root = Path(output_root).resolve()
        historical = Path(__file__).resolve().parents[2] / "results"
        if root == historical or historical in root.parents:
            raise ValueError("robotics tasks cannot use historical results directory")
        folder = root / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_task_" + uuid4().hex[:12])
        folder.mkdir(parents=True, exist_ok=False)
        started = perf_counter()
        initial_accounting = _accounting(self.agent)
        trace = {
            "run_id": folder.name,
            "instruction": task.instruction,
            "configuration": configuration or {},
            "components": self.agent.run_configuration.copy(),
            "search": asdict(self.agent.search.config),
            "horizon": self.agent.adapter.horizon,
            "verification_ground_truth": "true requested-object waypoint and signed geometry evaluation only; no inference feedback",
            "initial_physical_state": _physical_state(self.agent.environment),
            "subgoals": [{"goal": asdict(goal), "status": "pending"} for goal in task.subgoals],
            "success": False,
            "failed_subgoal": None,
            "error": None,
        }

        def save():
            trace["final_physical_state"] = _physical_state(self.agent.environment)
            trace["total_task_seconds"] = perf_counter() - started
            totals = _accounting(self.agent)
            trace["accounting"] = {k: totals[k] - initial_accounting[k] for k in totals}
            # Also report the shared session totals (includes any caller's prepare).
            trace["model_session_totals"] = totals
            backends = [getattr(self.agent.perception, "backend", None)]
            visual_backend = getattr(getattr(self.agent.visual_policy, "model", None), "backend", None)
            if visual_backend is not None and all(visual_backend is not b for b in backends):
                backends.append(visual_backend)
            trace["model_sessions"] = [b.diagnostics() for b in backends if b is not None]
            trace["total_mcts_search_seconds"] = sum(r.get("mcts_search_seconds", 0) for r in trace["subgoals"])
            trace["total_execution_seconds"] = sum(r.get("execution_seconds", 0) for r in trace["subgoals"])
            (folder / "task_trace.json").write_text(json.dumps(trace, indent=2, default=str) + "\n", encoding="utf-8")

        print(f"TASK: {task.instruction}", flush=True)
        save()
        for index, (goal, record) in enumerate(zip(task.subgoals, trace["subgoals"], strict=True)):
            subgoal_started = perf_counter()
            record.update(
                status="active", started_at=datetime.now(UTC).isoformat(), start_physical_state=_physical_state(self.agent.environment)
            )
            before = _accounting(self.agent)
            monitor = _ExecutionObserver(self.verifier, self.agent, visualization)
            print(f"SUBGOAL {index + 1}/{len(task.subgoals)}: {goal.target}", flush=True)
            save()
            try:
                # Cache and goal change, not physical reset. Search creates a new
                # root on every plan, so no old objective statistics are retained.
                physical_snapshot = self.agent.environment.snapshot()
                reset_perception = getattr(self.agent.perception, "reset", None)
                if reset_perception is not None:
                    reset_perception()
                self.agent.set_task(goal.task(self.agent.environment.get_state().steps))
                self.agent.prepare()  # Fresh observation and visual context bind.
                record["prepared_physical_state"] = _physical_state(self.agent.environment)
                record["first_observation_sequence"] = self.agent.observation.sequence
                record["preparation_preserved_snapshot"] = self.agent.environment.snapshot() == physical_snapshot
                if record["start_physical_state"] != record["prepared_physical_state"] or not record["preparation_preserved_snapshot"]:
                    raise RuntimeError("goal preparation changed physical state")
                record["log_root"] = str(folder / "subgoals" / goal.id)
                episode = self.agent.run(record["log_root"], visualization=monitor, configuration={"subgoal_id": goal.id})
                summary = json.loads((episode / "summary.json").read_text())
                steps = [json.loads(line) for line in (episode / "steps.jsonl").read_text().splitlines()]
                verification = self.verifier.verify(self.agent, goal, monitor.minimum)
                record.update(
                    run=str(episode),
                    verification=asdict(verification),
                    actions=[s["action"]["name"] for s in steps],
                    qwen_top1=[s["visual_decision"]["model_top1"] for s in steps if "visual_decision" in s],
                    mcts_overrides=sum(not s["visual_decision"]["selected_equals_model_top1"] for s in steps if "visual_decision" in s),
                    mcts_search_seconds=summary["total_mcts_search_seconds"],
                    execution_seconds=summary["total_execution_seconds"],
                    monitored_physics_substeps=monitor.substeps,
                    final_observation_sequence=self.agent.observation.sequence,
                    final_perceived_entity=asdict(self.agent.state.target()),
                    perception_errors_m=self.agent.decision_context.get("perception_errors_m"),
                )
                # The episode's success flag is not sufficient: enforce target
                # identity, final (not lift) waypoint and every sampled clearance.
                if not verification.success:
                    raise RuntimeError(verification.reason)
                record["status"] = "succeeded"
                print(f"SUBGOAL COMPLETE: {goal.target}; TCP error {verification.tcp_error_m:.4f} m", flush=True)
            except (Exception, KeyboardInterrupt) as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                trace.update(failed_subgoal=goal.id, error=record["error"])
                if isinstance(exc, KeyboardInterrupt):
                    raise
            finally:
                after = _accounting(self.agent)
                record.update(
                    end_physical_state=_physical_state(self.agent.environment),
                    finished_at=datetime.now(UTC).isoformat(),
                    elapsed_seconds=perf_counter() - subgoal_started,
                    accounting={k: after[k] - before[k] for k in after},
                    monitored_physics_substeps=monitor.substeps,
                    minimum_transit_clearance_m=monitor.minimum,
                )
                save()
            if record["status"] == "failed":
                break
        trace["success"] = all(r["status"] == "succeeded" for r in trace["subgoals"])
        save()
        print("TASK COMPLETE" if trace["success"] else f"TASK FAILED: {trace['error']}", flush=True)
        return TaskResult(trace["success"], folder, trace)

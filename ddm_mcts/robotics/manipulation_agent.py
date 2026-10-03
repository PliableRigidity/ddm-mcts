"""Composed language/observation/search/skill agent; frozen physics is delegated."""

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

from .manipulation import ManipulationAgent, PickupTask
from .manipulation_observation import ManipulationPerception
from .manipulation_search import plan_skill
from .manipulation_task import parse_manipulation_task
from .observation import Entity, GroundTruthObservationProvider
from .pick_place import ManipulationResult, PickPlaceAgent
from .placement import NextTo, verify_next_to


class CheckpointSkills(PickPlaceAgent):
    """Only adds an observation after the frozen, physically verified lift."""

    def __init__(self, world, checkpoint, **kwargs):
        super().__init__(world, **kwargs)
        self.checkpoint = checkpoint

    def _pickup(self, *args, **kwargs):
        result = super()._pickup(*args, **kwargs)
        self.checkpoint("after_lift", resolve=False)
        return result


class PhysicalAIManipulationAgent:
    def __init__(self, world, *, observation=None, perception=None, policy=None, search_config=None, skills=None):
        self.world = world
        self.observation = observation or GroundTruthObservationProvider(
            world, lambda: tuple(Entity(label, world.object_state(label).pose.position) for label in ("cube", "cylinder"))
        )
        self.perception = perception or ManipulationPerception(world)
        self.policy, self.search_config = policy, search_config
        self.skills = skills or CheckpointSkills(world, self._checkpoint)
        self.active = None

    def _checkpoint(self, stage, *, resolve=True):
        self.perception.invalidate(stage)
        for attempt in range(2):
            before = self.world.snapshot()
            observation = self.observation.observe()
            if self.world.snapshot() != before:
                raise RuntimeError("observation changed physical state")
            row = {
                "stage": stage,
                "attempt": attempt,
                "sequence": observation.sequence,
                "timestamp": observation.timestamp,
                "source": observation.source,
                "robot": asdict(observation.robot),
                "scene": {label: asdict(self.world.object_state(label)) for label in ("cube", "cylinder")},
            }
            self.active["trace"]["observations"].append(row)
            print(f"OBSERVATION: {stage} #{observation.sequence}", flush=True)
            if not resolve:
                return row
            op = self.active["operation"]
            labels = [op.target] + ([op.destination.reference] if isinstance(op.destination, NextTo) else [])
            try:
                row["perception"] = self.perception.resolve(observation, labels)
                return row
            except ValueError as exc:
                row["perception_failure"] = str(exc)
                if self.perception.backend:
                    row["model_diagnostics"] = self.perception.backend.diagnostics()
                if attempt == 1 or not hasattr(self.observation, "alternate_view"):
                    raise
                self.active["trace"].setdefault("semantic_recovery", []).append(
                    {"trigger": str(exc), "attempt": 1, "action": "fresh_alternate_view"}
                )
                self.observation.alternate_view()
                self.perception.invalidate("semantic_recovery")
        raise RuntimeError("semantic recovery exhausted")

    def run_instruction(self, instruction, output_root="robotics_runs", *, visualization=None):
        return self.run_task(parse_manipulation_task(instruction), output_root, visualization=visualization)

    def run_task(self, task, output_root="robotics_runs", *, visualization=None):
        root = Path(output_root).resolve()
        protected = Path(__file__).resolve().parents[2] / "results"
        if root == protected or protected in root.parents:
            raise ValueError("task traces cannot overwrite historical results")
        folder = root / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_manipulation_agent_" + uuid4().hex[:12])
        folder.mkdir(parents=True, exist_ok=False)
        start = perf_counter()
        model_before = self.perception.backend.diagnostics() if self.perception.backend else None
        trace = {
            "instruction": task.instruction,
            "parsed_operations": [asdict(op) for op in task.operations],
            "progress": [{"id": f"operation-{i + 1}", "status": "pending"} for i in range(len(task.operations))],
            "operations": [],
            "success": False,
            "reset_count": 0,
            "perception": "vlm" if self.perception.backend else "ground-truth",
        }
        print(f"INSTRUCTION: {task.instruction}", flush=True)
        for index, op in enumerate(task.operations):
            row = {
                "id": f"operation-{index + 1}",
                "operation": asdict(op),
                "status": "active",
                "observations": [],
                "start_state": self.skills._state(),
            }
            trace["operations"].append(row)
            self.active = {"operation": op, "trace": row}
            trace["progress"][index]["status"] = "active"
            try:
                print(f"OPERATION {index + 1}/{len(task.operations)}: {op.target}", flush=True)
                self._checkpoint("before_operation")
                planning = perf_counter()
                action, diagnostics = plan_skill(self.world, op, folder / "search_futures", policy=self.policy, config=self.search_config)
                row["planning"] = {**diagnostics, "seconds": perf_counter() - planning}
                row["selected_action"] = asdict(action)
                if action.destination is None:
                    result = ManipulationAgent(self.world).run(PickupTask(action.target), folder / "executed", visualization=visualization)
                    row["skill_log"] = str(result.folder)
                    row["physical_verification"] = result.trace
                    if not result.success:
                        raise RuntimeError("pickup_physical_verification_failed")
                    self._checkpoint("after_lift", resolve=False)
                    # Held target may be occluded; contacts/proprioception remain authoritative.
                    self._checkpoint("operation_complete", resolve=False)
                else:
                    result = self.skills.pick_and_place(action.target, action.destination, folder / "executed", visualization=visualization)
                    row["skill_log"] = str(result.folder)
                    row["physical_verification"] = result.trace
                    if not result.success:
                        raise RuntimeError("placement_physical_verification_failed")
                    row["status"] = "physically_complete"
                    observed = self._checkpoint("after_release_and_retreat")
                    physical = result.trace["operations"][0]["end_state"]["objects"][op.target]["pose"]["position"]
                    if not np.allclose(observed["scene"][op.target]["pose"]["position"], physical, atol=1e-9):
                        raise RuntimeError("reobservation_physical_result_inconsistent")
                    if isinstance(op.destination, NextTo):
                        relation = verify_next_to(
                            self.world.object_state(op.target), self.world.object_state(op.destination.reference), op.destination.gap
                        )
                        row["relation_verification"] = asdict(relation)
                        if not relation.success:
                            raise RuntimeError("relation_verification_failed")
                row["status"] = "verified"
            except Exception as exc:
                row["status"] = "failed"
                row["failure"] = {"reason": str(exc), "type": type(exc).__name__, "scene": self.skills._scene()}
            finally:
                trace["progress"][index]["status"] = row["status"]
                row["end_state"] = self.skills._state()
                trace["duration_seconds"] = perf_counter() - start
                (folder / "task_trace.json").write_text(json.dumps(trace, indent=2))
            if row["status"] == "failed":
                break
        trace["success"] = len(trace["operations"]) == len(task.operations) and all(r["status"] == "verified" for r in trace["operations"])
        trace["state_continuity"] = all(
            a["end_state"] == b["start_state"] for a, b in zip(trace["operations"], trace["operations"][1:], strict=False)
        )
        trace["final_scene"] = self.skills._scene()
        trace["cache_invalidations"] = self.perception.invalidations
        trace["logical_semantic_requests"] = self.perception.requests
        if self.perception.backend:
            trace["model"] = self.perception.backend.diagnostics()
            trace["model_session_initial"] = model_before
            initial_calls = model_before["inference_count"]
            trace["model_task_calls"] = trace["model"]["inference_count"] - initial_calls
            trace["model_task_inference_seconds"] = trace["model"]["inference_seconds"][initial_calls:]
        trace["duration_seconds"] = perf_counter() - start
        (folder / "task_trace.json").write_text(json.dumps(trace, indent=2))
        print("TASK COMPLETE" if trace["success"] else "TASK FAILED", flush=True)
        self.active = None
        return ManipulationResult(trace["success"], folder, trace)

    def close(self):
        self.observation.close()

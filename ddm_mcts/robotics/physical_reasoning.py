"""Closed-loop goal composition in the calibrated cube/cylinder MuJoCo domain."""

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from .interaction_search import search_push
from .interactions import InteractionExecutor
from .physical_goals import GoalConjunction, PhysicalGoal, Predicate, TemporalGoal, parse_physical_goals
from .physical_predicates import PhysicalState, support_surface
from .placement import NextTo, PlacementTarget, PlacementWorkspace, next_to_target
from .pose_control import PoseTarget


@dataclass(frozen=True)
class PhysicalReasoningResult:
    success: bool
    folder: Path
    trace: dict


class CompositionalPhysicalAgent:
    def __init__(self, world, *, observation=None, perception=None, search_config=None, policy=None):
        self.world, self.observation, self.perception = world, observation, perception
        self.search_config, self.policy = search_config, policy
        self.state = PhysicalState(world)

    def run_goal_instruction(self, instruction, output="robotics_runs", *, visualization=None):
        started = perf_counter()
        task = parse_physical_goals(instruction)
        folder = Path(output) / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_physical_reason_" + uuid4().hex[:12])
        folder.mkdir(parents=True)
        trace = {
            "instruction": instruction,
            "goals": asdict(task),
            "events": [],
            "goal_results": [],
            "success": False,
            "speculative_substeps": 0,
            "resets": 0,
            "state_continuity": [],
        }
        executor = InteractionExecutor(self.world, state=self.state, output=folder, visualization=visualization)
        try:
            previous_end = None
            for index, goal in enumerate(task.goals):
                start = self._physical_signature()
                if previous_end is not None and start != previous_end:
                    raise RuntimeError("operation_state_discontinuity")
                trace["state_continuity"].append({"start": start, "matches_previous_end": previous_end is None or start == previous_end})
                print(f"GOAL {index + 1}/{len(task.goals)}: {goal}", flush=True)
                observed = self.state.observe()
                trace["events"].append({"event": "observation", "world": observed})
                predicates = goal.predicates if isinstance(goal, GoalConjunction) else (goal,)
                if isinstance(goal, TemporalGoal):
                    elapsed = executor.wait(goal.duration)
                    verified = goal.duration - 1e-8 <= elapsed <= goal.duration + self.world.backend.model.opt.timestep + 1e-8
                    trace["goal_results"].append({"wait_elapsed": elapsed, "success": verified})
                    if not verified:
                        raise RuntimeError("physical_wait_duration_not_verified")
                    print(f"SIMULATION ADVANCED: {elapsed:.3f} s; PREDICATES REFRESHED", flush=True)
                    previous_end = self._physical_signature()
                    trace["state_continuity"][-1]["end"] = previous_end
                    continue
                first = predicates[0]
                initial = [asdict(self.state.evaluate(p)) for p in predicates]
                trace["events"].append({"event": "predicates_evaluated", "results": initial})
                if not all(p["satisfied"] for p in initial):
                    self._ground(first, trace)
                    affordances = self.state.affordances(first.subject) if first.subject != "tower" else {}
                    trace["events"].append({"event": "affordances_generated", "affordances": affordances})
                    if first.predicate == Predicate.HELD:
                        if not affordances["graspable"]:
                            raise RuntimeError("object_not_graspable")
                        executor.grasp(first.subject)
                    elif first.predicate in (Predicate.ON_TOP_OF, Predicate.NEXT_TO, Predicate.AT_LOCATION):
                        self._placement(first, executor, trace)
                    elif first.predicate == Predicate.TOPPLED:
                        self._topple(first, executor, trace)
                    else:
                        raise ValueError("no interaction generator for physical goal")
                final = [asdict(self.state.evaluate(p)) for p in predicates]
                success = all(p["satisfied"] for p in final)
                trace["goal_results"].append({"initial": initial, "final": final, "success": success})
                if not success:
                    raise RuntimeError("physical_goal_not_verified")
                if first.predicate == Predicate.ON_TOP_OF:
                    self.state.groups["tower"] = (first.subject, first.reference)
                previous_end = self._physical_signature()
                trace["state_continuity"][-1]["end"] = previous_end
                print("GOAL VERIFIED", flush=True)
            trace["success"] = len(trace["goal_results"]) == len(task.goals) and all(row["success"] for row in trace["goal_results"])
        except (ValueError, RuntimeError) as exc:
            trace["failure"] = {
                "goal_index": index,
                "phase": executor.phase,
                "reason": str(exc),
                "world": self.state.observe(),
                "tcp": asdict(self.world.controller.pose()),
                "gripper": asdict(executor.gripper.state()),
                "contacts": [asdict(c) for c in executor.contacts.contacts()],
                "predicates": [asdict(self.state.evaluate(p)) for p in predicates if isinstance(p, PhysicalGoal)],
            }
        finally:
            trace["duration_seconds"] = perf_counter() - started
            if self.perception and self.perception.backend:
                trace["vlm"] = self.perception.backend.diagnostics()
            trace["final_world"] = self.state.observe()
            trace["live_physics"] = executor.counts
            trace["interactions"] = executor.events
            (folder / "task_trace.json").write_text(json.dumps(trace, indent=2, default=str))
        return PhysicalReasoningResult(trace["success"], folder, trace)

    def _physical_signature(self):
        data = self.world.backend.data
        return {"time": float(data.time), "qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(), "ctrl": data.ctrl.tolist()}

    def _ground(self, goal, trace):
        # Semantic inference is needed to bind independent known objects before their
        # interaction. Occluded tower members retain verified identity, not new guesses.
        if not self.perception or goal.subject == "tower":
            return
        labels = [goal.subject] + ([goal.reference] if goal.reference and goal.reference != "table" else [])
        self.perception.invalidate("fresh physical goal")
        observation = self.observation.observe()
        for attempt in range(2):
            try:
                resolved = self.perception.resolve(observation, labels)
                trace["events"].append({"event": "semantic_grounding", "result": resolved, "alternate_view": attempt})
                return
            except ValueError as exc:
                trace["events"].append({"event": "semantic_recovery", "attempt": attempt, "reason": str(exc)})
                if attempt or not hasattr(self.observation, "alternate_view"):
                    raise
                self.observation.alternate_view()
                observation = self.observation.observe()

    def _placement(self, goal, executor, trace):
        obj = self.world.object_state(goal.subject)
        held = self.state.evaluate(PhysicalGoal(Predicate.HELD, goal.subject)).satisfied
        if not held and not self.state.affordances(goal.subject)["graspable"]:
            raise ValueError("placement_requires_graspable_or_held_object")
        if goal.predicate == Predicate.ON_TOP_OF:
            if goal.reference not in self.state.affordances(goal.subject)["placeable_on"]:
                raise ValueError("support_unavailable_for_placement")
            surface = support_surface(self.world, goal.reference)
            if not surface.contains_com(surface.pose.position):
                raise ValueError("invalid_support_region")
            position = list(surface.pose.position)
            position[2] += obj.dimensions[2] / 2
            desired = PoseTarget(tuple(position), (1, 0, 0, 0))
        else:
            surface = support_surface(self.world, "table")
            if goal.predicate == Predicate.NEXT_TO:
                target, _ = next_to_target(obj, self.world.object_state(goal.reference), NextTo(goal.reference), PlacementWorkspace())
            else:
                target = PlacementTarget(goal.location)
                PlacementWorkspace().validate(
                    target, obj, [self.world.object_state(label) for label in ("cube", "cylinder") if label != obj.label]
                )
            desired = target.pose
        trace["events"].append({"event": "support_surface_found", "surface": asdict(surface), "desired_object": asdict(desired)})
        if held:
            executor.target = goal.subject
            executor.phase = "carry"
            executor.last_force_time = float(self.world.backend.data.time)
            trace["events"].append(
                {
                    "event": "held_precondition_reused",
                    "object": goal.subject,
                    "measured_grasp": asdict(self.state.held_reference[goal.subject]),
                }
            )
        else:
            executor.grasp(goal.subject)
            if self.perception:
                self.perception.invalidate("verified lift changed scene")
        if goal.predicate == Predicate.ON_TOP_OF:
            surface = support_surface(self.world, goal.reference)
            position = list(surface.pose.position)
            position[2] += obj.dimensions[2] / 2
            desired = PoseTarget(tuple(position), (1, 0, 0, 0))
        trace["events"].append({"event": "verified_lift_observation", "world": self.state.observe()})
        print("TRANSPORTING TO OBSERVED SUPPORT SURFACE", flush=True)
        executor.move_held_object(goal.subject, desired, surface)
        print("SUPPORT CONTACT ESTABLISHED", flush=True)
        executor.release(goal.subject)
        print("RELEASED; RETREATED; SETTLING VERIFIED BY PREDICATES", flush=True)
        trace["events"].append({"event": "post_release_observation", "world": self.state.observe()})
        if self.observation:
            observation = self.observation.observe()
            trace["events"].append(
                {"event": "fresh_camera_after_release", "sequence": observation.sequence, "timestamp": observation.timestamp}
            )
        if self.perception:
            self.perception.invalidate("physical release changed scene")

    def _topple(self, goal, executor, trace):
        if any(self.state.evaluate(PhysicalGoal(Predicate.HELD, label)).satisfied for label in ("cube", "cylinder")):
            raise ValueError("push_requires_free_gripper")
        for attempt in range(2):
            print(f"SEARCHING PHYSICAL PUSH FUTURES; ATTEMPT {attempt + 1}/2", flush=True)
            action, search = search_push(self.world, self.state, goal.subject, policy=self.policy, config=self.search_config)
            trace["speculative_substeps"] += search["speculative_substeps"]
            trace["events"].append({"event": "search_result", "attempt": attempt, **search})
            print(f"SELECTED: {action}", flush=True)
            result = executor.push(action)
            if self.perception:
                self.perception.invalidate("physical push changed scene")
            trace["events"].append({"event": "push_executed", "result": result, "world": self.state.observe()})
            if self.state.evaluate(goal).satisfied:
                return
        raise RuntimeError("topple_retry_limit_exceeded")

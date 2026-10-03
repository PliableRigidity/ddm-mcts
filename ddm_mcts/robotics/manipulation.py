"""Explicit physical pickup stages, unique traces and independent verification.

This deterministic geometry/contact primitive is not an MCTS replacement. V3
planning remains unchanged; known-object pickup intentionally needs no Qwen.
"""

import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import mujoco
import numpy as np

from .contacts import ContactInspector, ContactRules
from .grasp import ParallelJawGraspGenerator, relative_position, verify_grasp, verify_lift
from .gripper import PandaGripper
from .pose_control import PoseTarget


class PickupStage(StrEnum):
    SELECT_TARGET = "select_target"
    COMPUTE_GRASP = "compute_grasp"
    SAFE_TRANSIT = "safe_transit"
    MOVE_TO_PREGRASP = "move_to_pregrasp"
    OPEN_GRIPPER = "open_gripper"
    VERIFY_PREGRASP = "verify_pregrasp"
    APPROACH_GRASP = "approach_grasp"
    VERIFY_GRASP_POSE = "verify_grasp_pose"
    CLOSE_GRIPPER = "close_gripper"
    VERIFY_GRASP = "verify_grasp"
    LIFT = "lift"
    HOLD = "hold"
    VERIFY_LIFT = "verify_lift"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True)
class PickupTask:
    target: str
    pregrasp_offset: float = 0.10
    lift_distance: float = 0.10
    hold_seconds: float = 1.0

    def __post_init__(self):
        if self.target not in ("cube", "cylinder") or not np.isfinite((self.pregrasp_offset, self.lift_distance, self.hold_seconds)).all():
            raise ValueError("pickup requires cube/cylinder and finite bounds")
        if self.pregrasp_offset < 0.06 or not 0.05 <= self.lift_distance <= 0.15 or self.hold_seconds < 0.5:
            raise ValueError("unsafe pregrasp/lift/hold bounds")


@dataclass(frozen=True)
class PickupResult:
    success: bool
    folder: Path
    trace: dict


class ManipulationAgent:
    def __init__(self, world, *, generator=None):
        self.world = world
        self.controller = world.controller
        self.gripper = PandaGripper(world.backend)
        self.contacts = ContactInspector(world.backend)
        self.generator = generator or ParallelJawGraspGenerator()
        self.stage = PickupStage.SELECT_TARGET

    def _pose_verified(self, pose):
        errors = self.controller.errors(pose)
        config = self.controller.pose_config
        if errors["position_error_m"] > config.position_tolerance or errors["orientation_error_rad"] > config.orientation_tolerance:
            raise RuntimeError(f"pose tolerance failed: {errors}")
        return errors

    def _clearance(self, target):
        model, data = self.world.backend.model, self.world.backend.data
        geom = model.geom(f"pickup_{target}_geom").id
        ids = {model.body(name).id for name in ("hand", "left_finger", "right_finger")}
        distances = [
            float(mujoco.mj_geomDistance(model, data, g, geom, 1.0, None))
            for g in range(model.ngeom)
            if model.geom_bodyid[g] in ids and model.geom_group[g] == 3
        ]
        return min(distances)

    def run(self, task: PickupTask, output_root="robotics_runs", *, visualization=None):
        root = Path(output_root).resolve()
        protected = Path(__file__).resolve().parents[2] / "results"
        if root == protected or protected in root.parents:
            raise ValueError("pickup logs cannot overwrite historical results")
        folder = root / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_pickup_" + uuid4().hex[:12])
        folder.mkdir(parents=True, exist_ok=False)
        started = perf_counter()
        initial = self.world.object_state(task.target)
        trace = {
            "run_id": folder.name,
            "task": asdict(task),
            "controller": asdict(self.controller.pose_config),
            "initial_object": asdict(initial),
            "initial_tcp": asdict(self.controller.pose()),
            "initial_qpos": self.world.backend.data.qpos.tolist(),
            "stages": [],
            "success": False,
            "physics_steps_monitored": 0,
            "minimum_contact_distance_m": None,
            "contact_violations": [],
            "scene_contact_configuration": {
                "pad_and_object_solref": [0.005, 1],
                "finger_stiffness_N_m": 400,
                "finger_damping_N_s_m": 20,
                "friction_unchanged": True,
            },
        }
        events = (folder / "events.jsonl").open("x", encoding="utf-8")

        def emit(kind, payload):
            events.write(
                json.dumps({"event": kind, "stage": str(self.stage), "sim_time": float(self.world.backend.data.time), **payload}) + "\n"
            )
            events.flush()

        def stage(value):
            self.stage = value
            row = {"stage": str(value), "sim_time": float(self.world.backend.data.time), "wall_seconds": perf_counter() - started}
            trace["stages"].append(row)
            emit("transition", row)
            print(f"Pickup {task.target}: {value}", flush=True)

        def trajectory(pose, errors):
            emit("trajectory", {"requested_pose": asdict(pose), "actual_tcp": asdict(self.controller.pose()), **errors})

        def monitor():
            contacts = self.contacts.contacts()
            allow = self.stage in (
                PickupStage.APPROACH_GRASP,
                PickupStage.VERIFY_GRASP_POSE,
                PickupStage.CLOSE_GRIPPER,
                PickupStage.VERIFY_GRASP,
                PickupStage.LIFT,
                PickupStage.HOLD,
                PickupStage.VERIFY_LIFT,
            )
            violations = ContactRules(initial.body_name, allow).violations(contacts)
            trace["physics_steps_monitored"] += 1
            for contact in contacts:
                if initial.body_name in (contact.body1, contact.body2):
                    old = trace["minimum_contact_distance_m"]
                    trace["minimum_contact_distance_m"] = contact.distance if old is None else min(old, contact.distance)
            # Log contact onset/end without dumping thousands of identical frames.
            pairs = sorted({(c.geom1, c.geom2) for c in contacts if initial.body_name in (c.body1, c.body2)})
            if pairs != monitor.previous:
                emit("contacts", {"contacts": [asdict(c) for c in contacts]})
                monitor.previous = pairs
            if violations:
                trace["contact_violations"].extend(asdict(c) for c in violations)
                raise RuntimeError(f"forbidden contact during {self.stage}: {violations[0]}")

        monitor.previous = []

        try:
            visual_scope = visualization.execution_scope() if visualization is not None else nullcontext()
            with visual_scope, self.world.backend.observe_steps(monitor):
                stage(PickupStage.SELECT_TARGET)
                stage(PickupStage.COMPUTE_GRASP)
                grasp = self.generator.generate(initial, pregrasp_offset=task.pregrasp_offset, lift_distance=task.lift_distance)
                trace["grasp"] = asdict(grasp)
                if visualization is not None:
                    visualization.target(grasp.pregrasp_pose.position)
                stage(PickupStage.SAFE_TRANSIT)
                current = self.controller.pose()
                height = max(current.position[2], grasp.pregrasp_pose.position[2]) + 0.04
                self.controller.move_to(
                    PoseTarget((current.position[0], current.position[1], height), current.quaternion), record=trajectory
                )
                stage(PickupStage.MOVE_TO_PREGRASP)
                # Rotate and translate only at transit clearance, then descend.
                self.controller.move_to(
                    PoseTarget((grasp.pregrasp_pose.position[0], grasp.pregrasp_pose.position[1], height), grasp.pregrasp_pose.quaternion),
                    record=trajectory,
                )
                self.controller.move_to(grasp.pregrasp_pose, record=trajectory)
                stage(PickupStage.OPEN_GRIPPER)
                trace["open_gripper"] = asdict(self.gripper.open())
                stage(PickupStage.VERIFY_PREGRASP)
                trace["pregrasp_errors"] = self._pose_verified(grasp.pregrasp_pose)
                trace["pregrasp_clearance_m"] = self._clearance(task.target)
                if self.gripper.get_width() < grasp.finger_width or trace["pregrasp_clearance_m"] < 0.02:
                    raise RuntimeError("pregrasp clearance or open width insufficient")
                stage(PickupStage.APPROACH_GRASP)
                if visualization is not None:
                    visualization.target(grasp.tcp_pose.position)
                self.controller.move_to(grasp.tcp_pose, record=trajectory)
                stage(PickupStage.VERIFY_GRASP_POSE)
                trace["grasp_pose_errors"] = self._pose_verified(grasp.tcp_pose)
                stage(PickupStage.CLOSE_GRIPPER)
                trace["closed_gripper"] = asdict(self.gripper.close())
                stage(PickupStage.VERIFY_GRASP)
                verification = verify_grasp(self.controller, self.gripper, self.contacts, self.world.object_state(task.target))
                trace["grasp_verification"] = asdict(verification)
                if not verification.success:
                    raise RuntimeError(verification.reason)
                relative = relative_position(self.controller, self.world.object_state(task.target))
                trace["initial_object_tcp_transform"] = {
                    "position": relative,
                    "rotation_matrix": (self.controller.pose().rotation.T @ self.world.object_state(task.target).pose.rotation).tolist(),
                }
                stage(PickupStage.LIFT)
                if visualization is not None:
                    visualization.target(grasp.lift_pose.position)
                self.controller.move_to(grasp.lift_pose, record=trajectory)
                trace["lift_pose_errors"] = self._pose_verified(grasp.lift_pose)
                stage(PickupStage.HOLD)
                # Hold checks every block; a transient lost grasp cannot pass at the end.
                hold = []
                cycles = int(np.ceil(task.hold_seconds / (self.world.backend.physics_steps * self.world.backend.model.opt.timestep)))
                for _ in range(cycles):
                    self.world.backend.advance()
                    check = verify_lift(
                        initial,
                        self.world.object_state(task.target),
                        relative,
                        self.controller,
                        self.gripper,
                        self.contacts,
                        lift_distance=task.lift_distance,
                    )
                    hold.append(asdict(check))
                    if not check.success:
                        raise RuntimeError(check.reason)
                trace["hold_checks"] = hold
                stage(PickupStage.VERIFY_LIFT)
                trace["lift_verification"] = asdict(check)
                stage(PickupStage.SUCCESS)
                trace["success"] = True
        except (Exception, KeyboardInterrupt) as exc:
            trace["failure_stage"] = str(self.stage)
            trace["reason"] = f"{type(exc).__name__}: {exc}"
            stage(PickupStage.FAILED)
            if isinstance(exc, KeyboardInterrupt):
                raise  # finally persists the failure trace; viewer context closes.
        finally:
            trace["final_object"] = asdict(self.world.object_state(task.target))
            trace["final_tcp"] = asdict(self.controller.pose())
            trace["final_object_tcp_transform"] = {
                "position": relative_position(self.controller, self.world.object_state(task.target)),
                "rotation_matrix": (self.controller.pose().rotation.T @ self.world.object_state(task.target).pose.rotation).tolist(),
            }
            trace["final_gripper"] = asdict(self.gripper.state())
            trace["final_qpos"] = self.world.backend.data.qpos.tolist()
            trace["elapsed_seconds"] = perf_counter() - started
            trace["final_contacts"] = [asdict(c) for c in self.contacts.contacts()]
            events.close()
            (folder / "summary.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
        return PickupResult(trace["success"], folder, trace)

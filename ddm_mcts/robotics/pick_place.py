"""Persistent deterministic pick/place operations built on the frozen pickup stack."""

import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import mujoco
import numpy as np

from .manipulation import ManipulationAgent, PickupTask
from .placement import NextTo, PlacementTarget, PlacementWorkspace, compose, elevated, measured_grasp, next_to_target, placement_poses
from .placement_verification import PlacementContactRules, verify_placement, verify_retention
from .pose_control import PoseTarget


@dataclass(frozen=True)
class PickPlace:
    target: str
    destination: PlacementTarget | NextTo

    def __post_init__(self):
        if self.target not in ("cube", "cylinder") or not isinstance(self.destination, (PlacementTarget, NextTo)):
            raise ValueError("pick/place requires cube/cylinder and a placement destination")


@dataclass(frozen=True)
class ManipulationPlan:
    operations: tuple[PickPlace, ...]

    def __post_init__(self):
        if not self.operations or not all(isinstance(op, PickPlace) for op in self.operations):
            raise ValueError("nonempty manipulation operation list required")
        object.__setattr__(self, "operations", tuple(self.operations))


@dataclass(frozen=True)
class PickPlaceConfig:
    max_retries: int = 1
    settle_seconds: float = 1.0
    preplace_clearance: float = 0.06

    def __post_init__(self):
        if (
            self.max_retries not in (0, 1)
            or not np.isfinite((self.settle_seconds, self.preplace_clearance)).all()
            or self.settle_seconds < 0.5
            or not 0.04 <= self.preplace_clearance <= 0.12
        ):
            raise ValueError("bounded retry/settling/clearance configuration required")


@dataclass(frozen=True)
class ManipulationResult:
    success: bool
    folder: Path
    trace: dict


class OperationFailure(RuntimeError):
    pass


class PickPlaceAgent:
    def __init__(self, world, *, pickup=None, workspace=None, config=None):
        self.world = world
        self.pickup = pickup or ManipulationAgent(world)
        self.controller = world.controller
        self.gripper = self.pickup.gripper
        self.contacts = self.pickup.contacts
        self.workspace = workspace or PlacementWorkspace()
        self.config = config or PickPlaceConfig()
        self.phase = "idle"

    def _scene(self):
        return {label: asdict(self.world.object_state(label)) for label in ("cube", "cylinder")}

    def _state(self):
        return {
            "integration_state": list(self.world.backend.snapshot().values),
            "qpos": self.world.backend.data.qpos.tolist(),
            "qvel": self.world.backend.data.qvel.tolist(),
            "time": float(self.world.backend.data.time),
            "objects": self._scene(),
        }

    def _scene_checks(self):
        checks = {}
        for label in ("cube", "cylinder"):
            obj = self.world.object_state(label)
            velocity = np.empty(6)
            backend = self.world.backend
            mujoco.mj_objectVelocity(
                backend.model, backend.data, mujoco.mjtObj.mjOBJ_BODY, backend.model.body(obj.body_name).id, velocity, 0
            )
            checks[label] = {
                "supported": self.contacts.touching(obj.body_name, "support_table"),
                "released": not any(self.contacts.touching(obj.body_name, finger) for finger in ("left_finger", "right_finger")),
                "linear_velocity": float(np.linalg.norm(velocity[3:])),
                "angular_velocity": float(np.linalg.norm(velocity[:3])),
                "upright": bool(obj.pose.rotation[2, 2] > np.cos(0.1)),
            }
            c = checks[label]
            c["success"] = (
                c["supported"] and c["released"] and c["upright"] and c["linear_velocity"] < 0.01 and c["angular_velocity"] < 0.10
            )
        return checks

    def _destination(self, operation):
        obj = self.world.object_state(operation.target)
        obstacles = [self.world.object_state(label) for label in ("cube", "cylinder") if label != operation.target]
        if isinstance(operation.destination, NextTo):
            reference = self.world.object_state(operation.destination.reference)
            return next_to_target(obj, reference, operation.destination, self.workspace, obstacles)
        return self.workspace.validate(operation.destination, obj, obstacles), None

    def pick_and_place(self, target, destination, output_root="robotics_runs", *, visualization=None):
        return self.run_manipulation_plan(ManipulationPlan((PickPlace(target, destination),)), output_root, visualization=visualization)

    def run_manipulation_plan(self, plan, output_root="robotics_runs", *, visualization=None):
        root = Path(output_root).resolve()
        protected = Path(__file__).resolve().parents[2] / "results"
        if root == protected or protected in root.parents:
            raise ValueError("manipulation traces cannot overwrite historical results")
        folder = root / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "_pick_place_" + uuid4().hex[:12])
        folder.mkdir(parents=True, exist_ok=False)
        started = perf_counter()
        trace = {
            "run_id": folder.name,
            "plan": asdict(plan),
            "config": asdict(self.config),
            "initial_state": self._state(),
            "operations": [],
            "success": False,
            "reset_count": 0,
        }
        with (folder / "events.jsonl").open("x") as events:

            def emit(event, **data):
                events.write(
                    json.dumps({"event": event, "phase": self.phase, "sim_time": float(self.world.backend.data.time), **data}) + "\n"
                )
                events.flush()

            for index, op in enumerate(plan.operations):
                print(f"OPERATION {index + 1}/{len(plan.operations)}: {op.target} -> {op.destination}", flush=True)
                row = {
                    "target": op.target,
                    "destination": asdict(op.destination),
                    "start_state": self._state(),
                    "recovery": [],
                    "success": False,
                    "physics_substeps": 0,
                    "intended_contacts": 0,
                    "forbidden_contacts": [],
                    "maximum_penetration_m": 0,
                    "transport": {
                        "path_length_m": 0,
                        "minimum_table_clearance_m": None,
                        "maximum_position_drift_m": 0,
                        "maximum_rotation_drift_rad": 0,
                    },
                }
                trace["operations"].append(row)
                op_started = perf_counter()

                def account(row=row, op=op):
                    row["physics_substeps"] += 1
                    cs = self.contacts.contacts()
                    row["intended_contacts"] += sum(
                        c.involves(f"pickup_{op.target}", b) for c in cs for b in ("left_finger", "right_finger", "support_table")
                    )
                    row["maximum_penetration_m"] = max(row["maximum_penetration_m"], max((max(0, -c.distance) for c in cs), default=0))

                try:
                    with self.world.backend.observe_steps(account):
                        self._operation(op, row, folder, visualization, emit)
                    row["success"] = True
                except (Exception, KeyboardInterrupt) as exc:
                    row["failure"] = {
                        "phase": self.phase,
                        "reason": str(exc),
                        "robot_pose": asdict(self.controller.pose()),
                        "scene": self._scene(),
                        "gripper": asdict(self.gripper.state()),
                        "contacts": [asdict(c) for c in self.contacts.contacts()],
                    }
                    emit("failure", **row["failure"])
                    if isinstance(exc, KeyboardInterrupt):
                        raise
                finally:
                    row["end_state"] = self._state()
                    row["duration_seconds"] = perf_counter() - op_started
                    trace["final_state"] = self._state()
                    trace["duration_seconds"] = perf_counter() - started
                    (folder / "summary.json").write_text(json.dumps(trace, indent=2))
                if not row["success"]:
                    break
                print(f"OPERATION {index + 1} COMPLETE", flush=True)
            trace["success"] = len(trace["operations"]) == len(plan.operations) and all(r["success"] for r in trace["operations"])
            trace["final_scene_verification"] = self._scene_checks()
            trace["success"] = trace["success"] and all(c["success"] for c in trace["final_scene_verification"].values())
            trace["state_continuity"] = all(
                a["end_state"] == b["start_state"] for a, b in zip(trace["operations"], trace["operations"][1:], strict=False)
            )
            (folder / "summary.json").write_text(json.dumps(trace, indent=2))
        print("PLAN COMPLETE" if trace["success"] else "PLAN FAILED", flush=True)
        return ManipulationResult(trace["success"], folder, trace)

    def _stage(self, phase, emit):
        self.phase = phase
        print(phase.upper(), flush=True)
        emit("transition")

    def _pickup(self, operation, row, folder, visual, emit):
        recoverable = {"move_to_pregrasp", "verify_pregrasp", "verify_grasp_pose", "verify_grasp"}
        for attempt in range(self.config.max_retries + 1):
            self._stage("pickup", emit)
            result = self.pickup.run(PickupTask(operation.target), folder, visualization=visual)
            row.setdefault("pickup_attempts", []).append({"folder": str(result.folder), "trace": result.trace})
            row["forbidden_contacts"].extend(result.trace.get("contact_violations", []))
            if result.success:
                return result
            reason = result.trace.get("reason", "pickup failed")
            if (
                result.trace.get("contact_violations")
                or result.trace.get("failure_stage") not in recoverable
                or attempt == self.config.max_retries
            ):
                raise OperationFailure(reason)
            self._stage("recovery", emit)
            recovery = {"attempt": attempt + 1, "trigger": reason}
            row["recovery"].append(recovery)
            scope = visual.execution_scope() if visual else nullcontext()

            def recovery_monitor():
                bad = PlacementContactRules(f"pickup_{operation.target}", "recovery").violations(self.contacts.contacts())
                if bad:
                    row["forbidden_contacts"].extend(asdict(c) for c in bad)
                    raise OperationFailure("forbidden_contact_during_recovery")

            with scope, self.world.backend.observe_steps(recovery_monitor):
                # Opening first avoids lifting an unverified partial grasp.
                self.gripper.open()
                current = self.controller.pose()
                self.controller.move_to(elevated(current, current.position[2] + 0.10))
            recovery["result"] = "retreated_and_reopened"
            emit("recovery", **recovery)

    def _operation(self, operation, row, folder, visual, emit):
        self._stage("validate_destination", emit)
        target, direction = self._destination(operation)
        row["placement_target"] = asdict(target)
        row["relation_direction"] = direction
        pickup = self._pickup(operation, row, folder, visual, emit)
        obj = self.world.object_state(operation.target)
        if obj.shape == "cylinder":
            # Axially symmetric upright placement preserves measured yaw rather
            # than twisting a radial contact grasp toward an arbitrary yaw=0.
            yaw = np.arctan2(obj.pose.rotation[1, 0], obj.pose.rotation[0, 0])
            target = replace(target, quaternion=(float(np.cos(yaw / 2)), 0, 0, float(np.sin(yaw / 2))))
            row["placement_target"] = asdict(target)
        relative = measured_grasp(self.controller.pose(), obj.pose)
        row["tcp_object_transform"] = asdict(relative)
        row["grasp_transform_before_lift"] = pickup.trace["initial_object_tcp_transform"]
        obstacles = [self.world.object_state(label) for label in ("cube", "cylinder") if label != operation.target]
        poses = placement_poses(self.controller.pose(), obj, target, relative, self.workspace, obstacles, self.config.preplace_clearance)
        row["placement_poses"] = asdict(poses)
        row["transport"]["transport_height_m"] = poses.transport_height
        previous = np.asarray(self.controller.pose().position)
        last_bilateral_force = float(self.world.backend.data.time)

        def monitor():
            nonlocal previous, last_bilateral_force
            cs = self.contacts.contacts()
            bad = PlacementContactRules(obj.body_name, self.phase).violations(cs)
            if bad:
                row["forbidden_contacts"].extend(asdict(c) for c in bad)
                raise OperationFailure(f"forbidden_contact: {bad[0]}")
            current = self.world.object_state(operation.target)
            tcp = self.controller.pose()
            if self.phase in ("transport", "preplace", "descend"):
                held = verify_retention(tcp, current, relative, self.gripper.get_width(), cs)
                tr = row["transport"]
                tr["maximum_position_drift_m"] = max(tr["maximum_position_drift_m"], held.position_drift)
                tr["maximum_rotation_drift_rad"] = max(tr["maximum_rotation_drift_rad"], held.rotation_drift)
                tr["maximum_full_rotation_drift_rad"] = max(tr.get("maximum_full_rotation_drift_rad", 0), held.full_rotation_drift)
                supported_descent = self.phase == "descend" and self.contacts.touching(obj.body_name, "support_table")
                now = float(self.world.backend.data.time)
                if held.left_contact and held.right_contact:
                    last_bilateral_force = now
                force_gap = now - last_bilateral_force
                geometric_contact = all(
                    any(c.involves(obj.body_name, name) and c.distance <= 0.0005 for c in cs) for name in ("left_finger", "right_finger")
                )
                # Contact solver forces can unload briefly while both pads still touch.
                # Accept at most 20 ms, with bilateral geometry and tight retention.
                transient_unloading = geometric_contact and force_gap <= 0.020 and 0.005 < held.width < 0.075
                tr["maximum_bilateral_force_gap_seconds"] = max(tr.get("maximum_bilateral_force_gap_seconds", 0), force_gap)
                if not held.success and not (
                    (supported_descent or transient_unloading) and held.position_drift < 0.015 and held.rotation_drift < 0.35
                ):
                    row["retention_failure"] = asdict(held)
                    raise OperationFailure(held.reason)
                if self.phase == "transport":
                    p = np.asarray(tcp.position)
                    tr["path_length_m"] += float(np.linalg.norm(p - previous))
                    previous = p
                    half = np.asarray(current.dimensions) / 2
                    bottom = current.pose.position[2] - float(np.dot(np.abs(current.pose.rotation[2]), half))
                    clearance = bottom - self.workspace.surface_z
                    old = tr["minimum_table_clearance_m"]
                    tr["minimum_table_clearance_m"] = clearance if old is None else min(old, clearance)
                    if clearance < 0.015:
                        raise OperationFailure("transport_table_clearance")

        def trajectory(pose, errors):
            emit(
                "trajectory",
                requested=asdict(pose),
                actual=asdict(self.controller.pose()),
                gripper=asdict(self.gripper.state()),
                contacts=[asdict(c) for c in self.contacts.contacts() if obj.body_name in (c.body1, c.body2)],
                tcp_object=asdict(measured_grasp(self.controller.pose(), self.world.object_state(operation.target).pose)),
                **errors,
            )

        def move(pose):
            if visual:
                visual.target(pose.position)
            self.controller.move_to(pose, linear_step=0.001, angular_step=0.01, record=trajectory)
            return self.pickup._pose_verified(pose)

        scope = visual.execution_scope() if visual else nullcontext()
        with scope, self.world.backend.observe_steps(monitor):
            self._stage("transport", emit)
            for waypoint in poses.transport:
                move(waypoint)
            row["transport"]["grasp_retained"] = True
            self._stage("preplace", emit)
            for attempt in range(self.config.max_retries + 1):
                try:
                    row["preplace_errors"] = move(poses.preplace)
                    break
                except RuntimeError as exc:
                    if (
                        isinstance(exc, OperationFailure)
                        or attempt == self.config.max_retries
                        or len(row["recovery"]) >= self.config.max_retries
                    ):
                        raise
                    row["recovery"].append({"attempt": attempt + 1, "trigger": "preplace_pose", "result": "repeating_bounded_pose_command"})
                    emit("recovery", **row["recovery"][-1])
            expected = compose(poses.preplace, relative)
            actual = self.world.object_state(operation.target)
            error = float(np.linalg.norm(np.asarray(actual.pose.position) - expected.position))
            clearance = actual.pose.position[2] - actual.dimensions[2] / 2 - self.workspace.surface_z
            distances = [
                float(
                    mujoco.mj_geomDistance(
                        self.world.backend.model,
                        self.world.backend.data,
                        self.world.backend.model.geom(f"pickup_{operation.target}_geom").id,
                        self.world.backend.model.geom(f"pickup_{o.label}_geom").id,
                        1.0,
                        None,
                    )
                )
                for o in obstacles
            ]
            row["preplace_verification"] = {
                "predicted_object_error_m": error,
                "table_clearance_m": clearance,
                "reference_clearance_m": min(distances, default=1.0),
            }
            if error >= 0.015 or clearance < 0.025 or min(distances, default=1.0) < 0.015:
                raise OperationFailure("preplace_geometry_verification_failed")
            self._stage("descend", emit)
            self._descend(poses.place, obj.body_name, row, emit, visual)
            self._stage("release", emit)
            row["release_gripper"] = asdict(self.gripper.open())
            if self.contacts.touching(obj.body_name, "left_finger") or self.contacts.touching(obj.body_name, "right_finger"):
                raise OperationFailure("release_failed: finger contact persists")
            self._stage("retreat", emit)
            row["retreat_errors"] = move(poses.retreat)
            self._stage("settle", emit)
            row["settling"] = self._settle(operation.target, target)
            self._stage("verify_placement", emit)
            row["placement_verification"] = row["settling"]["verification"]
            if not row["placement_verification"]["success"]:
                raise OperationFailure("placement_verification: " + row["placement_verification"]["reason"])
            if isinstance(operation.destination, NextTo):
                from .placement import verify_next_to

                self._stage("verify_relation", emit)
                relation = verify_next_to(
                    self.world.object_state(operation.target),
                    self.world.object_state(operation.destination.reference),
                    operation.destination.gap,
                )
                row["relation"] = asdict(relation)
                row["relation"]["reference_supported"] = self.contacts.touching(
                    f"pickup_{operation.destination.reference}", "support_table"
                )
                if not relation.success or not row["relation"]["reference_supported"]:
                    raise OperationFailure("relation_verification_failed")
            row["final_object"] = asdict(self.world.object_state(operation.target))
            row["final_tcp"] = asdict(self.controller.pose())
            self._stage("success", emit)

    def _descend(self, place, body, row, emit, visual):
        # Stop at actual support, including a bounded 2 mm compliance allowance.
        current = self.controller.pose()
        bottom = place.position[2] - 0.002
        count = max(1, int(np.ceil((current.position[2] - bottom) / 0.001)))
        for z in np.linspace(current.position[2], bottom, count + 1)[1:]:
            waypoint = PoseTarget((place.position[0], place.position[1], float(z)), place.quaternion)
            if visual:
                visual.target(waypoint.position)
            errors = self.controller.execute_pose(waypoint)
            emit("descent", tcp=asdict(self.controller.pose()), **errors)
            if self.contacts.touching(body, "support_table"):
                row["support_contact"] = {
                    "tcp": asdict(self.controller.pose()),
                    "contacts": [asdict(c) for c in self.contacts.contacts() if c.involves(body, "support_table")],
                    "sim_time": float(self.world.backend.data.time),
                }
                return
        raise OperationFailure("support_contact_not_established")

    def _settle(self, label, target):
        backend = self.world.backend
        count = int(np.ceil(self.config.settle_seconds / (backend.physics_steps * backend.model.opt.timestep)))
        samples = []
        for _ in range(count):
            backend.advance()
            obj = self.world.object_state(label)
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(
                backend.model, backend.data, mujoco.mjtObj.mjOBJ_BODY, backend.model.body(obj.body_name).id, velocity, 0
            )
            samples.append((obj, float(np.linalg.norm(velocity[3:])), float(np.linalg.norm(velocity[:3])), self.contacts.contacts()))
        # Stability must hold through the final half of the settling interval.
        tail = samples[len(samples) // 2 :]
        change = max(float(np.linalg.norm(np.asarray(s[0].pose.position) - tail[0][0].pose.position)) for s in tail)
        lin, ang = max(s[1] for s in tail), max(s[2] for s in tail)
        check = verify_placement(samples[-1][0], target, self.contacts.contacts(), lin, ang, pose_change=change)
        failures = [asdict(verify_placement(s[0], target, s[3], s[1], s[2], pose_change=change)) for s in tail]
        if any(not c["success"] for c in failures):
            check = replace(check, success=False, stable=False, reason=check.reason or "settling_window_unstable")
        return {
            "stable_window_checks": len(tail),
            "duration_seconds": count * backend.physics_steps * backend.model.opt.timestep,
            "pose_change_m": change,
            "verification": asdict(check),
        }


def rearrangement_plan():
    return ManipulationPlan((PickPlace("cylinder", NextTo("cube")), PickPlace("cube", PlacementTarget((0.46, 0.10, 0.37)))))

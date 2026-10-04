"""Reusable actuation/contact interactions, with separate live/search accounting."""

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from enum import StrEnum

import mujoco
import numpy as np

from .contacts import ContactInspector, ContactRules
from .gripper import PandaGripper
from .manipulation import PickupTask
from .physical_predicates import PhysicalState
from .pick_place import PickPlaceAgent
from .placement import elevated, measured_grasp, placement_tcp
from .placement_verification import verify_retention
from .pose_control import PoseTarget


class Primitive(StrEnum):
    MOVE_TCP = "move_tcp"
    OPEN_GRIPPER = "open_gripper"
    CLOSE_GRIPPER = "close_gripper"
    GRASP = "grasp"
    RELEASE = "release"
    MOVE_HELD_OBJECT = "move_held_object"
    CONTACT = "contact"
    PUSH = "push"
    WAIT = "wait"
    OBSERVE = "observe"
    VERIFY = "verify"


@dataclass(frozen=True)
class Push:
    target: str
    contact_point: tuple[float, float, float]
    direction: tuple[float, float, float]
    distance: float
    speed: float = 0.02

    def __post_init__(self):
        if (
            self.target not in ("cube", "cylinder")
            or np.asarray(self.contact_point).shape != (3,)
            or np.asarray(self.direction).shape != (3,)
        ):
            raise ValueError("known object and XYZ contact/direction required")
        if not np.isfinite((*self.contact_point, *self.direction, self.distance, self.speed)).all():
            raise ValueError("finite interaction parameters required")
        if not 0 < self.distance <= 0.12 or not 0.005 <= self.speed <= 0.1 or abs(np.linalg.norm(self.direction) - 1) > 1e-6:
            raise ValueError("bounded distance/speed and unit push direction required")

    @property
    def action_id(self):
        return f"PUSH:{self.target}:{self.direction}:{self.contact_point[2]:.4f}:{self.distance:.3f}"

    def __str__(self):
        return self.action_id


def pusher_tip_offset(world):
    """Distal collision-geometry extent beyond the TCP, in its local +Z axis."""
    model, data = world.backend.model, world.backend.data
    tcp = world.controller.pose()
    distal = []
    for geom in range(model.ngeom):
        if model.body(int(model.geom_bodyid[geom])).name not in ("left_finger", "right_finger") or not model.geom_contype[geom]:
            continue
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            vertices = model.mesh_vert[start : start + count]
        elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
            vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * model.geom_size[geom]
        else:
            continue
        world_vertices = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        local = (world_vertices - tcp.position) @ tcp.rotation
        distal.append(float(local[:, 2].max()))
    if not distal:
        raise ValueError("no calibrated physical finger collision geometry")
    return max(distal)


def push_candidates(world, label):
    obj = world.object_state(label)
    center = np.asarray(obj.pose.position)
    bottom = center[2] - obj.dimensions[2] / 2
    choices = []
    for sign in (-1, 1):
        for fraction, distance in ((0.30, 0.015), (0.95, 0.05)):
            point = center.copy()
            point[0] -= sign * obj.dimensions[0] / 2
            point[2] = bottom + fraction * obj.dimensions[2]
            end = center[0] + sign * distance
            if 0.365 <= end <= 0.675:
                choices.append(Push(label, tuple(point), (sign, 0, 0), distance, speed=0.01))
    return tuple(choices)


class InteractionExecutor:
    def __init__(self, world, *, state=None, output="robotics_runs", visualization=None):
        self.world = world
        self.state = state or PhysicalState(world)
        self.output = output
        self.visual = visualization
        self.contacts = ContactInspector(world.backend)
        self.gripper = PandaGripper(world.backend)
        self.controller = world.controller
        self.counts = {"substeps": 0, "intended_contacts": 0, "forbidden_contacts": 0, "maximum_penetration": 0.0}
        self.phase = "transit"
        self.target = None
        self.support = None
        self.events = []
        self.last_force_time = 0.0

    def account(self):
        cs = self.contacts.contacts()
        self.counts["substeps"] += 1
        self.counts["intended_contacts"] += len(cs)
        self.counts["maximum_penetration"] = max(self.counts["maximum_penetration"], max((-c.distance for c in cs), default=0))

    def execute(self, primitive, **parameters):
        """Public primitive boundary; language supplies goals, not servo values."""
        primitive = Primitive(primitive)
        if primitive == Primitive.OBSERVE:
            return self.state.observe()
        if primitive == Primitive.VERIFY:
            return self.state.evaluate(parameters["goal"])
        if primitive in (Primitive.OPEN_GRIPPER, Primitive.CLOSE_GRIPPER):
            if primitive == Primitive.OPEN_GRIPPER and self.phase == "carry":
                raise RuntimeError("held_object_requires_supported_release")
            with self.scope(), self.world.backend.observe_steps(self.monitor):
                return self.gripper.open() if primitive == Primitive.OPEN_GRIPPER else self.gripper.close()
        functions = {
            Primitive.MOVE_TCP: self.move_tcp,
            Primitive.GRASP: self.grasp,
            Primitive.RELEASE: self.release,
            Primitive.MOVE_HELD_OBJECT: self.move_held_object,
            Primitive.PUSH: self.push,
            Primitive.WAIT: self.wait,
        }
        if primitive == Primitive.CONTACT:
            action = parameters["action"]
            from dataclasses import replace

            return self.push(replace(action, distance=0.001))
        return functions[primitive](**parameters)

    def monitor(self):
        self.account()
        cs = self.contacts.contacts()
        target_body = f"pickup_{self.target}" if self.target else ""
        members = self.state.groups.get("tower", ()) if self.phase == "push" else ()
        intended_bodies = {target_body} | {f"pickup_{label}" for label in members}
        bad = []
        for contact in cs:
            body = next((b for b in (contact.body1, contact.body2) if b in intended_bodies), target_body)
            bad.extend(ContactRules(body, self.phase in ("carry", "descend", "release", "push"), 0.003).violations((contact,)))
        for c in cs:
            pair = {c.body1, c.body2}
            if c.distance < -0.003:
                bad.append(c)
            if target_body in pair and self.phase in ("carry", "push", "descend"):
                other = next((b for b in pair if b != target_body), "")
                allowed_support = (self.phase == "descend" and other == self.support) or (self.phase == "push" and other in intended_bodies)
                if other.startswith("pickup_") and not allowed_support:
                    bad.append(c)
                if other == "support_table" and self.phase == "carry":
                    bad.append(c)
        if bad:
            self.counts["forbidden_contacts"] += len(bad)
            raise RuntimeError(f"forbidden_contact: {bad[0]}")
        if self.phase in ("carry", "descend"):
            obj = self.world.object_state(self.target)
            original = self.state.held_reference[self.target]
            held = verify_retention(self.controller.pose(), obj, original, self.gripper.get_width(), cs)
            self.counts["maximum_position_drift"] = max(self.counts.get("maximum_position_drift", 0), held.position_drift)
            self.counts["maximum_rotation_drift"] = max(self.counts.get("maximum_rotation_drift", 0), held.rotation_drift)
            now = float(self.world.backend.data.time)
            if held.left_contact and held.right_contact:
                self.last_force_time = now
            geometry = all(any(c.involves(obj.body_name, f) and c.distance <= 0.0005 for c in cs) for f in ("left_finger", "right_finger"))
            unloaded = geometry and now - self.last_force_time <= 0.02 and 0.005 < held.width < 0.075
            supported = self.phase == "descend" and self.contacts.touching(obj.body_name, self.support)
            if not held.success and not ((unloaded or supported) and held.position_drift < 0.015 and held.rotation_drift < 0.35):
                raise RuntimeError("grasp_lost_during_transport")

    def scope(self):
        return self.visual.execution_scope() if self.visual else nullcontext()

    def move_tcp(self, pose, *, step=0.001):
        if self.visual:
            self.visual.target(pose.position)
        for attempt in range(2):
            result = self.controller.move_to(pose, linear_step=step, angular_step=0.01)
            errors = self.controller.errors(pose)
            if errors["position_error_m"] <= 0.002 and errors["orientation_error_rad"] <= 0.025:
                break
            if attempt:
                raise RuntimeError("bounded_pose_convergence_failure")
            self.events.append({"event": "pose_retry", "errors": errors})
        self.events.append({"primitive": Primitive.MOVE_TCP, "target": asdict(pose), "errors": errors})
        return result

    def wait(self, duration):
        if not np.isfinite(duration) or not 0 < duration <= 10:
            raise ValueError("bounded physical wait required")
        backend = self.world.backend
        initial = float(backend.data.time)
        original_steps = backend.physics_steps
        try:
            with self.scope(), backend.observe_steps(self.monitor):
                while backend.data.time - initial < duration - 1e-9:
                    remaining = duration - (float(backend.data.time) - initial)
                    backend.physics_steps = min(original_steps, max(1, int(np.ceil(remaining / backend.model.opt.timestep - 1e-8))))
                    backend.advance()
                    observed = self.state.observe()
                    self.events.append({"primitive": Primitive.OBSERVE, "world": observed})
        finally:
            backend.physics_steps = original_steps
        elapsed = float(backend.data.time) - initial
        self.events.append({"primitive": Primitive.WAIT, "requested": duration, "elapsed": elapsed})
        return elapsed

    def grasp(self, label):
        # Existing frozen pickup includes physical approach, closure, lift and hold verification.
        row = {"recovery": [], "forbidden_contacts": []}

        def emit(event, **parameters):
            self.events.append({"event": event, **parameters})

        try:
            with self.world.backend.observe_steps(self.account):
                result = PickPlaceAgent(self.world)._pickup(PickupTask(label), row, self.output, self.visual, emit)
        finally:
            self.events.append({"primitive": Primitive.GRASP, "attempts": row})
            self.counts["forbidden_contacts"] += len(row["forbidden_contacts"])
        self.events.append({"primitive": Primitive.GRASP, "success": result.success, "log": str(result.folder)})
        if not result.success:
            raise RuntimeError("grasp_failed: " + str(result.trace.get("failure_reason")))
        self.target = label
        self.phase = "carry"
        self.state.held_reference[label] = measured_grasp(self.controller.pose(), self.world.object_state(label).pose)
        self.last_force_time = float(self.world.backend.data.time)
        return result

    def move_held_object(self, label, desired, support):
        self.target, self.support = label, support.body
        relative = self.state.held_reference[label]
        tcp = placement_tcp(desired, relative)
        self.events.append(
            {
                "primitive": Primitive.MOVE_HELD_OBJECT,
                "object": label,
                "desired_object_pose": asdict(desired),
                "measured_tcp_object": asdict(relative),
                "required_tcp_pose": asdict(tcp),
                "support_surface": asdict(support),
            }
        )
        highest = max(
            self.world.object_state(o).pose.position[2] + self.world.object_state(o).dimensions[2] / 2 for o in ("cube", "cylinder")
        )
        height = max(self.controller.pose().position[2], highest + 0.10)
        pre = elevated(tcp, tcp.position[2] + 0.06)
        self.phase = "carry"
        with self.scope(), self.world.backend.observe_steps(self.monitor):
            self.move_tcp(elevated(self.controller.pose(), height))
            self.move_tcp(elevated(tcp, height))
            self.move_tcp(pre)
            self.phase = "descend"
            bottom = tcp.position[2] - 0.002
            for z in np.arange(self.controller.pose().position[2] - 0.001, bottom - 0.001, -0.001):
                waypoint = elevated(tcp, float(z))
                self.controller.execute_pose(waypoint)
                if self.contacts.touching(f"pickup_{label}", support.body):
                    self.events.append(
                        {
                            "primitive": Primitive.CONTACT,
                            "target": label,
                            "support": support.owner,
                            "contacts": [asdict(c) for c in self.contacts.contacts()],
                        }
                    )
                    return tcp
        raise RuntimeError("support_contact_not_established")

    def release(self, label):
        if not self.support or not self.contacts.touching(f"pickup_{label}", self.support):
            raise RuntimeError("release_requires_support")
        self.phase = "release"
        with self.scope(), self.world.backend.observe_steps(self.monitor):
            self.gripper.open()
            self.phase = "retreat"
            self.move_tcp(elevated(self.controller.pose(), self.controller.pose().position[2] + 0.10))
        self.phase = "settle"
        self.wait(1.0)
        self.events.append({"primitive": Primitive.RELEASE, "object": label})

    def push(self, action):
        self.target, self.phase = action.target, "transit"
        direction = np.asarray(action.direction)
        point = np.asarray(action.contact_point).copy()
        point[2] += pusher_tip_offset(self.world)
        # The closed Panda pad is the physical pusher. Its X half-extent is 8 mm.
        start = PoseTarget(tuple(point - direction * 0.025), (0, 1, 0, 0))
        contact_seen = False
        with self.scope(), self.world.backend.observe_steps(self.monitor):
            self.gripper.close()
            current = self.controller.pose()
            height = max(current.position[2], 0.55)
            self.move_tcp(elevated(current, height))
            self.move_tcp(elevated(start, height))
            self.move_tcp(start)
            self.phase = "push"
            increment = min(0.004, action.speed * self.world.backend.model.opt.timestep * self.world.backend.physics_steps)
            for distance in np.arange(increment, action.distance + 0.025 + increment / 2, increment):
                pose = PoseTarget(tuple(np.asarray(start.position) + direction * distance), start.quaternion)
                self.controller.execute_pose(pose)
                contact_seen |= any(
                    c.involves(f"pickup_{action.target}", finger) and c.normal_force > 0.01
                    for c in self.contacts.contacts()
                    for finger in ("left_finger", "right_finger")
                )
                tilt = np.arccos(np.clip(self.world.object_state(action.target).pose.rotation[2, 2], -1, 1))
                if tilt > np.pi / 4:
                    break
            current = self.controller.pose()
            self.move_tcp(elevated(current, current.position[2] + 0.10))
        self.phase = "settle"
        self.wait(1.0)
        result = {"action": asdict(action), "contact_seen": contact_seen}
        self.events.append({"primitive": Primitive.PUSH, **result})
        return result

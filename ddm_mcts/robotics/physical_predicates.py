"""Observed physical relationships, support geometry and interaction affordances."""

from dataclasses import dataclass

import mujoco
import numpy as np

from .contacts import ContactInspector
from .gripper import PandaGripper
from .physical_goals import Predicate
from .placement import measured_grasp, pose_from_matrix, verify_next_to
from .pose_control import PoseTarget


@dataclass(frozen=True)
class SupportSurface:
    owner: str
    body: str
    pose: PoseTarget
    normal: tuple[float, float, float]
    boundary: tuple[float, float]
    circular: bool = False
    margin: float = 0.005

    def contains_com(self, position):
        local = self.pose.rotation.T @ (np.asarray(position) - self.pose.position)
        if self.circular:
            return bool(np.linalg.norm(local[:2]) <= self.boundary[0] - self.margin)
        return bool(np.all(np.abs(local[:2]) <= np.asarray(self.boundary) - self.margin))


def support_surface(world, owner):
    if owner == "table":
        model, data = world.backend.model, world.backend.data
        geom = model.geom("pickup_table").id
        rotation = data.geom_xmat[geom].reshape(3, 3)
        position = data.geom_xpos[geom] + rotation[:, 2] * model.geom_size[geom, 2]
        return SupportSurface(
            owner,
            "support_table",
            pose_from_matrix(position, rotation),
            tuple(rotation[:, 2]),
            tuple(float(v) for v in model.geom_size[geom, :2]),
        )
    obj = world.object_state(owner)
    normal = obj.pose.rotation[:, 2]
    if normal[2] < 0.995:
        raise ValueError("support surface is not upright")
    position = np.asarray(obj.pose.position) + normal * obj.dimensions[2] / 2
    return SupportSurface(
        owner,
        obj.body_name,
        PoseTarget(tuple(position), obj.pose.quaternion),
        tuple(normal),
        tuple(np.asarray(obj.dimensions[:2]) / 2),
        obj.shape == "cylinder",
    )


@dataclass(frozen=True)
class PredicateResult:
    satisfied: bool
    metrics: dict

    def __post_init__(self):
        object.__setattr__(self, "satisfied", bool(self.satisfied))


class PhysicalState:
    """Fresh read-only state plus real, time-indexed settling evidence."""

    def __init__(self, world):
        self.world = world
        self.contacts = ContactInspector(world.backend)
        self.gripper = PandaGripper(world.backend)
        self.history = {label: [] for label in ("cube", "cylinder")}
        self.groups = {}
        self.held_reference = {}

    def observe(self):
        backend = self.world.backend
        objects = {}
        for label in self.history:
            obj = self.world.object_state(label)
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                backend.model, backend.data, mujoco.mjtObj.mjOBJ_BODY, backend.model.body(obj.body_name).id, velocity, 0
            )
            row = (float(backend.data.time), obj.pose, float(np.linalg.norm(velocity[3:])), float(np.linalg.norm(velocity[:3])))
            self.history[label].append(row)
            self.history[label] = [r for r in self.history[label] if r[0] >= row[0] - 2]
            objects[label] = {
                "position": obj.pose.position,
                "quaternion": obj.pose.quaternion,
                "linear_velocity": row[2],
                "angular_velocity": row[3],
            }
        from .physical_goals import PhysicalGoal

        relationships = []
        satisfied = []
        for label in objects:
            for predicate in (Predicate.HELD, Predicate.RELEASED, Predicate.UPRIGHT, Predicate.STABLE, Predicate.TOPPLED):
                result = self.evaluate(PhysicalGoal(predicate, label))
                if result.satisfied:
                    satisfied.append({"predicate": predicate, "subject": label})
            for support in ("table", *(other for other in objects if other != label)):
                for predicate in (Predicate.CONTACTING, Predicate.SUPPORTED_BY, Predicate.ON_TOP_OF):
                    if self.evaluate(PhysicalGoal(predicate, label, support)).satisfied:
                        relationships.append({"predicate": predicate, "subject": label, "reference": support})
        return {
            "time": float(backend.data.time),
            "objects": objects,
            "groups": self.groups.copy(),
            "relationships": relationships,
            "satisfied_predicates": satisfied,
            "affordances": {label: self.affordances(label) for label in objects},
        }

    def evaluate(self, goal):
        if goal.subject == "tower":
            members = self.groups.get("tower")
            if not members or goal.predicate != Predicate.TOPPLED:
                return PredicateResult(False, {"reason": "tower requires a previously verified support group"})
            upper, lower = members
            from .physical_goals import PhysicalGoal

            stack = self.evaluate(PhysicalGoal(Predicate.ON_TOP_OF, upper, lower))
            tilt = max(np.arccos(np.clip(self.world.object_state(o).pose.rotation[2, 2], -1, 1)) for o in members)
            return PredicateResult(
                bool(not stack.satisfied and tilt > np.pi / 4), {"stack_exists": stack.satisfied, "maximum_tilt": float(tilt)}
            )
        obj = self.world.object_state(goal.subject)
        cs = self.contacts.contacts()

        def touching(body):
            return any(c.involves(obj.body_name, body) and c.normal_force > 0.01 and c.distance <= 0.0005 for c in cs)

        tilt = float(np.arccos(np.clip(obj.pose.rotation[2, 2], -1, 1)))
        fingers = [touching(f) for f in ("left_finger", "right_finger")]
        p = goal.predicate
        if p == Predicate.CONTACTING:
            body = "support_table" if goal.reference == "table" else self.world.object_state(goal.reference).body_name
            return PredicateResult(touching(body), {"support": body})
        if p in (Predicate.SUPPORTED_BY, Predicate.ON_TOP_OF):
            try:
                surface = support_surface(self.world, goal.reference)
            except ValueError as exc:
                return PredicateResult(False, {"reason": str(exc)})
            bottom = obj.pose.position[2] - float(np.abs(obj.pose.rotation[2]) @ (np.asarray(obj.dimensions) / 2))
            vertical = bottom - surface.pose.position[2]
            com = surface.contains_com(obj.pose.position)
            supported = touching(surface.body) and com and abs(vertical) < 0.003 and obj.pose.position[2] > surface.pose.position[2]
            return PredicateResult(
                bool(supported),
                {
                    "vertical_error": float(vertical),
                    "horizontal_error": float(np.linalg.norm(np.asarray(obj.pose.position[:2]) - surface.pose.position[:2])),
                    "com_inside": com,
                    "contact": touching(surface.body),
                },
            )
        if p == Predicate.UPRIGHT:
            return PredicateResult(tilt < 0.10, {"tilt": tilt})
        if p == Predicate.TOPPLED:
            return PredicateResult(tilt > np.pi / 4 and touching("support_table") and not any(fingers), {"tilt": tilt})
        if p == Predicate.RELEASED:
            return PredicateResult(not any(fingers), {"finger_contacts": fingers})
        if p == Predicate.HELD:
            relative = measured_grasp(self.world.controller.pose(), obj.pose)
            original = self.held_reference.get(obj.label)
            drift = np.linalg.norm(np.asarray(relative.position) - original.position) if original else float("inf")
            width = self.gripper.get_width()
            return PredicateResult(
                bool(all(fingers) and 0.005 < width < 0.075 and drift < 0.015 and not touching("support_table")),
                {"finger_contacts": fingers, "width": width, "relative_drift": float(drift)},
            )
        if p == Predicate.STABLE:
            now = float(self.world.backend.data.time)
            samples = [r for r in self.history[obj.label] if r[0] >= now - 0.5 - 1e-8]
            duration = samples[-1][0] - samples[0][0] if samples else 0
            drift = max((np.linalg.norm(np.asarray(r[1].position) - samples[0][1].position) for r in samples), default=float("inf"))
            lin = max((r[2] for r in samples), default=float("inf"))
            ang = max((r[3] for r in samples), default=float("inf"))
            return PredicateResult(
                bool(duration >= 0.45 and drift < 0.002 and lin < 0.01 and ang < 0.1),
                {"duration": duration, "pose_drift": float(drift), "linear_velocity": lin, "angular_velocity": ang},
            )
        if p == Predicate.NEXT_TO:
            from dataclasses import asdict

            relation = verify_next_to(obj, self.world.object_state(goal.reference))
            return PredicateResult(relation.success, asdict(relation))
        if p == Predicate.AT_LOCATION:
            error = float(np.linalg.norm(np.asarray(obj.pose.position) - goal.location))
            return PredicateResult(error < 0.01 and touching("support_table") and not any(fingers), {"position_error": error})
        raise ValueError("unknown predicate")

    def affordances(self, label):
        obj = self.world.object_state(label)
        upright = obj.pose.rotation[2, 2] > 0.995
        from .physical_goals import PhysicalGoal

        graspable = bool(upright and max(obj.dimensions[:2]) < 0.065)
        held = self.evaluate(PhysicalGoal(Predicate.HELD, label)).satisfied
        supports = []
        if graspable or held:
            supports.append("table")
            for other in ("cube", "cylinder"):
                if other != label:
                    owner = self.world.object_state(other)
                    supported = any(
                        c.involves(owner.body_name, "support_table") and c.normal_force > 0.01 for c in self.contacts.contacts()
                    )
                    if owner.pose.rotation[2, 2] > 0.995 and supported:
                        supports.append(other)
        return {
            "graspable": graspable,
            "pushable": not held,
            "contactable": True,
            "support_surface": bool(upright),
            "placeable_on": tuple(supports),
        }

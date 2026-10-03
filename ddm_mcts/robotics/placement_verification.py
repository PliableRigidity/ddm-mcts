"""Independent physical retention, support, release and placement checks."""

from dataclasses import dataclass

import numpy as np

from .contacts import ContactRules
from .placement import measured_grasp
from .pose_control import rotation_error


@dataclass(frozen=True)
class PlacementContactRules:
    target: str
    phase: str
    maximum_penetration: float = 0.003

    def violations(self, contacts):
        allow = self.phase in ("transport", "preplace", "descend", "release", "recovery")
        bad = list(ContactRules(self.target, allow, self.maximum_penetration).violations(contacts))
        for c in contacts:
            pair = {c.body1, c.body2}
            if self.target not in pair:
                continue
            other = next((b for b in pair if b != self.target), self.target)
            if other.startswith("pickup_") or (other == "support_table" and self.phase in ("transport", "preplace")):
                bad.append(c)
            if c.distance < -self.maximum_penetration and c not in bad:
                bad.append(c)
        return tuple(bad)


@dataclass(frozen=True)
class RetentionVerification:
    success: bool
    position_drift: float
    rotation_drift: float
    full_rotation_drift: float
    left_contact: bool
    right_contact: bool
    width: float
    reason: str


def verify_retention(tcp, obj, original, width, contacts, *, position_limit=0.015, rotation_limit=0.35):
    relative = measured_grasp(tcp, obj.pose)
    dp = float(np.linalg.norm(np.asarray(relative.position) - original.position))
    dr = float(np.linalg.norm(rotation_error(relative.rotation, original.rotation)))
    full_dr = dr
    if obj.shape == "cylinder":
        dr = float(np.arccos(np.clip(np.dot(relative.rotation[:, 2], original.rotation[:, 2]), -1, 1)))
    fingers = [
        any(c.involves(obj.body_name, name) and c.normal_force > 0.01 and c.distance <= 0.0005 for c in contacts)
        for name in ("left_finger", "right_finger")
    ]
    success = all(fingers) and 0.005 < width < 0.075 and dp < position_limit and dr < rotation_limit
    return RetentionVerification(success, dp, dr, full_dr, *fingers, width, "" if success else "grasp_lost_during_transport")


@dataclass(frozen=True)
class PlacementVerification:
    success: bool
    position_error: float
    orientation_error: float
    supported: bool
    released: bool
    stable: bool
    upright: bool
    linear_velocity: float
    angular_velocity: float
    reason: str


def verify_placement(obj, target, contacts, linear_velocity, angular_velocity, *, pose_change=0, forbidden=False):
    position_error = float(np.linalg.norm(np.asarray(obj.pose.position) - target.position))
    # Cylinder yaw is symmetric; upright tilt is the relevant orientation.
    orientation_error = (
        float(np.arccos(np.clip(obj.pose.rotation[2, 2], -1, 1)))
        if obj.shape == "cylinder"
        else float(np.linalg.norm(rotation_error(target.pose.rotation, obj.pose.rotation)))
    )
    supported = any(c.involves(obj.body_name, target.support_surface) and c.normal_force > 0.01 for c in contacts)
    released = not any(c.involves(obj.body_name, name) for c in contacts for name in ("left_finger", "right_finger"))
    stable = linear_velocity < 0.01 and angular_velocity < 0.10 and pose_change < 0.002
    upright = bool(obj.pose.rotation[2, 2] > np.cos(target.orientation_tolerance))
    failures = [
        name
        for name, good in (
            ("position", position_error <= target.position_tolerance),
            ("orientation", orientation_error <= target.orientation_tolerance),
            ("unsupported", supported),
            ("still_held", released),
            ("unstable", stable),
            ("tipped", upright),
            ("forbidden_contact", not forbidden),
        )
        if not good
    ]
    return PlacementVerification(
        not failures,
        position_error,
        orientation_error,
        supported,
        released,
        stable,
        upright,
        linear_velocity,
        angular_velocity,
        ",".join(failures),
    )

"""Geometry-grounded parallel-jaw grasps and independent physical verifiers."""

from dataclasses import dataclass

import numpy as np

from .pose_control import PoseTarget, rotation_error


@dataclass(frozen=True)
class GraspPose:
    object_label: str
    tcp_pose: PoseTarget
    pregrasp_pose: PoseTarget
    lift_pose: PoseTarget
    approach_direction: tuple[float, float, float]
    finger_width: float
    strategy: str


class ParallelJawGraspGenerator:
    def generate(self, obj, *, pregrasp_offset=0.10, lift_distance=0.10):
        if obj.shape not in ("box", "cylinder") or min(pregrasp_offset, lift_distance) <= 0:
            raise ValueError("supported shape and positive offsets required")
        # Top-down TCP local Z points down; fingers close along local Y.
        # Cube jaws align with its horizontal principal axes. Cylinder is yaw-invariant.
        rotation = obj.pose.rotation
        if abs(rotation[2, 2]) < 0.99:
            raise ValueError("generator requires an upright object")
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0]) if obj.shape == "box" else 0.0
        quaternion = (0.0, float(np.cos(yaw / 2)), float(np.sin(yaw / 2)), 0.0)
        center = obj.pose.position
        width = obj.dimensions[1] + 0.018
        if not 0 < width <= 0.08:
            raise ValueError("object too wide for Panda parallel jaws")
        pose = PoseTarget(center, quaternion)
        return GraspPose(
            obj.label,
            pose,
            PoseTarget((center[0], center[1], center[2] + pregrasp_offset), quaternion),
            PoseTarget((center[0], center[1], center[2] + lift_distance), quaternion),
            (0, 0, -1),
            width,
            "top-down parallel-jaw box faces" if obj.shape == "box" else "top-down radial cylinder pinch",
        )


@dataclass(frozen=True)
class GraspVerification:
    success: bool
    left_contact: bool
    right_contact: bool
    finger_width: float
    object_relative_position: tuple[float, float, float]
    reason: str


def relative_position(controller, obj):
    tcp = controller.pose()
    return tuple(float(x) for x in tcp.rotation.T @ (np.asarray(obj.pose.position) - tcp.position))


def verify_grasp(controller, gripper, inspector, obj):
    left = inspector.touching("left_finger", obj.body_name)
    right = inspector.touching("right_finger", obj.body_name)
    relative = relative_position(controller, obj)
    width = gripper.get_width()
    success = left and right and 0.005 < width < 0.075 and np.linalg.norm(relative) < 0.035
    return GraspVerification(
        bool(success),
        left,
        right,
        width,
        relative,
        "bilateral contact and object inside jaws" if success else "missing bilateral contact or invalid jaw/object geometry",
    )


@dataclass(frozen=True)
class LiftVerification:
    success: bool
    height_increase: float
    relative_drift: float
    table_contact: bool
    remained_held: bool
    reason: str
    orientation_drift_rad: float = 0.0


def verify_lift(initial_object, current_object, initial_relative, controller, gripper, inspector, *, lift_distance=0.10):
    grasp = verify_grasp(controller, gripper, inspector, current_object)
    height = current_object.pose.position[2] - initial_object.pose.position[2]
    drift = float(np.linalg.norm(np.asarray(grasp.object_relative_position) - initial_relative))
    orientation_drift = float(np.linalg.norm(rotation_error(current_object.pose.rotation, initial_object.pose.rotation)))
    supported = inspector.touching(current_object.body_name, "support_table")
    success = grasp.success and height >= lift_distance * 0.75 and drift < 0.015 and orientation_drift < 0.35 and not supported
    return LiftVerification(
        bool(success),
        height,
        drift,
        supported,
        grasp.success,
        "object lifted and held without support" if success else "object failed to lift, slipped, dropped or remains supported",
        orientation_drift,
    )

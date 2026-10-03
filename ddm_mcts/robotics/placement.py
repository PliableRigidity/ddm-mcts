"""Rigid transforms and deterministic placement geometry, independent of execution."""

from dataclasses import dataclass

import mujoco
import numpy as np

from .pose_control import PoseTarget


def pose_from_matrix(position, rotation):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(rotation).ravel())
    return PoseTarget(tuple(position), tuple(q))


def compose(first, second):
    return pose_from_matrix(np.asarray(first.position) + first.rotation @ second.position, first.rotation @ second.rotation)


def inverse(pose):
    return pose_from_matrix(-pose.rotation.T @ pose.position, pose.rotation.T)


def measured_grasp(tcp, obj):
    """T_tcp_object, measured from actual physical poses."""
    return compose(inverse(tcp), obj)


def placement_tcp(desired_object, tcp_object):
    return compose(desired_object, inverse(tcp_object))


@dataclass(frozen=True)
class PlacementTarget:
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float] = (1, 0, 0, 0)
    support_surface: str = "support_table"
    position_tolerance: float = 0.01
    orientation_tolerance: float = 0.10

    def __post_init__(self):
        PoseTarget(self.position, self.quaternion)
        if (
            not np.isfinite((self.position_tolerance, self.orientation_tolerance)).all()
            or min(self.position_tolerance, self.orientation_tolerance) <= 0
        ):
            raise ValueError("positive finite placement tolerances required")
        if self.support_surface != "support_table":
            raise ValueError("only the calibrated support_table is supported")

    @property
    def pose(self):
        return PoseTarget(self.position, self.quaternion)


@dataclass(frozen=True)
class NextTo:
    reference: str
    gap: float = 0.04

    def __post_init__(self):
        if self.reference not in ("cube", "cylinder") or not np.isfinite(self.gap) or not 0.025 <= self.gap <= 0.07:
            raise ValueError("next_to requires known reference and gap 0.025–0.07 m")


@dataclass(frozen=True)
class PlacementWorkspace:
    x: tuple[float, float] = (0.34, 0.70)
    y: tuple[float, float] = (-0.22, 0.22)
    surface_z: float = 0.35
    margin: float = 0.015

    def validate(self, target, obj, obstacles=()):
        p = target.position
        radius = np.asarray(obj.dimensions) / 2
        if any(
            p[i] - radius[i] < bounds[0] + self.margin or p[i] + radius[i] > bounds[1] - self.margin
            for i, bounds in enumerate((self.x, self.y))
        ):
            raise ValueError("invalid_destination: object outside support workspace")
        if abs(p[2] - self.surface_z - radius[2]) > 0.002:
            raise ValueError("invalid_destination: object center must rest on support surface")
        if target.pose.rotation[2, 2] < 0.995:
            raise ValueError("invalid_destination: upright object required")
        for other in obstacles:
            if np.linalg.norm(np.asarray(p[:2]) - other.pose.position[:2]) < horizontal_extent(obj) + horizontal_extent(other) + 0.015:
                raise ValueError("invalid_destination: overlaps another object")
        return target


def horizontal_extent(obj, direction=None):
    """Support-function half-extent along a horizontal direction."""
    axis = np.array([1, 0, 0]) if direction is None else np.asarray(direction)
    local = obj.pose.rotation.T @ axis
    half = np.asarray(obj.dimensions) / 2
    if obj.shape == "cylinder":
        return float(half[0] * np.linalg.norm(local[:2]) + half[2] * abs(local[2]))
    return float(np.dot(half, np.abs(local)))


def next_to_target(obj, reference, destination, workspace, obstacles=()):
    if obj.label == reference.label:
        raise ValueError("invalid_destination: cannot place next to itself")
    # Deterministic candidates; bounds/obstacles determine which is feasible.
    candidates = []
    for direction in ((1, 0, 0), (0, -1, 0), (-1, 0, 0), (0, 1, 0)):
        d = np.asarray(direction)
        # Preserve the reference's open Panda jaw sweep along world Y.
        # Nearer Y candidates with a narrow gap obstruct the next regrasp.
        if direction[1] and destination.gap < 0.06:
            continue
        offset = horizontal_extent(obj, d) + horizontal_extent(reference, d) + destination.gap
        position = np.asarray(reference.pose.position) + offset * d
        position[2] = workspace.surface_z + obj.dimensions[2] / 2
        target = PlacementTarget(tuple(position))
        try:
            workspace.validate(target, obj, obstacles)
            candidates.append((float(np.linalg.norm(np.asarray(target.position) - obj.pose.position)), target, direction))
        except ValueError:
            continue
    if candidates:
        _, target, direction = min(candidates, key=lambda candidate: candidate[0])
        return target, direction
    raise ValueError("invalid_destination: no feasible next_to location")


@dataclass(frozen=True)
class PlacementPoses:
    transport: tuple[PoseTarget, ...]
    preplace: PoseTarget
    place: PoseTarget
    retreat: PoseTarget
    transport_height: float


def elevated(pose, height):
    return PoseTarget((pose.position[0], pose.position[1], height), pose.quaternion)


def placement_poses(current_tcp, obj, target, tcp_object, workspace, obstacles=(), clearance=0.06):
    required = placement_tcp(target.pose, tcp_object)
    obstacle_top = max([workspace.surface_z] + [o.pose.position[2] + o.dimensions[2] / 2 for o in obstacles])
    # Height refers to the object center; convert via measured held transform.
    height = max(obj.pose.position[2], obstacle_top + obj.dimensions[2] / 2 + clearance)
    high_object = elevated(target.pose, height)
    high_tcp = placement_tcp(high_object, tcp_object)
    preplace = placement_tcp(elevated(target.pose, target.position[2] + clearance), tcp_object)
    transit_start = elevated(current_tcp, max(current_tcp.position[2], high_tcp.position[2]))
    return PlacementPoses((transit_start, high_tcp), preplace, required, elevated(required, required.position[2] + 0.10), height)


@dataclass(frozen=True)
class RelationVerification:
    success: bool
    center_distance: float
    surface_gap: float
    overlap: float
    allowed_gap: tuple[float, float]
    supported_geometry: bool


def verify_next_to(obj, reference, gap=0.04, tolerance=0.01):
    delta = np.asarray(obj.pose.position) - reference.pose.position
    delta[2] = 0
    distance = float(np.linalg.norm(delta))
    direction = delta / distance if distance > 1e-12 else np.array([1, 0, 0])
    surface_gap = distance - horizontal_extent(obj, direction) - horizontal_extent(reference, direction)
    allowed = (gap - tolerance, gap + tolerance)
    upright = bool(min(obj.pose.rotation[2, 2], reference.pose.rotation[2, 2]) > 0.995)
    supported = all(abs(o.pose.position[2] - o.dimensions[2] / 2 - 0.35) < 0.002 for o in (obj, reference))
    return RelationVerification(
        upright and supported and allowed[0] <= surface_gap <= allowed[1], distance, surface_gap, max(0, -surface_gap), allowed, supported
    )

"""Optional six-dimensional TCP control alongside the frozen position controller."""

from dataclasses import dataclass

import mujoco
import numpy as np

from .controller import CartesianController
from .mujoco_backend import copy_simulator_data


def rotation_matrix(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("finite nonzero wxyz quaternion required")
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, q / np.linalg.norm(q))
    return matrix.reshape(3, 3)


def rotation_error(target, current):
    """World-frame SO(3) logarithm of R_target R_current.T (radians)."""
    relative = np.asarray(target) @ np.asarray(current).T
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, relative.ravel())
    if q[0] < 0:
        q = -q
    sine = np.linalg.norm(q[1:])
    return 2 * q[1:] if sine < 1e-10 else q[1:] * (2 * np.arctan2(sine, q[0]) / sine)


@dataclass(frozen=True)
class PoseTarget:
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]  # MuJoCo wxyz; local axes -> world

    def __post_init__(self):
        if np.asarray(self.position).shape != (3,) or not np.isfinite(self.position).all():
            raise ValueError("finite XYZ required")
        rotation_matrix(self.quaternion)
        q = np.asarray(self.quaternion) / np.linalg.norm(self.quaternion)
        object.__setattr__(self, "position", tuple(float(x) for x in self.position))
        object.__setattr__(self, "quaternion", tuple(float(x) for x in q))

    @property
    def rotation(self):
        return rotation_matrix(self.quaternion)


@dataclass(frozen=True)
class PoseIKConfig:
    damping: float = 0.02
    iterations: int = 100
    max_joint_update: float = 0.06
    position_weight: float = 1.0
    orientation_weight: float = 0.3
    position_tolerance: float = 0.002
    orientation_tolerance: float = 0.025

    def __post_init__(self):
        values = (
            self.damping,
            self.max_joint_update,
            self.position_weight,
            self.orientation_weight,
            self.position_tolerance,
            self.orientation_tolerance,
        )
        if not np.isfinite(values).all() or min(values) <= 0 or self.iterations < 1:
            raise ValueError("positive finite pose IK bounds required")


class PoseController(CartesianController):
    """Uses both MuJoCo Jacobians; writes actuator commands, never live qpos."""

    def __init__(self, *args, pose_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pose_config = pose_config or PoseIKConfig()

    def _rotation(self, data):
        return (data.site_xmat[self.site] if self.site is not None else data.xmat[self.body]).reshape(3, 3)

    def pose(self):
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, self._rotation(self.backend.data).ravel())
        return PoseTarget(self.position(), tuple(q))

    def errors(self, target):
        return {
            "position_error_m": float(np.linalg.norm(np.asarray(target.position) - self.position())),
            "orientation_error_rad": float(np.linalg.norm(rotation_error(target.rotation, self._rotation(self.backend.data)))),
        }

    def execute_pose(self, target: PoseTarget):
        model, data, config = self.backend.model, self.backend.data, self.pose_config
        copy_simulator_data(model, data, self.scratch)
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        weights = np.array([config.position_weight] * 3 + [config.orientation_weight] * 3)
        for _ in range(config.iterations):
            mujoco.mj_forward(model, self.scratch)
            ep = np.asarray(target.position) - self._position(self.scratch)
            er = rotation_error(target.rotation, self._rotation(self.scratch))
            if np.linalg.norm(ep) < config.position_tolerance / 4 and np.linalg.norm(er) < config.orientation_tolerance / 4:
                break
            if self.site is not None:
                mujoco.mj_jacSite(model, self.scratch, jacp, jacr, self.site)
            else:
                mujoco.mj_jacBody(model, self.scratch, jacp, jacr, self.body)
            jac = np.vstack((jacp[:, self.dofs], jacr[:, self.dofs])) * weights[:, None]
            error = np.concatenate((ep, er)) * weights
            dq = jac.T @ np.linalg.solve(jac @ jac.T + config.damping**2 * np.eye(6), error)
            self.scratch.qpos[self.qpos] += np.clip(dq, -config.max_joint_update, config.max_joint_update)
            for joint, address in zip(self.joints, self.qpos, strict=True):
                if model.jnt_limited[joint]:
                    self.scratch.qpos[address] = np.clip(self.scratch.qpos[address], *model.jnt_range[joint])
        commands = self.scratch.qpos[self.qpos].copy()
        for index, actuator in enumerate(self.actuators):
            if model.actuator_ctrllimited[actuator]:
                commands[index] = np.clip(commands[index], *model.actuator_ctrlrange[actuator])
        data.ctrl[self.actuators] = commands
        self.backend.advance()
        return self.errors(target)

    def move_to(self, target, *, linear_step=0.004, angular_step=0.04, settle_steps=12, record=None):
        if min(linear_step, angular_step) <= 0 or not np.isfinite((linear_step, angular_step)).all():
            raise ValueError("positive finite trajectory increments required")
        start = self.pose()
        distance = np.linalg.norm(np.asarray(target.position) - start.position)
        angle = np.linalg.norm(rotation_error(target.rotation, start.rotation))
        count = max(1, int(np.ceil(max(distance / linear_step, angle / angular_step))))
        q0, q1 = np.asarray(start.quaternion), np.asarray(target.quaternion)
        if np.dot(q0, q1) < 0:
            q1 = -q1
        theta = np.arccos(np.clip(np.dot(q0, q1), -1, 1))
        for i in range(1, count + 1):
            t = i / count
            q = (1 - t) * q0 + t * q1 if theta < 1e-8 else (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.sin(theta)
            waypoint = PoseTarget(tuple((1 - t) * np.asarray(start.position) + t * np.asarray(target.position)), tuple(q))
            errors = self.execute_pose(waypoint)
            if record is not None:
                record(waypoint, errors)
        for _ in range(settle_steps):
            errors = self.execute_pose(target)
        return errors

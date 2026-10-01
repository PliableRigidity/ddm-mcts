"""Position-only damped least-squares controller for joint position actuators."""

from dataclasses import dataclass

import mujoco
import numpy as np

from .mujoco_backend import copy_simulator_data


@dataclass(frozen=True)
class IKConfig:
    damping: float = 0.03
    max_joint_update: float = 0.08
    tolerance: float = 0.0005
    iterations: int = 40
    max_displacement: float = 0.05

    def __post_init__(self):
        if min(self.damping, self.max_joint_update, self.tolerance, self.max_displacement) <= 0 or self.iterations < 1:
            raise ValueError("IK bounds must be positive")


class CartesianController:
    def __init__(self, backend, body: str, joints: tuple[str, ...], actuators: tuple[str, ...], config=None):
        self.backend, self.config = backend, config or IKConfig()
        model = backend.model
        self.body = model.body(body).id
        self.joints = np.array([model.joint(name).id for name in joints])
        self.qpos = model.jnt_qposadr[self.joints]
        self.dofs = model.jnt_dofadr[self.joints]
        self.actuators = np.array([model.actuator(name).id for name in actuators])
        if len(joints) != len(actuators) or not len(joints):
            raise ValueError("one position actuator per joint required")
        if any(int(model.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)) for j in self.joints):
            raise ValueError("controller requires scalar joints")
        if any(model.actuator_trnid[a, 0] != j for a, j in zip(self.actuators, self.joints, strict=True)):
            raise ValueError("actuator/joint mapping mismatch")
        self.scratch = mujoco.MjData(model)

    def position(self):
        return tuple(float(x) for x in self.backend.data.xpos[self.body])

    def execute(self, displacement):
        delta = np.asarray(displacement, dtype=float)
        if delta.shape != (3,) or not np.isfinite(delta).all() or np.linalg.norm(delta) > self.config.max_displacement + 1e-12:
            raise ValueError("invalid or excessive Cartesian displacement")
        model, data = self.backend.model, self.backend.data
        target = np.asarray(self.position()) + delta
        copy_simulator_data(model, data, self.scratch)
        scratch = self.scratch
        jac = np.zeros((3, model.nv))
        for _ in range(self.config.iterations):
            mujoco.mj_forward(model, scratch)
            error = target - scratch.xpos[self.body]
            if np.linalg.norm(error) <= self.config.tolerance:
                break
            mujoco.mj_jacBody(model, scratch, jac, None, self.body)
            j = jac[:, self.dofs]
            dq = j.T @ np.linalg.solve(j @ j.T + self.config.damping**2 * np.eye(3), error)
            scratch.qpos[self.qpos] += np.clip(dq, -self.config.max_joint_update, self.config.max_joint_update)
            for joint, address in zip(self.joints, self.qpos, strict=True):
                if model.jnt_limited[joint]:
                    scratch.qpos[address] = np.clip(scratch.qpos[address], *model.jnt_range[joint])
        commands = scratch.qpos[self.qpos].copy()
        for index, actuator in enumerate(self.actuators):
            if model.actuator_ctrllimited[actuator]:
                commands[index] = np.clip(commands[index], *model.actuator_ctrlrange[actuator])
        data.ctrl[self.actuators] = commands
        self.backend.advance()

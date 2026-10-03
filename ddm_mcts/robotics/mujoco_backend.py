"""Optional MuJoCo world model; no rendering dependencies are initialized."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import mujoco
import numpy as np


@dataclass(frozen=True)
class PhysicsSnapshot:
    owner: str
    values: tuple[float, ...]


class MujocoBackend:
    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        xml: str | None = None,
        physics_steps: int = 150,
        gravity_compensation: bool = False,
        configure_spec=None,
    ):
        if physics_steps < 1 or (model_path is None) == (xml is None):
            raise ValueError("provide exactly one model path or XML and positive physics_steps")
        self.model = mujoco.MjModel.from_xml_string(xml) if xml is not None else mujoco.MjModel.from_xml_path(str(model_path))
        if gravity_compensation or configure_spec is not None:
            spec = mujoco.MjSpec.from_file(str(model_path)) if xml is None else mujoco.MjSpec.from_string(xml)
            if gravity_compensation:
                for body in spec.bodies:
                    if body.name != "world":
                        body.gravcomp = 1.0
            if configure_spec is not None:
                configure_spec(spec)
            self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.physics_steps = physics_steps
        self._step_callback: Callable[[], None] | None = None
        self._owner = uuid4().hex
        self.signature = mujoco.mjtState.mjSTATE_INTEGRATION
        self.reset()

    def reset(self, keyframe: str | None = None):
        if keyframe is None:
            mujoco.mj_resetData(self.model, self.data)
        else:
            key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
            if key < 0:
                raise ValueError(f"unknown keyframe: {keyframe}")
            mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        mujoco.mj_forward(self.model, self.data)

    def snapshot(self) -> PhysicsSnapshot:
        values = np.empty(mujoco.mj_stateSize(self.model, self.signature))
        mujoco.mj_getState(self.model, self.data, values, self.signature)
        return PhysicsSnapshot(self._owner, tuple(values))

    def restore(self, snapshot: PhysicsSnapshot):
        if snapshot.owner != self._owner:
            raise ValueError("snapshot belongs to a different backend")
        values = np.asarray(snapshot.values)
        if len(values) != mujoco.mj_stateSize(self.model, self.signature) or not np.isfinite(values).all():
            raise ValueError("invalid physics snapshot")
        mujoco.mj_setState(self.model, self.data, values, self.signature)
        mujoco.mj_forward(self.model, self.data)
        # Forward can update warmstarts; restore integration inputs after recomputation.
        mujoco.mj_setState(self.model, self.data, values, self.signature)

    @contextmanager
    def observe_steps(self, callback: Callable[[], None]):
        """Observe physics only inside an explicit execution scope, never search."""
        previous = self._step_callback

        def notify():
            if previous is not None:
                previous()
            callback()

        self._step_callback = notify
        try:
            yield
        finally:
            self._step_callback = previous

    def advance(self):
        if self._step_callback is None:
            mujoco.mj_step(self.model, self.data, nstep=self.physics_steps)
        else:
            for _ in range(self.physics_steps):
                mujoco.mj_step(self.model, self.data)
                self._step_callback()
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("non-finite simulator state")


def copy_simulator_data(model, source, destination):
    """Copy complete forward-dynamics inputs across supported Python bindings.

    Older MuJoCo Python releases do not expose mj_copyData. Integration state is
    sufficient for recomputing scratch kinematics/rendering; preserve warmstarts
    after forward, just as backend.restore does. No live source state is changed.
    """
    if hasattr(mujoco, "mj_copyData"):
        mujoco.mj_copyData(destination, model, source)
    else:
        signature = mujoco.mjtState.mjSTATE_INTEGRATION
        values = np.empty(mujoco.mj_stateSize(model, signature))
        mujoco.mj_getState(model, source, values, signature)
        mujoco.mj_setState(model, destination, values, signature)
        mujoco.mj_forward(model, destination)
        mujoco.mj_setState(model, destination, values, signature)

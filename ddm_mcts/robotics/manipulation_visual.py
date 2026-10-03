"""Pickup viewer reuses display-copy publication; no physics changes or search."""

import threading
import time
from contextlib import contextmanager

import mujoco
import numpy as np

from .mujoco_backend import copy_simulator_data
from .visual import VisualInspection


class PickupInspection(VisualInspection):
    def __init__(self, world, speed=1.0, *, launch=None):
        super().__init__(world, None, speed, launch=launch)

    def __enter__(self):
        if self._launch is None:
            from mujoco.viewer import launch_passive

            self._launch = launch_passive
        copy_simulator_data(self.model, self.world.backend.data, self.data)
        before = set(threading.enumerate())
        self.viewer = self._launch(self.model, self.data)
        self._viewer_threads = [t for t in threading.enumerate() if t not in before and t.daemon]
        try:
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = (0.4, 0, 0.35)
                self.viewer.cam.distance = 1.5
                self.viewer.cam.azimuth = 135
                self.viewer.cam.elevation = -25
                self.viewer.user_scn.ngeom = 1
                mujoco.mjv_initGeom(
                    self.viewer.user_scn.geoms[0],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.005, 0, 0]),
                    np.asarray(self.world.controller.position()),
                    np.eye(3).ravel(),
                    np.array([1, 0.2, 0.1, 0.5]),
                )
            self._publish()
            return self
        except BaseException:
            self._close()
            raise

    def target(self, position):
        with self.viewer.lock():
            self.viewer.user_scn.geoms[0].pos[:] = position
        self._publish()

    @contextmanager
    def execution_scope(self):
        deadline = time.perf_counter()
        next_frame = deadline

        def update():
            nonlocal deadline, next_frame
            self._check_open()
            now = time.perf_counter()
            if now >= next_frame:
                self._publish()
                next_frame = now + 1 / 60
            deadline += self.world.backend.model.opt.timestep / self.speed
            delay = deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

        with self.world.backend.observe_steps(update):
            yield
        self._publish()

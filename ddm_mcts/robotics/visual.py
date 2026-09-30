"""Optional observer: display copies isolate viewer events from planning physics."""

import copy
import threading
import time
from math import dist, isfinite

import mujoco
import numpy as np


def xyz(position):
    return "(" + ", ".join(f"{value:.4f}" for value in position) + ")"


class ViewerClosed(RuntimeError):
    """A user closed the viewer during an episode."""


class VisualInspection:
    def __init__(self, world, planner, speed=1.0, *, launch=None):
        if not isfinite(speed) or speed <= 0:
            raise ValueError("viewer speed must be finite and positive")
        self.world, self.planner, self.speed = world, planner, speed
        self.model = copy.copy(world.backend.model)
        self.data = mujoco.MjData(self.model)
        self._launch = launch
        self.viewer = None
        self._viewer_threads = []
        self._next_frame = 0.0

    def __enter__(self):
        if self._launch is None:
            from mujoco.viewer import launch_passive

            self._launch = launch_passive
        mujoco.mj_copyData(self.data, self.model, self.world.backend.data)
        existing_threads = set(threading.enumerate())
        self.viewer = self._launch(self.model, self.data)
        # Linux passive launch starts a daemon render thread. close() only requests
        # exit; let that thread finish before Python tears down native resources.
        self._viewer_threads = [thread for thread in threading.enumerate() if thread not in existing_threads and thread.daemon]
        try:
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = (0.35, 0, 0.4)
                self.viewer.cam.distance = 1.7
                self.viewer.cam.azimuth = 135
                self.viewer.cam.elevation = -25
                scene = self.viewer.user_scn
                scene.ngeom = 1
                mujoco.mjv_initGeom(
                    scene.geoms[0],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.012, 0, 0]),
                    np.asarray(self.world.task.goal),
                    np.eye(3).ravel(),
                    np.array([1.0, 0.2, 0.1, 0.8]),
                )
            self._publish()
            print(
                f"Panda Reach\nTarget: {xyz(self.world.task.goal)}\n"
                f"Policy: {type(self.planner.search.policy).__name__}\n"
                f"MCTS simulations: {self.planner.search.config.simulations}\n"
                f"Planning horizon: {self.planner.adapter.horizon}\n"
                f"Action step: {np.linalg.norm(self.world.actions[0].displacement):.3f} m\n"
                f"Execution speed: {self.speed:g}x",
                flush=True,
            )
            return self
        except BaseException:
            self._close()
            raise

    def _close(self):
        self.viewer.close()
        for thread in self._viewer_threads:
            thread.join(timeout=5)

    def __exit__(self, *exc):
        self._close()

    def _check_open(self):
        if not self.viewer.is_running():
            raise ViewerClosed("viewer closed during execution")

    def _publish(self):
        self._check_open()
        with self.viewer.lock():
            mujoco.mj_copyData(self.data, self.model, self.world.backend.data)
            # Recompute render positions on the display copy, not the live state.
            mujoco.mj_forward(self.model, self.data)
        self.viewer.sync()

    def before_plan(self, state, step):
        self._check_open()
        print(
            f"\nStep {step}\n  Position: {xyz(state.position)}\n  Distance: {dist(state.position, state.goal):.4f} m\n  Planning...",
            flush=True,
        )

    def after_plan(self, action, seconds):
        self._check_open()
        print(f"  Action: {action}\n  Plan time: {seconds:.3f} s", flush=True)

    def execute(self, world, action):
        # This scope surrounds ONLY selected live execution. Search never enters it.
        deadline = time.perf_counter()
        self._next_frame = deadline

        def on_step():
            nonlocal deadline
            self._check_open()
            now = time.perf_counter()
            if now >= self._next_frame:
                self._publish()
                self._next_frame = now + 1 / 60
            deadline += world.backend.model.opt.timestep / self.speed
            delay = deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

        with world.backend.observe_steps(on_step):
            return world.step(action)

    def after_step(self, state):
        self._publish()
        print(f"  New position: {xyz(state.position)}; error: {dist(state.position, state.goal):.4f} m", flush=True)

    def finish(self, state, summary):
        print(
            f"\n{'SUCCESS' if summary['success'] else 'TERMINATED (step limit)'}\n"
            f"Final position: {xyz(state.position)}\nTarget: {xyz(state.goal)}\n"
            f"Final error: {dist(state.position, state.goal):.4f} m\n"
            f"Actions executed: {summary['decisions']}\n"
            f"Total planning time: {summary['total_planning_seconds']:.2f} s\n"
            "Final scene remains open. Close the window or press Ctrl+C in the terminal.",
            flush=True,
        )

    def wait_until_closed(self):
        while self.viewer.is_running():
            # No live physics advances here; only the frozen final display is refreshed.
            try:
                self._publish()
            except ViewerClosed:
                break
            time.sleep(1 / 30)

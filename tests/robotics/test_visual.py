# ruff: noqa: E402
import json
import os
from contextlib import nullcontext
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")

from ddm_mcts.robotics.cli import build_parser, main
from ddm_mcts.robotics.core import RoboticsPlanner
from ddm_mcts.robotics.mujoco_backend import MujocoBackend
from ddm_mcts.robotics.reach import ReachTask, panda_reach
from ddm_mcts.robotics.run import run_episode
from ddm_mcts.robotics.visual import ViewerClosed, VisualInspection
from ddm_mcts.search.mcts import MCTSConfig


class FakeViewer:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.cam = mujoco.MjvCamera()
        self.user_scn = mujoco.MjvScene(model, maxgeom=10)
        self.running = True
        self.syncs = 0

    def lock(self):
        return nullcontext()

    def sync(self):
        self.syncs += 1
        # Simulate GUI physics edits: only the display copy may receive them.
        self.data.ctrl[:] = 0
        self.model.opt.gravity[:] = 0

    def is_running(self):
        return self.running

    def close(self):
        self.running = False


@pytest.fixture
def world():
    model = os.environ.get("PANDA_MODEL")
    if not model or not Path(model).is_file():
        pytest.skip("PANDA_MODEL required for fake-viewer Panda integration")
    return panda_reach(model, ReachTask((0.5945, 0.02, 0.6245)))


def test_parser():
    parser = build_parser()
    assert not parser.parse_args(["--target", "0", "0", "0"]).viewer
    args = parser.parse_args(["--target", "0", "0", "0", "--viewer", "--viewer-speed", ".5"])
    assert args.viewer and args.viewer_speed == 0.5
    with pytest.raises(SystemExit):
        main(["--target", "0", "0", "0", "--viewer-speed", "0"])


def test_step_callback_preserves_physics_and_cleanup():
    backend = MujocoBackend(
        xml='<mujoco><worldbody><body><joint type="slide"/><geom size=".1"/></body></worldbody></mujoco>', physics_steps=150
    )
    start = backend.snapshot()
    backend.advance()
    expected = backend.snapshot()
    backend.restore(start)
    times = []
    with backend.observe_steps(lambda: times.append(backend.data.time)):
        backend.advance()
    assert len(times) == 150
    assert backend.snapshot() == expected
    assert backend._step_callback is None
    with pytest.raises(RuntimeError), backend.observe_steps(lambda: (_ for _ in ()).throw(RuntimeError("stop"))):
        backend.advance()
    assert backend._step_callback is None


def test_search_invisible_marker_isolated_and_execution_observed(world, capsys):
    planner = RoboticsPlanner(world, config=MCTSConfig(10, seed=0))
    saved = world.snapshot()
    gravity = world.backend.model.opt.gravity.copy()
    with VisualInspection(world, planner, speed=10000, launch=FakeViewer) as observer:
        assert observer.model is not world.backend.model
        assert observer.data is not world.backend.data
        assert world.snapshot() == saved
        np.testing.assert_array_equal(world.backend.model.opt.gravity, gravity)
        marker = observer.viewer.user_scn.geoms[0]
        np.testing.assert_allclose(marker.pos, world.task.goal, atol=1e-7, rtol=0)
        assert marker.type == int(mujoco.mjtGeom.mjGEOM_SPHERE)
        syncs = observer.viewer.syncs
        result = planner.plan()
        assert observer.viewer.syncs == syncs
        assert world.snapshot() == saved
        state = observer.execute(world, result.action)
        observer.after_step(state)
        assert observer.viewer.syncs > syncs
        assert world.backend._step_callback is None
    assert not observer.viewer.running
    assert "UniformPolicy" in capsys.readouterr().out


def test_visual_episode_identical_to_headless(world, tmp_path, capsys):
    config = MCTSConfig(60, seed=0)
    headless = run_episode(world, RoboticsPlanner(world, config=config), tmp_path)
    expected = world.snapshot()
    world.reset()
    planner = RoboticsPlanner(world, config=config)
    with VisualInspection(world, planner, speed=10000, launch=FakeViewer) as observer:
        visual = run_episode(world, planner, tmp_path, observer=observer)
        assert world.snapshot() == expected
        frozen = world.snapshot()
        observer._publish()
        assert world.snapshot() == frozen

    def records(folder):
        return [
            (row["action"], row["state"], row["root_statistics"])
            for row in map(json.loads, (folder / "steps.jsonl").read_text().splitlines())
        ]

    assert records(headless) == records(visual)
    output = capsys.readouterr().out
    assert "Planning..." in output and "SUCCESS" in output and "Total planning time" in output
    assert json.loads((visual / "summary.json").read_text())["success"]


def test_cli_viewer_and_close(world, tmp_path, monkeypatch, capsys):
    import mujoco.viewer

    monkeypatch.setattr(mujoco.viewer, "launch_passive", FakeViewer)

    def close_final(self):
        assert self.world.task.is_success(self.world.get_state())
        self.viewer.close()

    monkeypatch.setattr(VisualInspection, "wait_until_closed", close_final)
    assert (
        main(
            [
                "--model",
                os.environ["PANDA_MODEL"],
                "--target",
                ".5945",
                ".02",
                ".6245",
                "--viewer",
                "--viewer-speed",
                "10000",
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert "SUCCESS" in capsys.readouterr().out


def test_early_close_retains_logs(world, tmp_path):
    planner = RoboticsPlanner(world, config=MCTSConfig(10, seed=0))
    with VisualInspection(world, planner, launch=FakeViewer) as observer:
        observer.viewer.close()
        with pytest.raises(ViewerClosed):
            run_episode(world, planner, tmp_path, observer=observer)
    summary = json.loads(next(tmp_path.glob("*/summary.json")).read_text())
    assert "ViewerClosed" in summary["error"]
    assert world.backend._step_callback is None


def test_render_thread_finishes_before_exit(world):
    import threading
    import time

    stopped = threading.Event()
    finished = threading.Event()

    class ThreadViewer(FakeViewer):
        def __init__(self, model, data):
            super().__init__(model, data)

            def render():
                stopped.wait()
                time.sleep(0.02)
                finished.set()

            threading.Thread(target=render, daemon=True).start()

        def close(self):
            super().close()
            stopped.set()

    with VisualInspection(world, RoboticsPlanner(world), launch=ThreadViewer):
        assert not finished.is_set()
    assert finished.is_set()


def test_ctrl_c_during_execution_retains_logs(world, tmp_path):
    planner = RoboticsPlanner(world, config=MCTSConfig(10, seed=0))
    with VisualInspection(world, planner, launch=FakeViewer) as observer:

        def interrupt(world, action):
            raise KeyboardInterrupt

        observer.execute = interrupt
        with pytest.raises(KeyboardInterrupt):
            run_episode(world, planner, tmp_path, observer=observer)
    summary = json.loads(next(tmp_path.glob("*/summary.json")).read_text())
    assert "KeyboardInterrupt" in summary["error"]
    assert not observer.viewer.running

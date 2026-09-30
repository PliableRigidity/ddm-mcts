# ruff: noqa: E402
import ast
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")

from ddm_mcts.policies.text_laya_policy import TextLayaPolicy
from ddm_mcts.robotics.cli import main, state_text
from ddm_mcts.robotics.core import RoboticsPlanner
from ddm_mcts.robotics.mujoco_backend import MujocoBackend
from ddm_mcts.robotics.reach import ReachTask, cartesian_actions, panda_reach
from ddm_mcts.robotics.run import run_episode
from ddm_mcts.search.mcts import MCTSConfig

XML = '<mujoco><worldbody><body><joint type="slide"/><geom size=".1" mass="1"/></body></worldbody><actuator><motor joint="0"/></actuator></mujoco>'
XML = XML.replace('<joint type="slide"/>', '<joint name="j" type="slide"/>').replace('joint="0"', 'joint="j"')


def test_snapshot_full_state_branching():
    backend = MujocoBackend(xml=XML, physics_steps=10)
    backend.data.ctrl[:] = 0.2
    backend.data.qfrc_applied[:] = 0.3
    saved = backend.snapshot()
    backend.advance()
    first = backend.snapshot()
    for _ in range(3):
        backend.restore(saved)
        assert backend.snapshot() == saved
        backend.advance()
        assert backend.snapshot() == first
    backend.restore(saved)
    backend.data.ctrl[:] = -1
    backend.advance()
    assert backend.snapshot() != first
    backend.restore(saved)
    backend.advance()
    assert backend.snapshot() == first
    with pytest.raises(ValueError):
        MujocoBackend(xml=XML).restore(saved)
    backend.reset()
    assert backend.data.time == 0


@pytest.fixture
def model():
    path = os.environ.get("PANDA_MODEL")
    if not path or not Path(path).is_file():
        pytest.skip("PANDA_MODEL must point to external Menagerie Panda XML")
    return path


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_panda_actions_bounds_and_restore(model, axis):
    world = panda_reach(model, ReachTask((0.8, 0.2, 0.5)))
    saved = world.snapshot()
    initial = np.asarray(world.get_state().position)
    action = world.actions[axis * 2]
    state = world.step(action)
    assert state.position[axis] > initial[axis] + 0.005
    model_data = world.backend.model
    assert np.all(world.backend.data.ctrl >= model_data.actuator_ctrlrange[:, 0])
    assert np.all(world.backend.data.ctrl <= model_data.actuator_ctrlrange[:, 1])
    assert np.all(np.abs(np.asarray(state.position) - initial) < 0.04)
    for _ in range(2):
        world.restore(saved)
        assert world.step(action) == state
    with pytest.raises(ValueError):
        world.controller.execute((1, 0, 0))
    world.reset()
    assert world.snapshot() == saved


@pytest.mark.parametrize(
    "offset,planar,guided",
    [
        ((0.04, 0.02, 0), False, False),
        ((-0.04, 0, 0.02), False, False),
        ((0, -0.04, -0.02), False, False),
        ((0.04, -0.02, 0), True, False),
        ((0.04, 0.02, -0.02), False, True),
    ],
)
def test_reach_planning_logging(model, tmp_path, offset, planar, guided):
    world = panda_reach(model, ReachTask((0.8, 0.2, 0.5)), axes=(0, 1) if planar else (0, 1, 2))
    initial = world.get_state()
    goal = tuple(x + dx for x, dx in zip(initial.position, offset, strict=True))
    world.task = ReachTask(goal, epsilon=0.012, max_steps=15)
    policy = None
    if guided:

        def predictor(text, question):
            assert "target XYZ" in text
            position = ast.literal_eval(text.split("Hand XYZ: ")[1].split(";")[0])
            target = ast.literal_eval(text.split("target XYZ: ")[1].split(";")[0])
            probabilities = {}
            for key, label in question["action"]["criteria"].items():
                action = next(action for action in world.actions if action.name == label)
                error = np.asarray(target) - position
                probabilities[key] = 10.0 if np.dot(error, action.displacement) > 0 else 1.0
            return {"answers": {"action": {"probabilities": probabilities}}}

        policy = TextLayaPolicy(state_text, objective="Reach target", predictor=predictor)
    planner = RoboticsPlanner(world, policy, MCTSConfig(60, seed=0), horizon=3)
    saved = world.snapshot()
    planner.plan()
    assert world.snapshot() == saved
    folder = run_episode(world, planner, tmp_path)
    assert world.task.is_success(world.get_state())
    assert (folder / "steps.jsonl").stat().st_size > 0
    assert json.loads((folder / "summary.json").read_text())["success"]
    if guided:
        assert policy.calls > 0
    world.reset()
    world.task = ReachTask(initial.position)
    assert world.legal_actions() == ()
    assert planner.search.policy is not None


def test_cli(model, tmp_path):
    world = panda_reach(model, ReachTask((0.8, 0.2, 0.5)))
    goal = tuple(x + 0.02 if i == 0 else x for i, x in enumerate(world.get_state().position))
    assert main(["--model", model, "--target", *map(str, goal), "--output", str(tmp_path)]) == 0


def test_action_validation():
    assert len(cartesian_actions()) == 6
    assert len(cartesian_actions(axes=(0, 1))) == 4
    with pytest.raises(ValueError):
        cartesian_actions(-1)
    with pytest.raises(ValueError):
        ReachTask((0, 0, 0), epsilon=0)


def test_policy_swap_and_log_errors(model, tmp_path):
    world = panda_reach(model, ReachTask((0.5945, 0.02, 0.6245)))
    saved = world.snapshot()
    uniform = RoboticsPlanner(world, config=MCTSConfig(10, seed=0))
    uniform.plan()
    mock = TextLayaPolicy(
        state_text, objective="Reach", predictor=lambda text, q: {"answers": {"action": {"probabilities": {"option_0": 1.0}}}}
    )
    guided = RoboticsPlanner(world, mock, MCTSConfig(10, seed=0))
    result = guided.plan()
    assert result.root.children[world.actions[0]].prior == 1
    assert world.snapshot() == saved
    assert mock.calls > 0
    with pytest.raises(ValueError, match="historical"):
        run_episode(world, uniform, Path(__file__).resolve().parents[2] / "results")

    class BrokenPlanner:
        search = uniform.search
        adapter = uniform.adapter

        def plan(self):
            raise RuntimeError("injected policy failure")

    with pytest.raises(RuntimeError, match="injected"):
        run_episode(world, BrokenPlanner(), tmp_path)
    summaries = list(tmp_path.glob("*/summary.json"))
    assert len(summaries) == 1
    assert "injected policy failure" in json.loads(summaries[0].read_text())["error"]

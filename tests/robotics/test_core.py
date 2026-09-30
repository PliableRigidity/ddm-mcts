from dataclasses import dataclass

import pytest

from ddm_mcts.robotics.core import RoboticsEnvironment, RoboticsPlanner, SimulatorAdapter, Task
from ddm_mcts.search.mcts import MCTSConfig


@dataclass(frozen=True)
class LineTask(Task):
    def evaluate(self, state):
        return -abs(2 - state)

    def is_success(self, state):
        return state == 2

    def is_terminal(self, state):
        return self.is_success(state) or abs(state) >= 5


class LineWorld(RoboticsEnvironment):
    task = LineTask()

    def __init__(self):
        self.value = 0

    def get_state(self):
        return self.value

    def legal_actions(self):
        return () if self.task.is_terminal(self.value) else (-1, 1)

    def step(self, action):
        if action not in self.legal_actions():
            raise ValueError("illegal")
        self.value += action
        return self.value

    def snapshot(self):
        return self.value

    def restore(self, snapshot):
        self.value = snapshot

    def reset(self):
        self.value = 0
        return self.value


def test_generic_search_and_isolation():
    world = LineWorld()
    adapter = SimulatorAdapter(world)
    root = adapter.initial_state()
    assert adapter.step(root, 1).observation == 1
    assert adapter.step(root, -1).observation == -1
    assert world.value == 0
    planner = RoboticsPlanner(world, config=MCTSConfig(40, seed=1))
    assert planner.plan().action == 1
    assert world.value == 0
    with pytest.raises(ValueError):
        adapter.step(root, 100)
    assert world.value == 0
    assert world.reset() == 0


def test_task_and_horizon():
    world = LineWorld()
    assert world.task.evaluate(0) == -2
    assert world.task.is_success(2)
    adapter = SimulatorAdapter(world, 1)
    child = adapter.step(adapter.initial_state(), 1)
    assert adapter.is_terminal(child)
    assert adapter.legal_actions(child) == ()
    with pytest.raises(ValueError):
        SimulatorAdapter(world, 0)


def test_generic_run_logs(tmp_path):
    import json

    from ddm_mcts.robotics.run import run_episode

    world = LineWorld()
    planner = RoboticsPlanner(world, config=MCTSConfig(40, seed=1))
    folder = run_episode(world, planner, tmp_path)
    assert json.loads((folder / "summary.json").read_text())["success"]
    assert len((folder / "steps.jsonl").read_text().splitlines()) == 2

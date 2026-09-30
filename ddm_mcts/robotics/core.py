"""Simulator-independent contracts and bridge to the existing search core."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from ddm_mcts.environments.base import Environment
from ddm_mcts.search.mcts import MCTS, MCTSConfig

S = TypeVar("S")
A = TypeVar("A")
B = TypeVar("B")


class Task(ABC, Generic[S]):
    @abstractmethod
    def evaluate(self, state: S) -> float: ...

    @abstractmethod
    def is_success(self, state: S) -> bool: ...

    @abstractmethod
    def is_terminal(self, state: S) -> bool: ...


class RoboticsEnvironment(ABC, Generic[S, A, B]):
    """Mutable world. Snapshots must include controller/task runtime state.

    Instances are sequential and not thread safe. Use one world per worker.
    """

    task: Task[S]

    @abstractmethod
    def get_state(self) -> S: ...

    @abstractmethod
    def legal_actions(self) -> tuple[A, ...]: ...

    @abstractmethod
    def step(self, action: A) -> S: ...

    @abstractmethod
    def snapshot(self) -> B: ...

    @abstractmethod
    def restore(self, snapshot: B) -> None: ...

    @abstractmethod
    def reset(self) -> S: ...


@dataclass(frozen=True)
class SearchState(Generic[S, B]):
    observation: S
    snapshot: B
    depth: int = 0


class SimulatorAdapter(Environment):
    """Present mutable physics as pure transitions, preserving the live world."""

    def __init__(self, world: RoboticsEnvironment, horizon: int = 3):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.world, self.horizon = world, horizon
        self.state_projection = None

    def initial_state(self):
        state = self.world.get_state()
        observation = self.state_projection(state) if self.state_projection is not None else state
        return SearchState(observation, self.world.snapshot())

    def legal_actions(self, state):
        if self.is_terminal(state):
            return ()
        saved = self.world.snapshot()
        try:
            self.world.restore(state.snapshot)
            return self.world.legal_actions()
        finally:
            self.world.restore(saved)

    def step(self, state, action):
        if self.is_terminal(state):
            raise ValueError("cannot step terminal search state")
        saved = self.world.snapshot()
        try:
            self.world.restore(state.snapshot)
            observation = self.world.step(action)
            if self.state_projection is not None:
                observation = self.state_projection(observation)
            return SearchState(observation, self.world.snapshot(), state.depth + 1)
        finally:
            self.world.restore(saved)

    def is_terminal(self, state):
        return state.depth >= self.horizon or self.world.task.is_terminal(state.observation)

    def get_reward(self, state, player):
        return self.world.task.evaluate(state.observation)

    def current_player(self, state):
        return 1

    def render(self, state):
        return str(state.observation)


class RoboticsPlanner:
    """Receding-horizon MCTS; planning never executes the selected action."""

    def __init__(self, world, policy=None, config: MCTSConfig | None = None, horizon: int = 3):
        self.adapter = SimulatorAdapter(world, horizon)
        self.search = MCTS(self.adapter, policy, config, evaluator=lambda state, player, rng: world.task.evaluate(state.observation))

    def plan(self, observation=None, *, state_projection=None):
        """Optional perceived root and predicted-observation projection.

        Physics still starts from the current simulator snapshot. No image stepping
        or privileged entity positions are implied by this projection boundary.
        """
        from dataclasses import replace

        previous = self.adapter.state_projection
        self.adapter.state_projection = state_projection
        try:
            root = self.adapter.initial_state()
            if observation is not None:
                root = replace(root, observation=observation)
            return self.search.search(root)
        finally:
            self.adapter.state_projection = previous

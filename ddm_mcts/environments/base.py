from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable
from typing import Generic, TypeVar

StateT = TypeVar("StateT", bound=Hashable)
ActionT = TypeVar("ActionT", bound=Hashable)


class Environment(ABC, Generic[StateT, ActionT]):
    """Deterministic, two-player, zero-sum environment."""

    @abstractmethod
    def initial_state(self) -> StateT: ...

    @abstractmethod
    def legal_actions(self, state: StateT) -> tuple[ActionT, ...]: ...

    @abstractmethod
    def step(self, state: StateT, action: ActionT) -> StateT: ...

    @abstractmethod
    def is_terminal(self, state: StateT) -> bool: ...

    @abstractmethod
    def get_reward(self, state: StateT, player: int) -> float:
        """Return terminal utility in [-1, 1] for player, or 0 otherwise."""

    @abstractmethod
    def current_player(self, state: StateT) -> int: ...

    @abstractmethod
    def render(self, state: StateT) -> str: ...

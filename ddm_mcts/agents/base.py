from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass(slots=True)
class Decision(Generic[ActionT]):
    action: ActionT
    simulations: int = 0
    statistics: list[dict[str, object]] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


class Agent(ABC, Generic[StateT, ActionT]):
    name: str

    @abstractmethod
    def decide(self, state: StateT) -> Decision[ActionT]: ...

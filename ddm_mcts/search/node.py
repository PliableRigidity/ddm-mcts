from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass(slots=True)
class Node(Generic[StateT, ActionT]):
    state: StateT
    parent: Node[StateT, ActionT] | None = None
    action: ActionT | None = None
    prior: float = 1.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[ActionT, Node[StateT, ActionT]] = field(default_factory=dict)
    expanded: bool = False

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

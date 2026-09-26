from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping, Sequence
from typing import Generic, TypeVar

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT", bound=Hashable)


def normalize_probabilities(
    probabilities: Mapping[ActionT, float], legal_actions: Sequence[ActionT]
) -> dict[ActionT, float]:
    """Filter illegal/non-finite weights and return a legal normalized distribution."""
    legal = tuple(legal_actions)
    if not legal:
        return {}
    cleaned: dict[ActionT, float] = {}
    for action in legal:
        try:
            value = float(probabilities.get(action, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        cleaned[action] = value if value >= 0.0 and value < float("inf") else 0.0
    total = sum(cleaned.values())
    if total <= 0.0:
        return {action: 1.0 / len(legal) for action in legal}
    return {action: value / total for action, value in cleaned.items()}


class Policy(ABC, Generic[StateT, ActionT]):
    @abstractmethod
    def probabilities(
        self, state: StateT, legal_actions: Sequence[ActionT]
    ) -> dict[ActionT, float]: ...

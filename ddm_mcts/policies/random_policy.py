from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Generic, TypeVar

from .base import Policy, normalize_probabilities

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


class UniformPolicy(Policy[StateT, ActionT], Generic[StateT, ActionT]):
    def probabilities(self, state: StateT, legal_actions: Sequence[ActionT]) -> dict[ActionT, float]:
        del state
        return normalize_probabilities({}, legal_actions)


class RandomPolicy(Policy[StateT, ActionT], Generic[StateT, ActionT]):
    """A seeded random prior, useful as a baseline policy-only agent."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def probabilities(self, state: StateT, legal_actions: Sequence[ActionT]) -> dict[ActionT, float]:
        del state
        return normalize_probabilities({a: self._rng.random() for a in legal_actions}, legal_actions)

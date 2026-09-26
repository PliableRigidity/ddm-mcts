from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Generic, TypeVar

from .base import Policy, normalize_probabilities

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT", bound=Hashable)


class MixedPolicy(Policy[StateT, ActionT], Generic[StateT, ActionT]):
    """Blend a learned policy with uniform legal-action probability.

    ``alpha`` controls trust in the learned prior. It is independent of MCTS's
    ``c_puct``, which controls the strength of exploration during tree search.
    """

    def __init__(self, policy: Policy[StateT, ActionT], alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.policy, self.alpha = policy, alpha

    def probabilities(self, state: StateT, legal_actions: Sequence[ActionT]) -> dict[ActionT, float]:
        legal = tuple(legal_actions)
        if not legal:
            return {}
        uniform = 1.0 / len(legal)
        if self.alpha == 0.0:
            return {action: uniform for action in legal}
        learned = normalize_probabilities(self.policy.probabilities(state, legal), legal)
        mixed = {action: (1.0 - self.alpha) * uniform + self.alpha * learned[action] for action in legal}
        return normalize_probabilities(mixed, legal)
